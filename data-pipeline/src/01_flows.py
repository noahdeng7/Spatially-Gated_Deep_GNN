"""
01_flows.py -- build origin-destination migration flows from INEGI microdata.

This is the spine of the panel. Everything else attaches to the O-D keys it
produces.

Input  : INEGI Censo 2020 person records. Which instrument is read is set by
         flows.use_basic_questionnaire:
           true  -> Cuestionario Basico, COMPLETE ENUMERATION, no weights
           false -> Cuestionario Ampliado, ~10% sample, FACTOR-weighted
Output : interim/flows_dyadic.parquet     off-diagonal domestic O-D pairs
         interim/flows_diagonal.parquet   origin == destination (kept, not deleted)
         interim/flows_excluded.parquet   foreign + not-specified origins
         interim/municipios_universe.parquet  harmonized municipal universe
         reports/flows_reconciliation.md  where every person record went

DESIGN NOTES
------------
* Every person record is accounted for in exactly one bucket, and the buckets sum
  back to the input. The reconciliation report proves it. Records are never
  dropped -- they are routed and counted.

* Non-migrants (origin == destination) go to their own file rather than being
  filtered away, so the diagonal is available later.

* International origins and "no especificado" codes become explicit categories.
  They are excluded from the dyadic panel because they are not municipality-to-
  municipality flows, but they are counted, because "how many people came from
  abroad" is a real number a reader will ask about.

* WEIGHTING depends on the instrument.
    - Full count: every record has weight exactly 1.0, so `migrants` is a
      literal headcount. The pipeline asserts migrants == migrants_unweighted;
      if that ever breaks, a weight leaked in and the column is no longer a
      count.
    - Sample: the FACTOR expansion factor scales counts to the population, and
      the unweighted count is retained alongside so small-cell noise stays
      inspectable -- a weighted flow of 400 built from 3 sampled people is a
      very different object from one built from 40.

  The full count also removes a large share of SAMPLING zeros. In a 10% sample
  a dyad with a true flow of 4 people has roughly a 66% chance of contributing
  no records at all and appearing as a zero indistinguishable from a structural
  one. Those dyads become the positive flows they actually are.

* Variable names are ASSERTED, not assumed. If a configured name is absent the
  script prints the actual header and stops. INEGI renames things between rounds
  and a silently-missing column would produce a plausible-looking empty panel.

Idempotent: re-running overwrites its outputs and reads nothing it wrote.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CVEGEO_DTYPE, PipelineError, ROOT, assert_cvegeo_valid, cvegeo_series,
    ensure_dirs, get_logger, load_config, log_rows, read_csv_smart, report_path,
    rel, resolve, run_step, write_interim,
)
from geo import apply_crosswalk, audit_codes, load_crosswalk, load_municipios  # noqa: E402

STEP = "01_flows"

# Origin classification buckets. Every record lands in exactly one.
BUCKET_DOMESTIC = "domestic"
BUCKET_NON_MIGRANT = "non_migrant"
BUCKET_FOREIGN = "foreign"
BUCKET_NOT_SPECIFIED = "not_specified"
BUCKET_UNDER_AGE = "under_min_age"
BUCKET_BAD_CODE = "unparseable_code"


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

def is_full_count(cfg) -> bool:
    """
    True when flows are built from the Cuestionario Basico (complete enumeration).

    Confirmed against INEGI's study description for the 2020 census: the Basic
    Questionnaire was an exhaustive enumeration of all inhabited dwellings, and
    the released microdata covers the entire enumerated population. It therefore
    carries NO expansion factor -- FACTOR exists only in the sample tables
    (Viviendas_CA, Personas_CA, Migrantes).
    """
    return bool(cfg["flows"].get("use_basic_questionnaire", False))


def _microdata_files(cfg, log) -> list[Path]:
    fcfg = cfg["flows"]
    full = is_full_count(cfg)

    if fcfg.get("microdata_is_sharded"):
        pattern = fcfg["microdata_glob"]
        matched = sorted(ROOT.glob(pattern))
        instrument = ("Cuestionario BASICO (full count)" if full
                      else "Cuestionario AMPLIADO (10% sample)")

        if not matched:
            raise PipelineError(
                f"no microdata files matched {pattern!r} under {ROOT}.\n"
                f"Expecting the {instrument} person tables.\n"
                "See config.yaml -> downloads.census_microdata for how to fetch "
                "them (the INEGI portal is JS-rendered, so this is a manual "
                "download)."
            )

        # Separate the two instruments by the "_CA" filename marker, then keep
        # only the one we asked for. Reading both would double-count every
        # sampled dwelling and inflate every flow -- so this split is not
        # optional, and whatever is excluded is REPORTED rather than dropped
        # quietly.
        ca = [f for f in matched if "_CA" in f.name.upper()]
        cb = [f for f in matched if "_CA" not in f.name.upper()]
        files, excluded = (cb, ca) if full else (ca, cb)

        if excluded:
            log.warning("excluding %d file(s) from the other instrument: %s%s",
                        len(excluded), [f.name for f in excluded[:4]],
                        " ..." if len(excluded) > 4 else "")
            log.warning("  Reading both the full count and the sample together "
                        "would double-count every sampled dwelling.")

        if not files:
            raise PipelineError(
                f"{pattern!r} matched {len(matched)} file(s), but none of them are "
                f"{instrument} tables.\n"
                f"  found: {[f.name for f in matched[:8]]}\n\n"
                + ("Full-count mode expects Cuestionario Basico tables (Personas*.csv "
                   "WITHOUT a _CA suffix). Either fetch them, or set "
                   "flows.use_basic_questionnaire: false to use the sample you have."
                   if full else
                   "Sample mode expects Cuestionario Ampliado tables (Personas_CA*.csv). "
                   "Either fetch them, or set flows.use_basic_questionnaire: true to "
                   "use the full count you have.")
            )

        log.info("microdata: %d shard(s), instrument = %s", len(files), instrument)
        return files

    key = "microdata_file" if full else "microdata_file_sample"
    path = resolve(cfg, fcfg.get(key, fcfg["microdata_file"]))
    if not path.exists():
        raise PipelineError(f"microdata not found: {path}")
    return [path]


def _required_vars(cfg) -> dict[str, str]:
    """
    Variables to read from the microdata.

    FACTOR is dropped in full-count mode: the Cuestionario Basico has no weight
    column, so asking for it would fail the header assertion on a file that is
    perfectly correct.
    """
    v = dict(cfg["flows"]["vars"])
    if cfg["flows"].get("min_age") is None:
        v.pop("age", None)
    if is_full_count(cfg):
        v.pop("factor", None)
    return v


def read_microdata(cfg, log) -> pd.DataFrame:
    """
    Read person records, keeping only the columns we need, all as strings.

    Reading geographic codes as `string` at the CSV boundary is the single most
    important line in this file. pandas will happily infer ENT as int64 and turn
    "01" into 1, and every downstream join then quietly misses Aguascalientes.
    """
    files = _microdata_files(cfg, log)
    want = _required_vars(cfg)
    usecols = list(want.values())

    frames = []
    total_raw = 0
    for i, path in enumerate(files):
        header = read_csv_smart(path, nrows=0)
        missing = [c for c in usecols if c not in header.columns]
        if missing:
            raise PipelineError(
                f"\n{path.name} is missing configured variable(s): {missing}\n"
                f"Columns actually present ({len(header.columns)}):\n"
                f"  {sorted(header.columns.tolist())}\n\n"
                "Fix config.yaml -> flows.vars to match the codebook shipped with "
                "this census round. Variable naming shifts between rounds; "
                "ENT_PAIS_RES_5A in particular has changed form historically."
            )

        df = read_csv_smart(
            path,
            usecols=usecols,
            dtype={c: CVEGEO_DTYPE for c in usecols},
            low_memory=False,
        )
        total_raw += len(df)
        frames.append(df)
        if (i + 1) % 8 == 0 or i == len(files) - 1:
            log.info("  read %d/%d shards, %s records so far",
                     i + 1, len(files), f"{total_raw:,}")

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    # invert the name map so downstream code uses canonical names
    df = df.rename(columns={v: k for k, v in want.items()})
    log.info("ROWS  %-38s  in=%-12s out=%-12s", "read microdata (all shards)",
             f"{total_raw:,}", f"{len(df):,}")
    return df


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def classify_origin(df: pd.DataFrame, cfg, log) -> pd.DataFrame:
    """
    Assign every record to exactly one origin bucket.

    Order matters: age restriction first (the question was not asked), then
    not-specified, then foreign, then non-migrant, then domestic.
    """
    sc = cfg["flows"]["special_codes"]
    bucket = pd.Series(BUCKET_DOMESTIC, index=df.index, dtype="object")

    # -- age: the 5-year question is only meaningful for people who existed --
    min_age = cfg["flows"].get("min_age")
    if min_age is not None:
        age = pd.to_numeric(df["age"], errors="coerce")
        under = age.notna() & (age < float(min_age))
        bucket[under] = BUCKET_UNDER_AGE
        log.info("  bucket %-18s %s (age < %s)", BUCKET_UNDER_AGE,
                 f"{int(under.sum()):,}", min_age)

    ent = df["ent_res_5a"].astype(CVEGEO_DTYPE).str.strip()
    mun = df["mun_res_5a"].astype(CVEGEO_DTYPE).str.strip()
    free = bucket == BUCKET_DOMESTIC

    # -- not specified --------------------------------------------------------
    ns_ent = {str(x).strip() for x in (sc.get("not_specified_ent") or [])}
    ns_mun = {str(x).strip() for x in (sc.get("not_specified_mun") or [])}
    is_ns = (
        ent.isin(ns_ent) | mun.isin(ns_mun)
        # a padded form of the same sentinel
        | ent.str.zfill(2).isin({s.zfill(2) for s in ns_ent})
        | mun.str.zfill(3).isin({s.zfill(3) for s in ns_mun})
    ).fillna(False)
    bucket[free & is_ns] = BUCKET_NOT_SPECIFIED

    # -- foreign --------------------------------------------------------------
    free = bucket == BUCKET_DOMESTIC
    ent_num = pd.to_numeric(ent, errors="coerce")
    fmin = sc.get("foreign_ent_min")
    is_foreign = (ent_num.notna() & (ent_num >= float(fmin))) if fmin else pd.Series(False, index=df.index)
    bucket[free & is_foreign.fillna(False)] = BUCKET_FOREIGN

    # -- non-migrant sentinel (some rounds code stayers rather than blanking) --
    free = bucket == BUCKET_DOMESTIC
    nm = {str(x).strip() for x in (sc.get("non_migrant_ent") or [])}
    is_nm_code = (ent.isin(nm) | ent.str.zfill(2).isin({s.zfill(2) for s in nm})).fillna(False)
    # Blank residence-5-years-ago also means "did not move" in the 2020 file.
    is_blank = ent.isna() | (ent == "")
    bucket[free & (is_nm_code | is_blank)] = BUCKET_NON_MIGRANT

    df = df.copy()
    df["origin_bucket"] = bucket

    for b, n in df["origin_bucket"].value_counts().items():
        log.info("  bucket %-18s %s", b, f"{n:,}")
    return df


# ---------------------------------------------------------------------------
# build keys
# ---------------------------------------------------------------------------

def build_keys(df: pd.DataFrame, cfg, log) -> pd.DataFrame:
    """Construct 5-char CVEGEO for current residence and 2015 residence."""
    df = df.copy()
    df["dest"] = cvegeo_series(df["ent_current"], df["mun_current"])

    # Origin codes are only meaningful for records that named a Mexican
    # municipality. Everything else stays NA by construction.
    parseable = df["origin_bucket"].isin([BUCKET_DOMESTIC])
    orig = pd.Series(pd.NA, index=df.index, dtype=CVEGEO_DTYPE)
    # ENT_PAIS_RES_5A is a 3-digit field ("001".."032" for states; >= 33 is a
    # country and already routed to the foreign bucket). cvegeo_series refuses
    # any code wider than 2 digits rather than truncate, so canonicalize the
    # domestic subset through a numeric round-trip -- explicit, and a value
    # that is not a genuine 1..99 state code still fails loudly there.
    ent_domestic = (
        pd.to_numeric(df.loc[parseable, "ent_res_5a"], errors="coerce")
        .astype("Int64").astype("string").str.zfill(2)
    )
    orig.loc[parseable] = cvegeo_series(
        ent_domestic, df.loc[parseable, "mun_res_5a"]
    )
    df["orig"] = orig

    # Destination must always parse -- it is the census enumeration geography.
    bad_dest = df["dest"].isna()
    if bool(bad_dest.any()):
        raise PipelineError(
            f"{int(bad_dest.sum()):,} record(s) have an unparseable CURRENT "
            "municipality. That should be impossible in census microdata -- "
            "check flows.vars.ent_current / mun_current against the codebook."
        )

    # A domestic-bucket record whose origin failed to parse is a real data
    # problem, not a category. Re-bucket it so it is counted, not silently NA.
    bad_orig = parseable & df["orig"].isna()
    if bool(bad_orig.any()):
        log.warning("%s domestic record(s) had an unparseable ORIGIN code; "
                    "re-bucketed to %s", f"{int(bad_orig.sum()):,}", BUCKET_BAD_CODE)
        df.loc[bad_orig, "origin_bucket"] = BUCKET_BAD_CODE

    return df


def harmonize(df: pd.DataFrame, cfg, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map 2015-vintage codes onto the 2020 municipal universe.

    Applied to BOTH orig and dest so the panel is balanced -- if a split child is
    folded into its parent on one side only, the row count and the diagonal both
    go wrong.
    """
    xwalk = load_crosswalk(cfg, log)
    strategy = cfg["geometry"]["crosswalk_strategy"]

    df = df.copy()
    df["orig"] = apply_crosswalk(df["orig"], xwalk, strategy, log, "orig (2015 residence)")
    df["dest"] = apply_crosswalk(df["dest"], xwalk, strategy, log, "dest (2020 residence)")
    return df, xwalk


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def record_weights(df: pd.DataFrame, cfg, log) -> pd.Series:
    """
    Per-record weight.

    Full count  -> exactly 1.0 for every person. The Cuestionario Basico is a
                   complete enumeration; there is nothing to expand, and
                   `migrants` is then a literal headcount rather than an
                   estimate.
    Sample      -> the FACTOR expansion factor, which must be present and
                   numeric for every record.
    """
    if is_full_count(cfg):
        log.info("      weights = 1.0 per record (full count -- no expansion)")
        return pd.Series(1.0, index=df.index, dtype="float64")

    factor = pd.to_numeric(df["factor"], errors="coerce")
    n_bad = int(factor.isna().sum())
    if n_bad:
        raise PipelineError(
            f"{n_bad:,} record(s) have a non-numeric expansion FACTOR. "
            "Refusing to treat these as zero-weight -- that would silently shrink "
            "the population total. Inspect flows.vars.factor.\n"
            "If these are Cuestionario Basico files, set "
            "flows.use_basic_questionnaire: true -- the CB has no FACTOR column."
        )
    return factor.astype("float64")


def aggregate_od(df: pd.DataFrame, cfg, log, label: str) -> pd.DataFrame:
    """
    Collapse person records to O-D cells.

    Emits the weighted count and the raw record count. On the full count these
    are identical by construction, which is itself a useful invariant -- the
    pipeline asserts it below.
    """
    factor = record_weights(df, cfg, log)

    work = pd.DataFrame({
        "orig": df["orig"].astype(CVEGEO_DTYPE),
        "dest": df["dest"].astype(CVEGEO_DTYPE),
        "w": factor.astype("float64"),
    })

    n_in = len(work)
    out = (
        work.groupby(["orig", "dest"], dropna=False, observed=True)
        .agg(migrants=("w", "sum"), migrants_unweighted=("w", "size"))
        .reset_index()
    )
    out["migrants_unweighted"] = out["migrants_unweighted"].astype("int64")
    log_rows(log, f"aggregate to O-D cells [{label}]", n_in, len(out))
    log.info("      %s: weighted total=%s  unweighted total=%s  cells=%s",
             label, f"{out['migrants'].sum():,.0f}",
             f"{out['migrants_unweighted'].sum():,}", f"{len(out):,}")

    if is_full_count(cfg):
        # On a complete enumeration every weight is 1, so the weighted total
        # must equal the record count exactly. If it does not, a weight got in
        # from somewhere and `migrants` is no longer a headcount.
        diff = float((out["migrants"] - out["migrants_unweighted"]).abs().max())
        if diff > 1e-9:
            raise PipelineError(
                f"full-count mode: weighted and unweighted counts disagree by up "
                f"to {diff:g} in {label}. Every weight should be exactly 1.0. "
                "Something applied an expansion factor."
            )
        log.info("      full-count invariant holds: migrants == migrants_unweighted")
    return out


# ---------------------------------------------------------------------------
# reconciliation
# ---------------------------------------------------------------------------

def write_reconciliation(cfg, df: pd.DataFrame, parts: dict[str, pd.DataFrame], log) -> Path:
    """
    Prove that every input record is accounted for.

    A reader should be able to add up the buckets and get back the number of
    person records that went in. If they cannot, this pipeline lost data
    somewhere and the panel is wrong.
    """
    factor = record_weights(df, cfg, log).fillna(0.0)
    tab = (
        pd.DataFrame({"origin_bucket": df["origin_bucket"], "w": factor})
        .groupby("origin_bucket", observed=True)
        .agg(records=("w", "size"), weighted=("w", "sum"))
        .reset_index()
        .sort_values("records", ascending=False, ignore_index=True)
    )
    tab["pct_records"] = (100.0 * tab["records"] / len(df)).round(3)

    full = is_full_count(cfg)
    lines = [
        "# Flow construction reconciliation",
        "",
        f"Census round: **{cfg['project']['census_year']}** "
        f"(migration window {cfg['project']['census_year'] - cfg['project']['migration_window_years']}"
        f"-{cfg['project']['census_year']})",
        "",
        f"Instrument: **{'Cuestionario Basico -- COMPLETE ENUMERATION' if full else 'Cuestionario Ampliado -- ~10% sample'}**",
        "",
        f"Person records read: **{len(df):,}**",
        (f"Population represented: **{factor.sum():,.0f}** "
         "(equal to the record count -- every weight is 1.0)" if full else
         f"Weighted population represented: **{factor.sum():,.0f}**"),
        "",]
    if full:
        lines += [
            "Because this is a complete enumeration, `migrants` is a literal",
            "headcount rather than a weighted estimate, and `migrants_unweighted`",
            "is identical to it. There is no sampling error and no small-cell",
            "noise to inspect.",
            "",
        ]
    lines += [
        "## Where every record went",
        "",
        "Each person record falls into exactly one bucket. The buckets sum to the",
        "input. Nothing is dropped; records not eligible for the dyadic panel are",
        "excluded from it *and counted here*.",
        "",
        "| bucket | records | % of records | weighted persons | in dyadic panel? |",
        "|---|---:|---:|---:|---|",
    ]
    in_panel = {
        BUCKET_DOMESTIC: "yes (off-diagonal)",
        BUCKET_NON_MIGRANT: "no -- diagonal, written separately",
        BUCKET_FOREIGN: "no -- not a municipal origin",
        BUCKET_NOT_SPECIFIED: "no -- origin unknown",
        BUCKET_UNDER_AGE: "no -- question not applicable",
        BUCKET_BAD_CODE: "no -- code failed to parse (INVESTIGATE)",
    }
    for _, r in tab.iterrows():
        lines.append(
            f"| `{r['origin_bucket']}` | {r['records']:,} | {r['pct_records']:.3f}% | "
            f"{r['weighted']:,.0f} | {in_panel.get(r['origin_bucket'], '?')} |"
        )
    lines += [
        f"| **total** | **{tab['records'].sum():,}** | **100.000%** | "
        f"**{tab['weighted'].sum():,.0f}** | |",
        "",
    ]

    checksum_ok = int(tab["records"].sum()) == len(df)
    lines += [
        f"Checksum: buckets sum to input records -- **{'PASS' if checksum_ok else 'FAIL'}**",
        "",
        "## Output tables",
        "",
        "| table | rows (O-D cells) | weighted migrants |",
        "|---|---:|---:|",
    ]
    for name, part in parts.items():
        wt = part["migrants"].sum() if "migrants" in part else float("nan")
        lines.append(f"| `{name}` | {len(part):,} | {wt:,.0f} |")

    lines += [
        "",
        "## Reconciliation against INEGI published tabulados",
        "",
    ]
    published = (cfg.get("validation") or {}).get("published_internal_migrants_2015_2020")
    dyadic = parts.get("flows_dyadic")
    if published and dyadic is not None:
        ours = float(dyadic["migrants"].sum())
        disc = (ours - float(published)) / float(published)
        lines += [
            f"- Published national internal migrants (5-year): **{float(published):,.0f}**",
            f"- This pipeline, weighted off-diagonal total: **{ours:,.0f}**",
            f"- Discrepancy: **{disc:+.3%}**",
            "",
            "A discrepancy of a few tenths of a percent is expected from sample",
            "design and from the boundary harmonization folding split children",
            "into parents. A discrepancy of several percent means a bucket is",
            "misclassified -- check the sentinel codes in `flows.special_codes`.",
        ]
    else:
        lines += [
            "- **NOT RUN.** `validation.published_internal_migrants_2015_2020` is null.",
            "",
            "[TODO] Read the national 5-year internal migrant total off the INEGI",
            "  migration tabulado (Censo 2020, migracion interna, residencia 5 anos",
            "  antes) and set it in config.yaml. Until then this pipeline has no",
            "  external check on its headline number.",
        ]

    path = report_path(cfg, "flows_reconciliation.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("WROTE reconciliation report -> %s", rel(path))
    if not checksum_ok:
        raise PipelineError("bucket checksum failed: records were lost during classification")
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(STEP, cfg)
    log.info("=" * 78)
    log.info("STEP %s", STEP)
    log.info("=" * 78)

    raw = read_microdata(cfg, log)
    n_input = len(raw)

    df = classify_origin(raw, cfg, log)
    df = build_keys(df, cfg, log)
    df, _xwalk = harmonize(df, cfg, log)

    # --- split into buckets --------------------------------------------------
    domestic = df[df["origin_bucket"] == BUCKET_DOMESTIC]
    log_rows(log, "select domestic-origin records", len(df), len(domestic))

    od = aggregate_od(domestic, cfg, log, "domestic")

    diagonal = od[od["orig"] == od["dest"]].reset_index(drop=True)
    dyadic = od[od["orig"] != od["dest"]].reset_index(drop=True)
    log_rows(log, "split diagonal off (kept, not deleted)", len(od), len(dyadic))
    log.info("      diagonal retained separately: %s cell(s), %s weighted stayers",
             f"{len(diagonal):,}", f"{diagonal['migrants'].sum():,.0f}")

    # Non-migrants identified by sentinel/blank rather than by orig==dest also
    # belong on the diagonal. Their destination IS their origin by definition.
    nm = df[df["origin_bucket"] == BUCKET_NON_MIGRANT]
    if len(nm):
        nm2 = nm.copy()
        nm2["orig"] = nm2["dest"]
        nm_od = aggregate_od(nm2, cfg, log, "non_migrant (sentinel/blank)")
        diagonal = (
            pd.concat([diagonal, nm_od], ignore_index=True)
            .groupby(["orig", "dest"], as_index=False, observed=True)
            .agg(migrants=("migrants", "sum"),
                 migrants_unweighted=("migrants_unweighted", "sum"))
        )
        log.info("      diagonal after folding in sentinel non-migrants: %s cell(s)",
                 f"{len(diagonal):,}")

    # Excluded categories, counted rather than deleted.
    excluded_mask = df["origin_bucket"].isin(
        [BUCKET_FOREIGN, BUCKET_NOT_SPECIFIED, BUCKET_UNDER_AGE, BUCKET_BAD_CODE]
    )
    exc = df[excluded_mask]
    factor_exc = record_weights(exc, cfg, log).fillna(0.0)
    excluded = (
        pd.DataFrame({
            "origin_category": exc["origin_bucket"],
            "dest": exc["dest"].astype(CVEGEO_DTYPE),
            "w": factor_exc,
        })
        .groupby(["origin_category", "dest"], as_index=False, observed=True)
        .agg(migrants=("w", "sum"), migrants_unweighted=("w", "size"))
    )
    log.info("      excluded categories: %s cell(s), %s weighted persons",
             f"{len(excluded):,}", f"{excluded['migrants'].sum():,.0f}")

    # --- audit codes against 2020 geometry -----------------------------------
    try:
        muni = load_municipios(cfg, log)
        # Audit against the RAW geometry codes: an omission in the crosswalk
        # shows up here as an orphan, loudly.
        obs = pd.concat([dyadic["orig"], dyadic["dest"]], ignore_index=True)
        audit = audit_codes(obs, muni["cvegeo"], "flows (orig+dest)", log)
        if len(audit):
            p = report_path(cfg, "code_audit_flows.csv")
            audit.to_csv(p, index=False)
            log.info("WROTE code audit -> %s", rel(p))
        # The universe the panel is built on must be HARMONIZED (as the
        # docstring promises): crosswalk children folded into parents, same
        # code set the flows themselves were remapped onto. Writing the raw
        # geometry codes here put 11 post-census child codes into the spine
        # as all-zero rows with NA covariates -- discovered when the
        # harmonized geometry export refused to join against the panel.
        harmonized = apply_crosswalk(muni["cvegeo"], _xwalk,
                                     cfg["geometry"]["crosswalk_strategy"],
                                     log, "universe")
        universe = pd.DataFrame(
            {"cvegeo": pd.Series(sorted(harmonized.dropna().unique()),
                                 dtype=CVEGEO_DTYPE)}
        )
        log.info("      universe: %d raw geometry codes -> %d harmonized",
                 len(muni), len(universe))
        write_interim(cfg, universe, "municipios_universe", log)
    except PipelineError as exc:
        log.warning("code audit SKIPPED -- geometry unavailable: %s", exc)
        log.warning("  Orphan detection is deferred to 07_assemble.py, which will "
                    "fail hard rather than inner-join them away.")

    # --- validate and write --------------------------------------------------
    assert_cvegeo_valid(dyadic["orig"], "flows.orig")
    assert_cvegeo_valid(dyadic["dest"], "flows.dest")
    dup = dyadic.duplicated(subset=["orig", "dest"])
    if bool(dup.any()):
        raise PipelineError(f"{int(dup.sum())} duplicate (orig,dest) key(s) after aggregation")

    write_interim(cfg, dyadic, "flows_dyadic", log)
    write_interim(cfg, diagonal, "flows_diagonal", log)
    write_interim(cfg, excluded, "flows_excluded", log)

    write_reconciliation(cfg, df, {
        "flows_dyadic": dyadic,
        "flows_diagonal": diagonal,
        "flows_excluded": excluded,
    }, log)

    log_rows(log, "STEP TOTAL 01_flows", n_input, len(dyadic))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_step(main, STEP))



