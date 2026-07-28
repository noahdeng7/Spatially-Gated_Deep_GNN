"""
02_population.py -- municipal population at both ends of the migration window.

Input  : INEGI Censo 2020 tabulados (FULL COUNT municipal totals, not the sample)
         CONAPO municipal population projections (for 2015)
Output : interim/population.parquet
         reports/population_coverage.md

WHY TWO VINTAGES
----------------
Origin population belongs at the START of the migration window. The stock at
risk of emigrating between 2015 and 2020 is the 2015 population, not the 2020
one -- and the 2020 origin population is mechanically *depleted* by the very
flow being modelled, so using it puts the dependent variable on both sides of
the equation. That endogeneity is the reason `population.origin_pop_year`
defaults to 2015.

Destination population is 2020 by the mirror-image argument: it is the stock
that received the flow.

Both columns are always emitted (`pop_orig_2015`, `pop_orig_2020`). The config
flag only decides which one the canonical `pop_orig` aliases, so switching the
assumption is a config change and a re-run of 07, not a re-extraction.

The full count is used rather than the sample because these are denominators.
Sampling error in a denominator propagates into every per-capita covariate.

Idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CVEGEO_DTYPE, PipelineError, ROOT, assert_cvegeo_valid, cvegeo_series,
    ensure_dirs, get_logger, load_config, log_rows, missingness_table,
    read_csv_smart, report_path, rel, resolve, run_step, write_interim,
)
from geo import apply_crosswalk, load_crosswalk  # noqa: E402

STEP = "02_population"

# Column-name candidates. INEGI/CONAPO tabulados are published in several
# shapes; rather than hardcode one, probe for these and fail loudly with the
# real header if none match.
CVEGEO_CANDIDATES = ["CVEGEO", "cvegeo", "CVE_GEO", "CLAVE", "cve_geo"]
ENT_CANDIDATES = ["CVE_ENT", "ENTIDAD", "cve_ent", "ENT", "clave_entidad"]
MUN_CANDIDATES = ["CVE_MUN", "MUNICIPIO", "cve_mun", "MUN", "clave_municipio"]
POP_CANDIDATES = ["POBTOT", "POB_TOTAL", "poblacion", "POBLACION", "pob_total", "TOTAL",
                  "POB"]   # POB: CONAPO base_municipios_final_datos_{01,02}.csv
# Year column. CONAPO files ship the accented form; the unaccented and
# mojibake variants are here because the same file re-exported through a
# latin-1/utf-8 mismatch is common enough to be worth tolerating.
YEAR_CANDIDATES = ["AÑO", "Año", "ANIO", "ANO", "ANIO_REGISTRO",
                   "year", "YEAR", "anio", "ano"]


def _strip_bom(name) -> str:
    """
    Remove a leading byte-order mark from a column name, in either encoding form.

    U+FEFF is the BOM decoded as UTF-8; "\\xef\\xbb\\xbf" is the same three bytes
    decoded as latin-1, which is how this pipeline reads INEGI CSVs.
    """
    s = str(name)
    for bom in ("﻿", "\xef\xbb\xbf", "ï»¿"):
        if s.startswith(bom):
            s = s[len(bom):]
    return s.strip()


def _pick(cols, candidates, what, path):
    for c in candidates:
        if c in cols:
            return c
    raise PipelineError(
        f"\ncould not find a {what} column in {Path(path).name}\n"
        f"  tried: {candidates}\n"
        f"  header actually present: {sorted(map(str, cols))}\n\n"
        "Add the real column name to the *_CANDIDATES list in src/02_population.py, "
        "or re-export the tabulado with standard column names."
    )


def _read_keyed(path: Path, log) -> pd.DataFrame:
    """Read a municipal table and attach a clean 5-char cvegeo."""
    if not path.exists():
        raise PipelineError(
            f"not found: {rel(path)}\n"
            "See config.yaml -> downloads for what to fetch."
        )
    # dtype=str throughout: these files carry zero-padded codes.
    df = read_csv_smart(path, log, dtype=str, low_memory=False)

    # Strip the UTF-8 BOM and stray whitespace from header names. INEGI's ITER
    # export carries a BOM on its first column, so an exact-match lookup for
    # "ENTIDAD" finds nothing and the state code appears to be missing.
    #
    # Two forms are handled because it depends how the file was decoded:
    #   U+FEFF        when read as UTF-8
    #   "ï»¿"         the same three bytes (EF BB BF) decoded as latin-1,
    #                 which is how this pipeline reads INEGI CSVs
    renamed = {c: _strip_bom(c) for c in df.columns}
    n_fixed = sum(1 for k, v in renamed.items() if k != v)
    if n_fixed:
        log.info("      cleaned %d column name(s) (BOM/whitespace)", n_fixed)
    df = df.rename(columns=renamed)

    log.info("READ  %-38s rows=%s cols=%d", path.name, f"{len(df):,}", df.shape[1])

    cols = list(df.columns)
    try:
        key = _pick(cols, CVEGEO_CANDIDATES, "CVEGEO", path)
        df["cvegeo"] = df[key].astype(CVEGEO_DTYPE).str.strip().str.zfill(5)
    except PipelineError:
        ent = _pick(cols, ENT_CANDIDATES, "state code", path)
        mun = _pick(cols, MUN_CANDIDATES, "municipality code", path)
        log.info("      composing cvegeo from %s + %s", ent, mun)
        df["cvegeo"] = cvegeo_series(df[ent], df[mun])
    return df


def _collapse_locality_dimension(df: pd.DataFrame, log) -> pd.DataFrame:
    """
    Reduce a locality-level table (ITER) to one row per municipality.

    INEGI's ITER file is published at LOCALITY level and interleaves aggregate
    rows at every level of the hierarchy:

        MUN=000, LOC=0000   national / state totals
        MUN!=000, LOC=0000  MUNICIPAL totals      <-- the rows we want
        LOC!=0000           individual localities

    Measured on the real 2020 file: 195,662 rows, of which only 2,469 are
    municipal totals. Those 2,469 sum to exactly 126,014,024, the published
    national population. Summing every row instead gives 505,248,533 -- four
    times Mexico's population, because each person is counted once in their
    locality, again in their municipal total, again in their state total and
    again nationally.

    A quadruple-count of every denominator in the panel would not look obviously
    wrong in a regression table, which is precisely why this filter is explicit
    and logged rather than folded into a groupby.
    """
    if "LOC" not in df.columns:
        return df

    n_in = len(df)
    # The municipal-total row is the one whose locality code is all zeros.
    is_muni_total = df["LOC"].str.strip().str.zfill(4).eq("0000")
    out = df[is_muni_total].copy()
    log.info("      ITER is locality-level: keeping only municipal-total rows "
             "(LOC == 0000)")
    log_rows(log, "collapse locality dimension", n_in, len(out))
    if out.empty:
        raise PipelineError(
            "no rows with LOC == 0000 -- cannot identify municipal totals. "
            "Check the locality column in this tabulado."
        )
    return out.drop(columns=[c for c in ("LOC", "NOM_LOC") if c in out.columns])


def _drop_aggregate_rows(df: pd.DataFrame, log) -> pd.DataFrame:
    """
    Drop state and national aggregate rows, which carry municipality code 000.

    Runs after _collapse_locality_dimension so both levels of double-counting
    are removed.
    """
    n_in = len(df)
    is_state_total = df["cvegeo"].str.endswith("000")
    if bool(is_state_total.any()):
        log.info("      dropping %d state/national-total row(s) "
                 "(municipality code 000)", int(is_state_total.sum()))
    out = df[~is_state_total].copy()
    log_rows(log, "drop state/national aggregate rows", n_in, len(out))
    return out


def load_census_2020(cfg, log) -> pd.DataFrame:
    path = resolve(cfg, cfg["population"]["census_2020_tabulado"])
    df = _read_keyed(path, log)
    df = _collapse_locality_dimension(df, log)

    pop_col = _pick(list(df.columns), POP_CANDIDATES, "population", path)
    out = df[["cvegeo", pop_col]].rename(columns={pop_col: "pop_2020"})
    out["pop_2020"] = pd.to_numeric(out["pop_2020"], errors="coerce")
    out = _drop_aggregate_rows(out, log)

    bad = out["pop_2020"].isna()
    if bool(bad.any()):
        log.warning("%d municipality row(s) have non-numeric 2020 population; "
                    "left as NA and reported in coverage, not zero-filled",
                    int(bad.sum()))

    # Sanity check against the published national total. A hierarchy-collapse
    # bug shows up here as a multiple of the true figure, which is the single
    # most likely way this step goes wrong.
    total = float(out["pop_2020"].sum())
    log.info("      municipal rows=%s  population sum=%s",
             f"{len(out):,}", f"{total:,.0f}")
    if not (1.15e8 <= total <= 1.35e8):
        raise PipelineError(
            f"municipal population sums to {total:,.0f}, outside the plausible "
            "range for Mexico in 2020 (~126 million).\n"
            "A total that is a MULTIPLE of ~126M means aggregate rows survived "
            "the hierarchy collapse and people are being counted at more than "
            "one level. A total far below it means municipal rows were dropped."
        )
    return out


def load_conapo_2015(cfg, log) -> pd.DataFrame:
    """
    CONAPO projections are a long panel (one row per municipality per year, often
    also per sex). Filter to the target year and total across any sex dimension.
    """
    path = resolve(cfg, cfg["population"]["conapo_projections"])
    df = _read_keyed(path, log)

    target = cfg["project"]["census_year"] - cfg["project"]["migration_window_years"]
    year_col = next((c for c in YEAR_CANDIDATES if c in df.columns), None)
    if year_col is None:
        raise PipelineError(
            f"\nCONAPO file has no year column (tried {YEAR_CANDIDATES}).\n"
            f"  header: {sorted(map(str, df.columns))}\n"
            "The projections file is a long panel; without a year column we cannot "
            "isolate the window-start population."
        )

    n_in = len(df)
    df = df[pd.to_numeric(df[year_col], errors="coerce") == target]
    log_rows(log, f"filter CONAPO to year {target}", n_in, len(df))
    if df.empty:
        raise PipelineError(
            f"CONAPO file contains no rows for year {target}. "
            f"Years present: {sorted(pd.to_numeric(df[year_col], errors='coerce').dropna().unique())[:20]}"
        )

    pop_col = _pick(list(df.columns), POP_CANDIDATES, "population", path)
    df["_pop"] = pd.to_numeric(df[pop_col], errors="coerce")

    # Sum across any residual dimension (sex, age group) to a municipal total.
    out = df.groupby("cvegeo", as_index=False, observed=True)["_pop"].sum()
    out = out.rename(columns={"_pop": f"pop_{target}"})
    log_rows(log, "collapse CONAPO to municipal totals", len(df), len(out))
    out = _drop_aggregate_rows(out, log)
    return out


def try_load_conapo_2015(cfg, log) -> pd.DataFrame | None:
    """
    Load the CONAPO 2015 series, or explain precisely why we cannot.

    CONAPO is a MANUAL download (the published link 404s and the agency's newer
    domain fails TLS). Whether its absence is fatal depends on configuration:

      origin_pop_year == 2015 -> FATAL. The configured analytic choice cannot be
                                 honoured, and silently substituting the 2020
                                 figure would reintroduce the depletion
                                 endogeneity the 2015 default exists to avoid.
      origin_pop_year == 2020 -> a warning. pop_2015 is emitted as NA; nothing
                                 the panel actually uses is missing.
    """
    target = cfg["project"]["census_year"] - cfg["project"]["migration_window_years"]
    path = resolve(cfg, cfg["population"]["conapo_projections"])
    wanted = int(cfg["population"]["origin_pop_year"]) == target

    if path.exists():
        return load_conapo_2015(cfg, log)

    if wanted:
        raise PipelineError(
            f"CONAPO projections not found: {rel(path)}\n\n"
            f"population.origin_pop_year is {target}, so this file is REQUIRED --\n"
            "it is the only source for municipal population at the start of the\n"
            "migration window.\n\n"
            "CONAPO is a manual download; run `python src/00_download.py --only "
            "conapo_projections`\nfor instructions (the published link 404s and "
            "the new domain fails TLS verification).\n\n"
            "OR, if CONAPO proves unobtainable, set population.origin_pop_year to "
            f"{cfg['project']['census_year']}.\n"
            "That is a real analytic cost, not a free substitution: the 2020 "
            "origin population is\nmechanically depleted by the very outflow being "
            "modelled, which puts the dependent\nvariable on both sides of the "
            "equation. Say so in your write-up if you do it."
        )

    log.warning("CONAPO projections not found at %s", rel(path))
    log.warning("  population.origin_pop_year is %s, so the %d series is not "
                "required. pop_%d will be emitted as NA.",
                cfg["population"]["origin_pop_year"], target, target)
    return None


def harmonize_and_merge(p2020: pd.DataFrame, p2015: pd.DataFrame | None,
                        cfg, log) -> pd.DataFrame:
    """
    Apply the boundary crosswalk, then SUM child populations into the parent.

    Summing is the right aggregation here and it is worth being explicit about
    why: population is extensive (it adds across a partition of space), unlike
    the intensive covariates later in the pipeline (GDP per capita, temperature)
    which must be population-weighted rather than summed.
    """
    xwalk = load_crosswalk(cfg, log)
    strategy = cfg["geometry"]["crosswalk_strategy"]
    target = cfg["project"]["census_year"] - cfg["project"]["migration_window_years"]

    inputs = [("2020", p2020)] + ([(str(target), p2015)] if p2015 is not None else [])
    frames = {}
    for name, df in inputs:
        d = df.copy()
        n_in = len(d)
        d["cvegeo"] = apply_crosswalk(d["cvegeo"], xwalk, strategy, log, f"population {name}")
        valcol = [c for c in d.columns if c != "cvegeo"][0]
        d = d.groupby("cvegeo", as_index=False, observed=True)[valcol].sum(min_count=1)
        log_rows(log, f"aggregate population {name} to parent", n_in, len(d))
        frames[name] = d

    if p2015 is None:
        merged = frames["2020"].copy()
        # Emit the column so the schema is stable whether or not CONAPO was
        # available -- downstream code should never have to branch on it.
        merged[f"pop_{target}"] = pd.NA
        log.warning("      pop_%d emitted as all-NA (CONAPO unavailable)", target)
        return merged

    merged = frames["2020"].merge(frames[str(target)], on="cvegeo",
                                  how="outer", indicator=True)

    only_2020 = int((merged["_merge"] == "left_only").sum())
    only_2015 = int((merged["_merge"] == "right_only").sum())
    if only_2020 or only_2015:
        log.warning("population vintages do not align: %d municipality/ies only in "
                    "2020, %d only in 2015. OUTER joined and reported -- not "
                    "inner-joined away.", only_2020, only_2015)
    merged = merged.drop(columns="_merge")
    return merged


def main() -> int:
    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(STEP, cfg)
    log.info("=" * 78)
    log.info("STEP %s", STEP)
    log.info("=" * 78)

    year_start = cfg["project"]["census_year"] - cfg["project"]["migration_window_years"]

    p2020 = load_census_2020(cfg, log)
    p2015 = try_load_conapo_2015(cfg, log)
    pop = harmonize_and_merge(p2020, p2015, cfg, log)

    assert_cvegeo_valid(pop["cvegeo"], "population.cvegeo")
    if bool(pop["cvegeo"].duplicated().any()):
        raise PipelineError("duplicate cvegeo in population table after aggregation")

    # --- canonical alias, driven by config -----------------------------------
    origin_year = int(cfg["population"]["origin_pop_year"])
    dest_year = int(cfg["population"]["dest_pop_year"])
    for y in (origin_year, dest_year):
        if f"pop_{y}" not in pop.columns:
            raise PipelineError(
                f"config asks for population year {y} but only "
                f"{[c for c in pop.columns if c.startswith('pop_')]} were built."
            )

    pop["pop_origin_canonical"] = pop[f"pop_{origin_year}"]
    pop["pop_dest_canonical"] = pop[f"pop_{dest_year}"]
    pop.attrs["origin_pop_year"] = origin_year
    pop.attrs["dest_pop_year"] = dest_year

    log.info("CHOICE origin population vintage = %d  (%s)", origin_year,
             "start of migration window -- avoids the depletion endogeneity"
             if origin_year == year_start else
             "END of window -- note pop_orig is mechanically depleted by the "
             "outflow being modelled; see README")
    log.info("CHOICE dest population vintage   = %d", dest_year)

    # --- sanity --------------------------------------------------------------
    for col in [c for c in pop.columns if c.startswith("pop_")]:
        neg = pop[col] < 0
        if bool(neg.any()):
            raise PipelineError(f"{col}: {int(neg.sum())} negative value(s)")
        log.info("      %-24s n=%s  sum=%s  min=%s  max=%s",
                 col, f"{int(pop[col].notna().sum()):,}",
                 f"{pop[col].sum():,.0f}", f"{pop[col].min():,.0f}"
                 if pop[col].notna().any() else "NA",
                 f"{pop[col].max():,.0f}" if pop[col].notna().any() else "NA")

    miss = missingness_table(pop, log)

    lines = [
        "# Population coverage",
        "",
        f"Municipalities: **{len(pop):,}**",
        "",
        f"- Origin population vintage in use: **{origin_year}**",
        f"- Destination population vintage in use: **{dest_year}**",
        "",
        "Both vintages are stored regardless; `population.origin_pop_year` in",
        "config.yaml selects which one `pop_orig` aliases in the final panel.",
        "",
        "## Missingness",
        "",
        "| column | dtype | missing | % |",
        "|---|---|---:|---:|",
    ]
    for _, r in miss.iterrows():
        lines.append(f"| `{r['column']}` | {r['dtype']} | {r['n_missing']:,} | {r['pct_missing']:.3f}% |")
    p = report_path(cfg, "population_coverage.md")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("WROTE %s", rel(p))

    write_interim(cfg, pop, "population", log)
    log_rows(log, "STEP TOTAL 02_population", len(p2020), len(pop))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_step(main, STEP))



