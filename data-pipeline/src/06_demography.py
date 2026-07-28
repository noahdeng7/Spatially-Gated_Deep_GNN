"""
06_demography.py -- youth share of the working-age population, by municipality.

Input  : INEGI Censo 2020 age-by-municipality tabulado (preferred, full count)
         or the weighted microdata as a fallback
Output : interim/demography.parquet
         reports/demography_bands.md

THE VARIABLE
------------
    dempres_orig = P[youth_min..youth_max] / P[working_min..working_max]

Default bands (config `demography.*`): 15-29 over 15-64.

A NOTE ON THE SPECIFICATION
---------------------------
The variable as originally specified reads "youth share of working-age
population, origin COUNTRY". For an internal-migration panel that is a
degenerate quantity: every origin is in Mexico, so a country-level measure is a
constant and drops out under any fixed effect -- it cannot identify anything.

It is therefore built at the **origin municipality** level, which is the level
at which it actually varies and the level the rest of this panel is keyed on.
This is a deliberate deviation from the literal specification and it is recorded
in the codebook as well as here.

AGE BANDS ARE CONFIGURABLE, NOT HARDCODED
-----------------------------------------
The literature does not agree: 15-29/15-64 (UN youth), 15-24/15-64 (ILO),
20-34/15-64 (peak migration propensity). The choice materially moves the
variable, so it lives in config and the chosen bands are stamped into the output
and the codebook. The raw numerator and denominator are ALSO emitted so the
ratio can be reconstructed or redefined without re-running the extraction.

Idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CVEGEO_DTYPE, PipelineError, ROOT, assert_cvegeo_valid, cvegeo_series,
    ensure_dirs, get_logger, load_config, log_rows, missingness_table,
    read_csv_smart, report_path, rel, resolve, run_step, write_interim,
)
from geo import apply_crosswalk, load_crosswalk  # noqa: E402

STEP = "06_demography"

AGE_CANDIDATES = ["EDAD", "edad", "AGE", "GRUPO_EDAD", "grupo_edad"]
POP_CANDIDATES = ["POBTOT", "POB", "poblacion", "POBLACION", "VALOR", "valor", "TOTAL"]


def _bands(cfg) -> tuple[int, int, int, int]:
    d = cfg["demography"]
    ymin, ymax = int(d["youth_min_age"]), int(d["youth_max_age"])
    wmin, wmax = int(d["working_min_age"]), int(d["working_max_age"])
    if not (wmin <= ymin <= ymax <= wmax):
        raise PipelineError(
            f"incoherent age bands: youth [{ymin}, {ymax}] is not contained in "
            f"working-age [{wmin}, {wmax}]. The ratio would not be a share."
        )
    return ymin, ymax, wmin, wmax


# ---------------------------------------------------------------------------
# ITER five-year age bands
# ---------------------------------------------------------------------------
# INEGI's ITER file carries banded age counts at FULL COUNT. The bands below are
# the non-overlapping 5-year series. ITER also ships overlapping convenience
# bands (P_15A17, P_18A24, P_0A2 ...) which are deliberately EXCLUDED here --
# mixing them into a sum would double-count the overlapping years.
ITER_AGE_BANDS: dict[str, tuple[int, int]] = {
    "P_0A4": (0, 4), "P_5A9": (5, 9), "P_10A14": (10, 14),
    "P_15A19": (15, 19), "P_20A24": (20, 24), "P_25A29": (25, 29),
    "P_30A34": (30, 34), "P_35A39": (35, 39), "P_40A44": (40, 44),
    "P_45A49": (45, 49), "P_50A54": (50, 54), "P_55A59": (55, 59),
    "P_60A64": (60, 64), "P_65A69": (65, 69), "P_70A74": (70, 74),
    "P_75A79": (75, 79), "P_80A84": (80, 84),
}
# Open-ended top band, usable only when the requested range is open-ended too.
ITER_TOP_BAND = ("P_85YMAS", 85)
# ITER's own 15-64 aggregate, used as an independent cross-check on the sum.
ITER_WORKING_AGE_TOTAL = "POB15_64"


def _bands_covering(lo: int, hi: int) -> list[str] | None:
    """
    Columns whose 5-year bands tile [lo, hi] exactly.

    Returns None when the requested range does not align with band boundaries --
    e.g. 16-30 cuts across P_15A19 and P_30A34, and there is no honest way to
    split a band without assuming a within-band age distribution. The caller
    then falls back to the microdata, where single-year ages make any range
    exact.
    """
    chosen = [c for c, (b_lo, b_hi) in ITER_AGE_BANDS.items()
              if b_lo >= lo and b_hi <= hi]
    if not chosen:
        return None
    covered = sorted((ITER_AGE_BANDS[c] for c in chosen), key=lambda t: t[0])
    # contiguous, and spanning exactly the requested range
    if covered[0][0] != lo or covered[-1][1] != hi:
        return None
    for (_, prev_hi), (next_lo, _) in zip(covered, covered[1:]):
        if next_lo != prev_hi + 1:
            return None
    return chosen


def load_from_iter_bands(cfg, log) -> pd.DataFrame | None:
    """
    Build the youth share from ITER's banded age counts, at FULL COUNT.

    Preferred over the microdata when the configured bands align with ITER's
    5-year boundaries, which they do for every common convention:

        15-29 (UN youth)   = P_15A19 + P_20A24 + P_25A29
        15-24 (ILO)        = P_15A19 + P_20A24
        20-34 (peak)       = P_20A24 + P_25A29 + P_30A34
        15-64 (working)    = P_15A19 ... P_60A64

    Returns None if the file is unavailable or the bands do not align, so the
    caller can fall through to the microdata.

    Returns a numerator/denominator frame directly rather than a long age table,
    because banded counts cannot be exploded to single years without inventing a
    within-band distribution.
    """
    path = resolve(cfg, cfg["demography"]["age_tabulado"])
    if not path.exists():
        log.info("ITER age file not present at %s", rel(path))
        return None

    ymin, ymax, wmin, wmax = _bands(cfg)
    youth_cols = _bands_covering(ymin, ymax)
    work_cols = _bands_covering(wmin, wmax)

    if youth_cols is None or work_cols is None:
        log.warning("configured age bands (youth %d-%d, working %d-%d) do not "
                    "align with ITER's 5-year boundaries; falling back to "
                    "microdata where single-year ages make any range exact",
                    ymin, ymax, wmin, wmax)
        return None

    df = read_csv_smart(path, log, dtype=str, low_memory=False)
    df = df.rename(columns={c: str(c).lstrip("﻿\xef\xbb\xbf").strip()
                            for c in df.columns})

    needed = set(youth_cols) | set(work_cols)
    missing = sorted(needed - set(df.columns))
    if missing:
        log.warning("ITER file lacks expected age columns %s; falling back", missing)
        return None

    # Municipal totals only. ITER is locality-level and interleaves aggregates
    # at every level -- see the note in 02_population._collapse_locality_dimension.
    if "LOC" in df.columns:
        n_in = len(df)
        df = df[df["LOC"].str.strip().str.zfill(4).eq("0000")]
        log_rows(log, "ITER: keep municipal-total rows (LOC == 0000)", n_in, len(df))

    if "CVEGEO" in df.columns:
        df["cvegeo"] = df["CVEGEO"].astype(CVEGEO_DTYPE).str.strip().str.zfill(5)
    else:
        ent = next((c for c in ("ENTIDAD", "CVE_ENT", "cve_ent") if c in df.columns), None)
        mun = next((c for c in ("MUN", "CVE_MUN", "cve_mun") if c in df.columns), None)
        if not (ent and mun):
            log.warning("ITER file has no usable geographic key; falling back")
            return None
        df["cvegeo"] = cvegeo_series(df[ent], df[mun])

    df = df[~df["cvegeo"].isna() & ~df["cvegeo"].str.endswith("000")]

    num = np.zeros(len(df), dtype="float64")
    for c in youth_cols:
        num += pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()
    den = np.zeros(len(df), dtype="float64")
    for c in work_cols:
        den += pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()

    out = pd.DataFrame({"cvegeo": df["cvegeo"].to_numpy(),
                        "pop_youth": num, "pop_working_age": den})
    out["cvegeo"] = out["cvegeo"].astype(CVEGEO_DTYPE)

    log.info("      youth  %d-%d from %s", ymin, ymax, sorted(youth_cols))
    log.info("      working %d-%d from %s", wmin, wmax, sorted(work_cols))

    # Independent cross-check: ITER publishes its own 15-64 aggregate. If our
    # band sum disagrees with it, one of the two is not what we think it is.
    if (wmin, wmax) == (15, 64) and ITER_WORKING_AGE_TOTAL in df.columns:
        published = pd.to_numeric(df[ITER_WORKING_AGE_TOTAL],
                                  errors="coerce").fillna(0.0).to_numpy()
        ours, theirs = float(den.sum()), float(published.sum())
        rel_diff = abs(ours - theirs) / theirs if theirs else float("inf")
        log.info("      CROSS-CHECK working-age total: bands=%s  %s=%s  diff=%.4f%%",
                 f"{ours:,.0f}", ITER_WORKING_AGE_TOTAL, f"{theirs:,.0f}",
                 100 * rel_diff)
        if rel_diff > 0.01:
            raise PipelineError(
                f"summed 5-year bands give a 15-64 population of {ours:,.0f}, but "
                f"ITER's own {ITER_WORKING_AGE_TOTAL} column says {theirs:,.0f} "
                f"({100 * rel_diff:.2f}% apart).\n"
                "One of these is not measuring what it appears to. Most likely an "
                "overlapping band crept into ITER_AGE_BANDS and is double-counting."
            )

    log.info("      source = ITER banded age counts (FULL COUNT, no sampling error)")
    return out


def load_from_tabulado(cfg, log) -> pd.DataFrame | None:
    """
    Preferred path: single-year-of-age counts by municipality, full count.

    Returns None (rather than raising) if the file is absent or is banded rather
    than single-year, so main() can fall back to the microdata.
    """
    path = resolve(cfg, cfg["demography"]["age_tabulado"])
    if not path.exists():
        log.warning("age tabulado not found at %s", path)
        return None

    df = read_csv_smart(path, log, dtype=str, low_memory=False)
    log.info("READ  %-38s rows=%s cols=%d", path.name, f"{len(df):,}", df.shape[1])

    age_col = next((c for c in AGE_CANDIDATES if c in df.columns), None)
    pop_col = next((c for c in POP_CANDIDATES if c in df.columns), None)
    if not age_col or not pop_col:
        log.warning("tabulado lacks an age column (%s) or population column (%s); "
                    "header: %s", age_col, pop_col, sorted(map(str, df.columns))[:25])
        return None

    if "CVEGEO" in df.columns:
        df["cvegeo"] = df["CVEGEO"].astype(CVEGEO_DTYPE).str.strip().str.zfill(5)
    else:
        ent = next((c for c in ("CVE_ENT", "ENTIDAD", "cve_ent") if c in df.columns), None)
        mun = next((c for c in ("CVE_MUN", "MUNICIPIO", "cve_mun") if c in df.columns), None)
        if not (ent and mun):
            log.warning("tabulado has no usable geographic key")
            return None
        df["cvegeo"] = cvegeo_series(df[ent], df[mun])

    age = pd.to_numeric(df[age_col], errors="coerce")
    if age.notna().mean() < 0.5:
        log.warning("age column %r is not single-year-of-age (only %.0f%% numeric) "
                    "-- it is probably banded. Falling back to microdata, where "
                    "arbitrary config bands can actually be honoured.",
                    age_col, 100 * age.notna().mean())
        return None

    df["_age"] = age
    df["_pop"] = pd.to_numeric(df[pop_col], errors="coerce")
    out = df.loc[~df["cvegeo"].isna() & ~df["cvegeo"].str.endswith("000"),
                 ["cvegeo", "_age", "_pop"]]
    log_rows(log, "tabulado: drop aggregate rows", len(df), len(out))
    log.info("      source = full-count tabulado (preferred: no sampling error "
             "in a denominator)")
    return out.rename(columns={"_age": "age", "_pop": "n"})


def load_from_microdata(cfg, log) -> pd.DataFrame:
    """
    Fallback: weighted person records from the 10% sample.

    Acceptable but second-best. These are denominators, and sampling error in a
    denominator propagates into the ratio -- worse in small municipalities,
    which is exactly where the ratio is noisiest already.
    """
    if not cfg["demography"].get("use_microdata_fallback", True):
        raise PipelineError(
            "age tabulado unavailable and demography.use_microdata_fallback is "
            "false. Fetch the tabulado (config downloads.tabulado_poblacion) or "
            "enable the fallback."
        )

    fcfg = cfg["flows"]
    files = (sorted(ROOT.glob(fcfg["microdata_glob"]))
             if fcfg.get("microdata_is_sharded")
             else [resolve(cfg, fcfg["microdata_file"])])
    if not files or not files[0].exists():
        raise PipelineError(
            "neither the age tabulado nor the census microdata could be found; "
            "cannot build the age distribution."
        )

    log.warning("using MICRODATA fallback for the age distribution (%d shard(s)). "
                "These counts carry sampling error; the full-count tabulado is "
                "preferred for denominators.", len(files))

    v = fcfg["vars"]
    want = [v["ent_current"], v["mun_current"], v["age"], v["factor"]]
    frames = []
    for path in files:
        header = read_csv_smart(path, nrows=0)
        missing = [c for c in want if c not in header.columns]
        if missing:
            raise PipelineError(
                f"{path.name} missing {missing}; header: {sorted(header.columns)}"
            )
        frames.append(read_csv_smart(path, usecols=want,
                                  dtype={c: CVEGEO_DTYPE for c in want},
                                  low_memory=False))
    df = pd.concat(frames, ignore_index=True)

    df["cvegeo"] = cvegeo_series(df[v["ent_current"]], df[v["mun_current"]])
    df["age"] = pd.to_numeric(df[v["age"]], errors="coerce")
    df["n"] = pd.to_numeric(df[v["factor"]], errors="coerce")

    n_in = len(df)
    df = df.dropna(subset=["cvegeo", "age", "n"])
    log_rows(log, "microdata: drop records with missing age/factor/key", n_in, len(df))

    out = (df.groupby(["cvegeo", "age"], as_index=False, observed=True)["n"]
             .sum())
    log_rows(log, "microdata: collapse to cvegeo x age", len(df), len(out))
    return out


def build_shares(ages: pd.DataFrame, cfg, log) -> pd.DataFrame:
    ymin, ymax, wmin, wmax = _bands(cfg)
    log.info("CHOICE age bands: youth = [%d, %d], working-age = [%d, %d]",
             ymin, ymax, wmin, wmax)

    # Age sentinels for "no especificado" (often 999 or blank) must not be
    # counted into the working-age denominator.
    sentinel = ages["age"] > 130
    if bool(sentinel.any()):
        log.warning("      dropping %s record-group(s) with age > 130 "
                    "(non-specified sentinel)", f"{int(sentinel.sum()):,}")
        ages = ages[~sentinel]

    youth = ages["age"].between(ymin, ymax)
    working = ages["age"].between(wmin, wmax)

    out = (
        ages.assign(_y=np.where(youth, ages["n"], 0.0),
                    _w=np.where(working, ages["n"], 0.0))
        .groupby("cvegeo", as_index=False, observed=True)
        .agg(pop_youth=("_y", "sum"), pop_working_age=("_w", "sum"))
    )
    log_rows(log, "aggregate age cells to municipal numerator/denominator",
             len(ages), len(out))
    return _finalise_shares(out, cfg, log)


def _finalise_shares(out: pd.DataFrame, cfg, log) -> pd.DataFrame:
    """
    Turn a numerator/denominator frame into the ratio, with guards.

    Shared by both construction paths -- ITER banded counts and single-year age
    cells -- so the division, the zero-denominator handling and the [0, 1] range
    assertion behave identically whichever source produced the counts.
    """
    ymin, ymax, wmin, wmax = _bands(cfg)

    zero_denom = out["pop_working_age"] <= 0
    if bool(zero_denom.any()):
        log.warning("      %d municipality/ies have zero working-age population; "
                    "dempres_orig set NA rather than dividing by zero: %s",
                    int(zero_denom.sum()), out.loc[zero_denom, "cvegeo"].head(10).tolist())

    out["dempres_orig"] = np.where(zero_denom, np.nan,
                                   out["pop_youth"] / out["pop_working_age"])

    # Stamp the bands into the data so a stray parquet is self-describing.
    out["youth_band"] = f"{ymin}-{ymax}"
    out["working_age_band"] = f"{wmin}-{wmax}"

    s = out["dempres_orig"]
    impossible = s.notna() & ((s < 0) | (s > 1))
    if bool(impossible.any()):
        raise PipelineError(
            f"{int(impossible.sum())} municipality/ies have dempres_orig outside "
            "[0, 1]. The youth band is supposed to be a subset of the working-age "
            "band, so this indicates a band or aggregation error."
        )

    log.info("      dempres_orig: n=%s  min=%.4f  median=%.4f  mean=%.4f  max=%.4f",
             f"{int(s.notna().sum()):,}", float(s.min()), float(s.median()),
             float(s.mean()), float(s.max()))
    return out


def main() -> int:
    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(STEP, cfg)
    log.info("=" * 78)
    log.info("STEP %s", STEP)
    log.info("=" * 78)
    log.info("NOTE  built at ORIGIN MUNICIPALITY level, not origin country -- a "
             "country-level measure is constant across an internal-migration "
             "panel and identifies nothing. Deviation recorded in the codebook.")

    # Preference order, best first:
    #   1. ITER banded counts   -- FULL COUNT, no sampling error, no manual
    #                              download needed. Works whenever the configured
    #                              bands align with ITER's 5-year boundaries.
    #   2. single-year tabulado -- full count, any band
    #   3. microdata            -- honours any band, but carries sampling error
    #                              in a DENOMINATOR when the sample is in use
    banded = load_from_iter_bands(cfg, log)
    ages = None
    if banded is not None:
        source = "census_2020_iter_bands_full_count"
    else:
        ages = load_from_tabulado(cfg, log)
        source = "census_2020_tabulado_full_count"
        if ages is None:
            ages = load_from_microdata(cfg, log)
            source = "census_2020_microdata_weighted"

    n_in = len(banded) if banded is not None else len(ages)

    # Harmonize before aggregating: a split child's counts must land in the
    # parent before the numerator and denominator are formed, not after.
    xwalk = load_crosswalk(cfg, log)
    strategy = cfg["geometry"]["crosswalk_strategy"]

    if banded is not None:
        # Already numerator/denominator; harmonize then sum into the parent.
        banded = banded.copy()
        banded["cvegeo"] = apply_crosswalk(banded["cvegeo"], xwalk, strategy,
                                           log, "demography")
        dem = (banded.groupby("cvegeo", as_index=False, observed=True)
               [["pop_youth", "pop_working_age"]].sum())
        log_rows(log, "aggregate banded counts to parent", len(banded), len(dem))
        dem = _finalise_shares(dem, cfg, log)
    else:
        ages = ages.copy()
        ages["cvegeo"] = apply_crosswalk(ages["cvegeo"], xwalk, strategy,
                                         log, "demography")
        dem = build_shares(ages, cfg, log)

    dem["demography_source"] = source

    assert_cvegeo_valid(dem["cvegeo"], "demography.cvegeo")
    if bool(dem["cvegeo"].duplicated().any()):
        raise PipelineError("duplicate cvegeo in demography table")

    missingness_table(dem, log)

    ymin, ymax, wmin, wmax = _bands(cfg)
    s = dem["dempres_orig"]
    lines = [
        "# Youth share of working-age population (`dempres_orig`)",
        "",
        f"Source: **{source}**",
        f"Definition: population aged **{ymin}-{ymax}** divided by population",
        f"aged **{wmin}-{wmax}**, 2020 census.",
        "",
        "## Deviation from the original specification",
        "",
        'The variable was specified as youth share of the origin **country**.',
        "For an internal-migration panel every origin is Mexico, so a country-level",
        "measure is a constant: it has no variance and cannot identify anything,",
        "and it is absorbed by any origin fixed effect. It is built at **origin",
        "municipality** level instead -- the level at which it varies and the level",
        "the rest of the panel is keyed on.",
        "",
        "## Age bands are a configuration choice",
        "",
        "The literature is not settled, and the choice moves the variable:",
        "",
        "| convention | youth band | source |",
        "|---|---|---|",
        "| UN youth (default here) | 15-29 | UN DESA |",
        "| ILO youth | 15-24 | ILO |",
        "| peak migration propensity | 20-34 | migration literature |",
        "",
        f"Currently: `demography.youth_min_age`={ymin}, `youth_max_age`={ymax}, ",
        f"`working_min_age`={wmin}, `working_max_age`={wmax}.",
        "",
        "`pop_youth` and `pop_working_age` are emitted alongside the ratio so it",
        "can be reconstructed or redefined without re-running the extraction.",
        "",
        "## Distribution",
        "",
        "| statistic | value |",
        "|---|---:|",
        f"| municipalities | {len(dem):,} |",
        f"| non-missing | {int(s.notna().sum()):,} |",
        f"| min | {float(s.min()):.4f} |",
        f"| p25 | {float(s.quantile(.25)):.4f} |",
        f"| median | {float(s.median()):.4f} |",
        f"| p75 | {float(s.quantile(.75)):.4f} |",
        f"| max | {float(s.max()):.4f} |",
        "",
    ]
    if source.endswith("microdata_weighted"):
        lines += [
            "## Caveat: microdata fallback in use",
            "",
            "The full-count tabulado was unavailable, so these counts come from the",
            "10% sample scaled by the expansion factor. They carry sampling error,",
            "and this quantity is a **denominator** -- the error propagates into",
            "the ratio and is worst in small municipalities, which are already the",
            "noisiest. Prefer the tabulado when it is available.",
            "",
        ]
    p = report_path(cfg, "demography_bands.md")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("WROTE %s", rel(p))

    write_interim(cfg, dem, "demography", log)
    log_rows(log, "STEP TOTAL 06_demography", n_in, len(dem))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_step(main, STEP))



