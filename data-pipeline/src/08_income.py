"""
08_income.py -- mean household income per municipality, from INEGI ICMM 2020.

Input  : INEGI ICMM open-data CSV (income.file)
         INEGI ITER 2020 tabulado (population weights for the split-child fold)
Output : interim/income.parquet
         reports/income_summary.md

*** THIS VARIABLE IS A MODELLED ESTIMATE, NOT A DIRECT MEASUREMENT, AND IT IS A
    MEAN, NOT A MEDIAN. ***

The Mexican census does not ask household income, so there is no observed
municipal income anywhere. ICMM (Ingreso Corriente para los Municipios de
Mexico, INEGI, released October 2023) fills the gap by SMALL-AREA ESTIMATION:
a statistical model fit on the ENIGH 2020 (nueva serie) household survey, with
auxiliary variables from the 2020 Census and administrative records, is applied
to every municipality. The published quantity is `icpth` -- "Ingreso Corriente
Promedio Trimestral por Hogar 2020", the MEAN current quarterly income per
household, in nominal 2020 pesos. INEGI publishes no municipal MEDIAN; small-area
models target the mean.

WHAT THE FILE LOOKS LIKE
------------------------
Long format, one row per (ent, mun, est). `est` selects which estimate:

    1  Valor                              -> hh_income      (the point estimate)
    2  Error estandar                     -> hh_income_se
    3  Limite inferior de confianza       -> hh_income_ci_low
    4  Limite superior de confianza       -> hh_income_ci_high
    5  Coeficiente de variacion (%)       -> hh_income_cv

Rows with mun == 000 are state (and, at ent==00, national) aggregates and are
dropped. This step pivots `est` into columns, one row per municipality.

UNITS
-----
Left exactly as published: QUARTERLY, nominal 2020 pesos, per household. No PPP
conversion. Unlike gdp (which the gravity literature wants in international
dollars), income here is a descriptive covariate and is clearest in the units
INEGI reports; forcing it through the 2017-int$ PPP factor used for GDP would
mix a 2020-nominal series with a 2017-real one under one deflator. A user who
wants annual pesos multiplies by 4. `log_hh_income` is ln of the quarterly peso
value.

THE SPLIT-CHILD FOLD (why income cannot be summed like population)
------------------------------------------------------------------
Household income is an INTENSIVE per-household quantity. When a municipality was
split after 2015, the panel folds the child back into its pre-split parent (see
geo.apply_crosswalk / README.md#boundary-harmonization). For an extensive count
-- population, value added -- folding means SUM. For a mean it does not: the
pre-split parent's mean is the household-weighted AVERAGE of the two 2020
municipalities' means,

    M_parent = (H_p * M_p + H_c * M_c) / (H_p + H_c)

Household counts H are not in the ICMM file, so 2020 POPULATION (from ITER, at
the unharmonized municipal level) is used as the weight -- a good proxy whose
only error is a difference in mean household size between parent and child. In
the 2020 vintage this touches exactly ONE municipality: 23005 Benito Juarez
absorbs 23011 Puerto Morelos. Folded municipalities are flagged
`icmm_2020_harmonized`, and their se/ci/cv are set NA, because a weighted mean of
two small-area point estimates carries no clean combined standard error.

Idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CVEGEO_DTYPE, PipelineError, assert_cvegeo_valid, cvegeo_series, ensure_dirs,
    get_logger, load_config, log_rows, missingness_table, read_csv_smart,
    report_path, rel, resolve, run_step, write_interim,
)
from geo import apply_crosswalk, load_crosswalk  # noqa: E402

STEP = "08_income"

# est code -> output column. From diccionario_datos_icmm_2020.csv / catalogos/est.csv.
EST_COLUMN = {
    "1": "hh_income",         # Valor -- the point estimate
    "2": "hh_income_se",      # Error estandar
    "3": "hh_income_ci_low",  # Limite inferior de confianza
    "4": "hh_income_ci_high", # Limite superior de confianza
    "5": "hh_income_cv",      # Coeficiente de variacion (%)
}
UNCERTAINTY_COLS = ["hh_income_se", "hh_income_ci_low", "hh_income_ci_high", "hh_income_cv"]


# ---------------------------------------------------------------------------
# load + pivot
# ---------------------------------------------------------------------------

def load_icmm(cfg, log) -> pd.DataFrame:
    """
    Read the ICMM long file and pivot `est` into columns, one row per muni.

    Returns unharmonized municipal rows (2020 vintage) keyed on a 5-char cvegeo.
    """
    path = resolve(cfg, cfg["income"]["file"])
    if not path.exists():
        raise PipelineError(
            f"ICMM file not found: {rel(path)}\n\n"
            "Run `python src/00_download.py --only icmm_2020` to fetch it "
            "(URL is VERIFIED in config), or place the open-data CSV at the "
            "path in income.file."
        )

    df = read_csv_smart(path, log, dtype=str, low_memory=False)
    df = df.rename(columns={c: str(c).lstrip("﻿").strip() for c in df.columns})
    log.info("READ  %-38s rows=%s cols=%d", path.name, f"{len(df):,}", df.shape[1])

    for col in ("ent", "mun", "est", "icpth"):
        if col not in df.columns:
            raise PipelineError(
                f"ICMM file missing expected column {col!r}. Header: "
                f"{sorted(map(str, df.columns))}. The open-data layout is "
                "(ent, mun, est, icpth); a different shape means a different "
                "product or vintage."
            )

    # Drop aggregates: mun == 000 is a state total (and ent==00 & mun==000 the
    # national total). Municipal rows have a non-zero municipality code.
    mun3 = df["mun"].astype("string").str.strip().str.zfill(3)
    n_in = len(df)
    df = df[mun3 != "000"].copy()
    log_rows(log, "drop state/national aggregate rows (mun==000)", n_in, len(df))

    df["cvegeo"] = cvegeo_series(df["ent"], df["mun"])
    if bool(df["cvegeo"].isna().any()):
        raise PipelineError(f"{int(df['cvegeo'].isna().sum())} ICMM row(s) have an "
                            "unparseable (ent, mun) key")

    unknown_est = sorted(set(df["est"].astype(str)) - set(EST_COLUMN))
    if unknown_est:
        raise PipelineError(
            f"ICMM has unexpected est code(s) {unknown_est}; expected {sorted(EST_COLUMN)} "
            "(1=Valor 2=SE 3=CI low 4=CI high 5=CV). Check catalogos/est.csv for "
            "this vintage."
        )

    df["icpth"] = pd.to_numeric(df["icpth"], errors="coerce")
    wide = (df.pivot_table(index="cvegeo", columns="est", values="icpth", aggfunc="first")
              .rename(columns=EST_COLUMN)
              .reset_index())
    wide.columns.name = None

    # Every municipality must carry a point estimate; the rest are diagnostics.
    if "hh_income" not in wide.columns:
        raise PipelineError("no est==1 (Valor) rows found; nothing to pivot into "
                            "hh_income.")
    n_missing_val = int(wide["hh_income"].isna().sum())
    if n_missing_val:
        raise PipelineError(f"{n_missing_val} municipality/ies have no point "
                            "estimate (est==1). The pivot is incomplete.")
    for c in UNCERTAINTY_COLS:
        if c not in wide.columns:
            wide[c] = np.nan

    nonpos = wide["hh_income"] <= 0
    if bool(nonpos.any()):
        raise PipelineError(f"{int(nonpos.sum())} municipality/ies have hh_income "
                            "<= 0; a mean income should be strictly positive.")

    wide["cvegeo"] = wide["cvegeo"].astype(CVEGEO_DTYPE)
    log.info("      pivoted to %s municipalities x {value, se, ci_low, ci_high, cv}",
             f"{len(wide):,}")
    return wide[["cvegeo", "hh_income", *UNCERTAINTY_COLS]]


# ---------------------------------------------------------------------------
# split-child fold (population-weighted, because income is intensive)
# ---------------------------------------------------------------------------

def _unharmonized_pop_2020(cfg, log) -> pd.Series:
    """
    Municipal 2020 population at the UNHARMONIZED (2020-vintage) level, from ITER.

    Needed as fold weights: the panel's population.parquet has already folded
    children into parents, so it cannot tell 23005 from 23011. ITER still lists
    them separately (LOC == 0000 rows). Returns a cvegeo -> POBTOT Series.
    """
    path = resolve(cfg, cfg["population"]["census_2020_tabulado"])
    if not path.exists():
        raise PipelineError(
            f"ITER tabulado not found at {rel(path)}; needed to weight the "
            "split-child income fold. It is the same VERIFIED download as the "
            "population step (downloads.tabulado_poblacion)."
        )
    df = read_csv_smart(path, log, dtype=str, low_memory=False)
    df = df.rename(columns={c: str(c).lstrip("﻿").strip() for c in df.columns})

    if "LOC" in df.columns:
        df = df[df["LOC"].astype("string").str.strip().str.zfill(4).eq("0000")]
    ent = next((c for c in ("ENTIDAD", "CVE_ENT", "cve_ent") if c in df.columns), None)
    mun = next((c for c in ("MUN", "CVE_MUN", "cve_mun") if c in df.columns), None)
    pop = next((c for c in ("POBTOT", "POB", "poblacion") if c in df.columns), None)
    if not (ent and mun and pop):
        raise PipelineError(
            f"ITER file lacks a usable (entidad, mun, POBTOT) triple; header: "
            f"{sorted(map(str, df.columns))}"
        )
    df = df.copy()
    df["cvegeo"] = cvegeo_series(df[ent], df[mun])
    df = df[~df["cvegeo"].isna() & ~df["cvegeo"].str.endswith("000")]
    s = pd.to_numeric(df[pop], errors="coerce")
    return pd.Series(s.to_numpy(), index=df["cvegeo"].astype(str).to_numpy())


def harmonize_income(wide: pd.DataFrame, cfg, log) -> pd.DataFrame:
    """
    Fold post-2015 split children into their parents, population-weighted.

    Non-colliding municipalities (the overwhelming majority) pass through with
    their published se/ci/cv intact and source `icmm_2020`. Where two or more
    2020 municipalities map to one pre-split parent, the point estimate becomes a
    household-weighted (population-weighted) mean, the uncertainty columns are
    set NA, and the source is `icmm_2020_harmonized`.
    """
    xwalk = load_crosswalk(cfg, log)
    strategy = cfg["geometry"]["crosswalk_strategy"]

    df = wide.copy()
    df["cvegeo_h"] = apply_crosswalk(df["cvegeo"], xwalk, strategy, log, "income")

    sizes = df.groupby("cvegeo_h").size()
    collide = set(sizes[sizes > 1].index)

    df["hh_income_source"] = "icmm_2020"

    if collide:
        weight_mode = cfg["income"].get("harmonize_weight", "population")
        involved = sorted(df.loc[df["cvegeo_h"].isin(collide), "cvegeo"].astype(str))
        log.warning("      %d parent(s) absorb a post-2015 split child; income is "
                    "INTENSIVE, so folding uses a %s-weighted mean, not a sum. "
                    "Affected 2020 municipalities: %s",
                    len(collide), weight_mode, involved)

        weights = None
        if weight_mode == "population":
            weights = _unharmonized_pop_2020(cfg, log)

        rows = []
        for parent, grp in df[df["cvegeo_h"].isin(collide)].groupby("cvegeo_h"):
            codes = grp["cvegeo"].astype(str).tolist()
            if weights is not None and all(c in weights.index for c in codes):
                w = np.array([float(weights[c]) for c in codes], dtype="float64")
            else:
                missing = [c for c in codes if weights is None or c not in getattr(weights, "index", [])]
                log.warning("        %s: no population weight for %s; falling back "
                            "to an UNWEIGHTED mean for this fold (documented, but "
                            "verify -- a small child then counts equally with a "
                            "large parent)", parent, missing)
                w = np.ones(len(codes), dtype="float64")
            v = grp["hh_income"].to_numpy(dtype="float64")
            folded_value = float(np.average(v, weights=w))
            log.info("        fold %s <- %s  value %s (weights %s)",
                     parent, codes, f"{folded_value:,.1f}",
                     ", ".join(f"{x:,.0f}" for x in w))
            rows.append({"cvegeo_h": parent, "hh_income": folded_value,
                         **{c: np.nan for c in UNCERTAINTY_COLS},
                         "hh_income_source": "icmm_2020_harmonized"})
        folded = pd.DataFrame(rows)
        kept = df[~df["cvegeo_h"].isin(collide)].copy()
        out = pd.concat(
            [kept[["cvegeo_h", "hh_income", *UNCERTAINTY_COLS, "hh_income_source"]],
             folded],
            ignore_index=True)
    else:
        out = df[["cvegeo_h", "hh_income", *UNCERTAINTY_COLS, "hh_income_source"]].copy()

    out = out.rename(columns={"cvegeo_h": "cvegeo"})
    out["cvegeo"] = out["cvegeo"].astype(CVEGEO_DTYPE)
    log_rows(log, "harmonize income to the parent universe", len(df), len(out))

    if bool(out["cvegeo"].duplicated().any()):
        dups = out.loc[out["cvegeo"].duplicated(), "cvegeo"].tolist()
        raise PipelineError(f"duplicate cvegeo after the income fold: {dups[:10]}. "
                            "A collision was not collapsed.")

    # Squares and logs, named unambiguously -- same convention as 04_gdp.py:
    #   hh_income_sq     = hh_income ** 2        a square of a LEVEL, in pesos^2
    #   log_hh_income    = ln(hh_income)         a log, dimensionless
    #   log_hh_income_sq = (ln hh_income) ** 2   a SQUARED LOG -- the quadratic
    #                                            term for a log-linear gravity spec
    # NOT emitted: ln(hh_income^2) == 2*ln(hh_income), exactly collinear with
    # log_hh_income. Income is guaranteed positive above, but guard the log anyway.
    v = out["hh_income"]
    out["hh_income_sq"] = v ** 2
    out["log_hh_income"] = np.where(v > 0, np.log(v.where(v > 0)), np.nan)
    out["log_hh_income_sq"] = out["log_hh_income"] ** 2
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def write_report(cfg, df: pd.DataFrame, log) -> Path:
    icfg = cfg["income"]
    cv = df["hh_income_cv"].dropna()
    caution = int((cv >= icfg["cv_caution_threshold"]).sum())
    unreliable = int((cv >= icfg["cv_unreliable_threshold"]).sum())
    n_folded = int((df["hh_income_source"] == "icmm_2020_harmonized").sum())
    v = df["hh_income"]

    lines = [
        "# Mean household income (`hh_income`) -- INEGI ICMM 2020",
        "",
        "**Modelled small-area estimate, not a direct measurement. A MEAN, not a "
        "median.**",
        "",
        "`hh_income` is the *Ingreso Corriente Promedio Trimestral por Hogar 2020*:",
        "the mean current **quarterly** income per household, in **nominal 2020",
        "pesos**, estimated for every municipality by small-area estimation from",
        "the ENIGH 2020 (nueva serie) survey plus 2020 Census and administrative",
        "auxiliary variables. INEGI publishes no municipal median. Released October",
        "2023.",
        "",
        "## Coverage",
        "",
        "| | |",
        "|---|---:|",
        f"| municipalities with income | {len(df):,} |",
        f"| missing point estimate | {int(v.isna().sum()):,} |",
        f"| folded (population-weighted, `icmm_2020_harmonized`) | {n_folded:,} |",
        "",
        "## Distribution (quarterly pesos per household)",
        "",
        "| statistic | value |",
        "|---|---:|",
        f"| min | {v.min():,.0f} |",
        f"| p25 | {v.quantile(.25):,.0f} |",
        f"| median | {v.median():,.0f} |",
        f"| mean | {v.mean():,.0f} |",
        f"| p75 | {v.quantile(.75):,.0f} |",
        f"| max | {v.max():,.0f} |",
        "",
        "## Reliability (coefficient of variation)",
        "",
        "INEGI's convention for the CV: `<15%` reliable, `15-30%` use with caution,",
        "`>30%` not recommended.",
        "",
        "| band | municipalities |",
        "|---|---:|",
        f"| CV < {icfg['cv_caution_threshold']}% (reliable) | {int((cv < icfg['cv_caution_threshold']).sum()):,} |",
        f"| {icfg['cv_caution_threshold']}-{icfg['cv_unreliable_threshold']}% (caution) | {caution - unreliable:,} |",
        f"| >= {icfg['cv_unreliable_threshold']}% (not recommended) | {unreliable:,} |",
        f"| CV reported | {len(cv):,} |",
        "",
        f"CV range: {cv.min():.2f}% - {cv.max():.2f}% (median {cv.median():.2f}%).",
        "",
        "## Boundary harmonization",
        "",
        "Household income is intensive, so a post-2015 split child is folded into",
        "its parent by a **population-weighted mean**, not a sum. The folded",
        "municipalities carry `hh_income_source = icmm_2020_harmonized` and have",
        "their se/ci/cv set NA (a weighted mean of two SAE point estimates has no",
        "clean combined standard error). See `08_income.py` and",
        "README.md#boundary-harmonization.",
        "",
        "## Caveats for use",
        "",
        "1. **Modelled, not measured.** Every value is a small-area model output;",
        "   the census does not ask income. Lean on the CV, and treat within-region",
        "   rankings as more trustworthy than precise levels.",
        "2. **Mean, not median.** No municipal median exists; a mean is more",
        "   sensitive to the top of the distribution.",
        "3. **Quarterly nominal 2020 pesos.** Not PPP-adjusted and not comparable in",
        "   level to `src_gdppc` / `dst_gdppc` (2017 international dollars).",
        "",
    ]
    p = report_path(cfg, "income_summary.md")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("WROTE %s", rel(p))
    return p


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
    log.warning("hh_income is a MODELLED small-area estimate (INEGI ICMM), a MEAN "
                "not a median, in quarterly nominal 2020 pesos. See "
                "reports/income_summary.md")

    wide = load_icmm(cfg, log)
    inc = harmonize_income(wide, cfg, log)

    assert_cvegeo_valid(inc["cvegeo"], "income.cvegeo")

    # Reliability accounting, using INEGI's CV convention.
    cv = inc["hh_income_cv"].dropna()
    unreliable = int((cv >= cfg["income"]["cv_unreliable_threshold"]).sum())
    if unreliable:
        log.warning("      %d municipality/ies have CV >= %s%% (INEGI: 'not "
                    "recommended'). They are kept but flagged via hh_income_cv; "
                    "threshold them downstream if needed.",
                    unreliable, cfg["income"]["cv_unreliable_threshold"])
    else:
        log.info("      reliability: every municipal CV is below the "
                 "'not recommended' threshold (%s%%); max CV = %.2f%%",
                 cfg["income"]["cv_unreliable_threshold"],
                 float(cv.max()) if len(cv) else float("nan"))

    missingness_table(inc, log)
    write_report(cfg, inc, log)
    write_interim(cfg, inc, "income", log)
    log_rows(log, "STEP TOTAL 08_income", len(wide), len(inc))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_step(main, STEP))
