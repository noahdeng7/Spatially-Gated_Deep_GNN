"""
04_gdp.py -- municipal GDP per capita, PPP-adjusted, from two independent sources.

Input  : INEGI Censos Economicos 2019 municipal gross value added (VACB)
         Kummu et al. (or Wang & Sun) gridded GDP per capita
         WorldPop 2020 grid (for population-weighted zonal aggregation)
         interim/population.parquet
Output : interim/gdp.parquet
         reports/gdp_source_comparison.md

*** THIS VARIABLE IS AN ESTIMATE, NOT AN OFFICIAL STATISTIC. ***

Mexico publishes no official municipal GDP series. INEGI's ITAEE stops at the
state level. Everything below is a construction, and the construction has known
biases that are documented in README.md#gdp-is-constructed and repeated in the
codebook. The most important one:

    Censal gross value added covers economic-census establishments. It
    systematically UNDERSTATES subsistence agriculture and informal activity.
    Those are concentrated in exactly the poor rural municipalities that send
    the most migrants, so the measure is biased downward precisely where the
    gravity model is most sensitive to it. Treat the level as indicative and
    lean on the cross-source comparison below.

TWO SOURCES, ALWAYS BOTH
------------------------
Both are computed on every run and correlated. `gdp.source` chooses which one
feeds the canonical columns; it does not switch the other one off. If the two
disagree beyond `gdp.min_cross_source_correlation` (default 0.80) the build
STOPS with DecisionRequired rather than silently picking one -- a disagreement
that large means at least one of them is measuring something other than what we
think it is, and that is a question for a human.

SQUARES AND LOGS
----------------
Emitted explicitly, because these are different objects and the naming in the
literature is sloppy:

    gdppc_sq      = gdppc ** 2               a square, in (int$)^2
    log_gdppc     = ln(gdppc)                a log, dimensionless
    log_gdppc_sq  = (ln gdppc) ** 2          a SQUARED LOG -- the quadratic term
                                             you want in a log-linear gravity
                                             specification

Note what is NOT emitted: ln(gdppc**2). It equals 2*ln(gdppc), so it is exactly
collinear with log_gdppc and contributes nothing. If you find yourself wanting
it, you want log_gdppc_sq instead.

Idempotent.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DecisionRequired, PipelineError, ROOT, assert_cvegeo_valid, ensure_dirs,
    get_logger, load_config, log_rows, missingness_table, read_interim,
    read_csv_smart, report_path, rel, resolve, run_step, write_interim,
)
from geo import apply_crosswalk, assert_geometry_vintage, load_crosswalk, load_municipios  # noqa: E402

STEP = "04_gdp"

UNIT_SCALE = {
    "pesos": 1.0,
    "thousands_of_pesos": 1_000.0,
    "millions_of_pesos": 1_000_000.0,
}


# ---------------------------------------------------------------------------
# source A: Censos Economicos
# ---------------------------------------------------------------------------

def load_censos_economicos(cfg, pop: pd.DataFrame, log) -> pd.DataFrame:
    """
    Municipal gross value added / population -> GDP per capita in PPP int$.

    Chain: nominal pesos (CE vintage) -> deflate to PPP base year -> divide by
    the PPP conversion factor -> international dollars of the base year.
    """
    ccfg = cfg["gdp"]["censos_economicos"]
    path = resolve(cfg, ccfg["file"])
    if not path.exists():
        # Fatal only if this is the source the panel actually uses. Censos
        # Economicos is a MANUAL fetch (SAIC is an interactive tool with no
        # public API), so when the gridded product is selected there is no
        # reason to block on it.
        if cfg["gdp"]["source"] != "censos_economicos":
            raise FileNotFoundError(str(path))
        raise PipelineError(
            f"Censos Economicos file not found: {rel(path)}\n\n"
            "gdp.source is 'censos_economicos', so this file is REQUIRED.\n\n"
            "It is a manual export: SAIC (inegi.org.mx/app/saic) is an "
            "interactive\nquery tool with no public API. See "
            "docs/MANUAL_DOWNLOADS.md section 4.\n\n"
            "OR set gdp.source: gridded to build the panel from the Kummu "
            "gridded\nproduct alone, which is already downloaded. You lose the "
            "cross-source\ncorrelation check -- see the note that 04_gdp writes "
            "into the report."
        )

    df = read_csv_smart(path, log, dtype=str, low_memory=False)
    log.info("READ  %-38s rows=%s cols=%d", path.name, f"{len(df):,}", df.shape[1])

    va_field = ccfg["value_added_field"]
    if va_field not in df.columns:
        raise PipelineError(
            f"\nvalue-added field {va_field!r} not in {path.name}\n"
            f"  header: {sorted(map(str, df.columns))}\n\n"
            "Set gdp.censos_economicos.value_added_field to the real column. In "
            "SAIC exports the gross value added column is usually labelled with "
            "some form of 'valor agregado censal bruto' / VACB."
        )

    # key
    if "CVEGEO" in df.columns:
        df["cvegeo"] = df["CVEGEO"].astype("string").str.strip().str.zfill(5)
    else:
        from common import cvegeo_series
        ent = next((c for c in ("CVE_ENT", "ENTIDAD", "cve_ent") if c in df.columns), None)
        mun = next((c for c in ("CVE_MUN", "MUNICIPIO", "cve_mun") if c in df.columns), None)
        if not (ent and mun):
            raise PipelineError(
                f"Censos Economicos file has no usable geographic key. "
                f"Header: {sorted(map(str, df.columns))}"
            )
        df["cvegeo"] = cvegeo_series(df[ent], df[mun])

    df["va_nominal"] = pd.to_numeric(df[va_field], errors="coerce")

    n_in = len(df)
    df = df[~df["cvegeo"].isna() & ~df["cvegeo"].str.endswith("000")]
    log_rows(log, "CE: drop aggregate/unkeyed rows", n_in, len(df))

    # Harmonize CE codes BEFORE aggregating: the SAIC catalog carries
    # window-created municipalities (Seybaplaya, the Chiapas 2017/2011
    # creations, ...) under their own codes, and without this their value
    # added would fail the join to the harmonized universe and silently
    # vanish from the parent's total.
    from geo import apply_crosswalk, load_crosswalk
    xwalk = load_crosswalk(cfg, log)
    df["cvegeo"] = apply_crosswalk(df["cvegeo"], xwalk,
                                   cfg["geometry"]["crosswalk_strategy"],
                                   log, "censos economicos")

    # A SAIC export is usually one row per municipality per sector; sum
    # sectors (and any crosswalk-folded children) to a municipal total.
    va = df.groupby("cvegeo", as_index=False, observed=True)["va_nominal"].sum(min_count=1)
    log_rows(log, "CE: sum sectors to municipal total", len(df), len(va))

    # --- units -> pesos ------------------------------------------------------
    units = ccfg.get("units", "thousands_of_pesos")
    if units not in UNIT_SCALE:
        raise PipelineError(f"unknown gdp.censos_economicos.units: {units!r}. "
                            f"Expected one of {sorted(UNIT_SCALE)}")
    va["va_pesos"] = va["va_nominal"] * UNIT_SCALE[units]
    log.info("      CE units=%s (x%s) -> total VA = %.3e pesos",
             units, f"{UNIT_SCALE[units]:,.0f}", va["va_pesos"].sum())

    # --- deflate + PPP -------------------------------------------------------
    ppp = cfg["gdp"]["ppp"]
    factor = ppp.get("conversion_factor")
    deflator = ppp.get("deflator_from_year_to_base")
    base = ppp.get("base_year", 2017)

    if factor is None:
        raise PipelineError(
            "\ngdp.ppp.conversion_factor is null.\n\n"
            "[TODO] Read the World Bank series PA.NUS.PPP (PPP conversion factor, "
            f"GDP, LCU per international $) for Mexico, base year {base}, and set "
            "it in config.yaml.\n\n"
            "This is deliberately not defaulted: hardcoding a plausible-looking "
            "PPP factor would silently rescale every GDP coefficient in the model "
            "by an unknown constant."
        )

    if deflator is None:
        if int(ccfg["year"]) != int(base):
            raise PipelineError(
                f"\ngdp.ppp.deflator_from_year_to_base is null but the Censos "
                f"Economicos vintage ({ccfg['year']}) differs from the PPP base "
                f"year ({base}).\n\n"
                "[TODO] Set the INEGI INPC ratio that converts "
                f"{ccfg['year']} pesos into {base} pesos. Without it the series "
                "mixes nominal vintages and the PPP conversion is meaningless."
            )
        deflator = 1.0

    va["va_real_base"] = va["va_pesos"] * float(deflator)
    va["va_intl_dollars"] = va["va_real_base"] / float(factor)
    log.info("      deflator(%s->%s)=%s  PPP factor=%s  -> total VA = %.3e int$",
             ccfg["year"], base, deflator, factor, va["va_intl_dollars"].sum())

    # --- per capita ----------------------------------------------------------
    out = va.merge(pop[["cvegeo", "pop_2020"]], on="cvegeo", how="left")
    missing_pop = out["pop_2020"].isna() | (out["pop_2020"] <= 0)
    if bool(missing_pop.any()):
        log.warning("      %d municipality/ies have no usable 2020 population; "
                    "gdppc left NA rather than divided by zero",
                    int(missing_pop.sum()))
    out["gdppc_ce"] = np.where(missing_pop, np.nan,
                               out["va_intl_dollars"] / out["pop_2020"])
    return out[["cvegeo", "va_intl_dollars", "gdppc_ce"]]


# ---------------------------------------------------------------------------
# source B: gridded GDP per capita
# ---------------------------------------------------------------------------

def load_gridded_gdp(cfg, log) -> pd.DataFrame:
    """
    Zonally aggregate a gridded GDP-per-capita product to municipal polygons,
    population-weighted by the WorldPop grid.

    GDP per capita is an INTENSIVE quantity, so the zonal statistic must be a
    population-weighted mean, not a sum and not an unweighted mean. An unweighted
    polygon mean would let empty desert cells vote as loudly as a dense city.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    gcfg = cfg["gdp"]["gridded"]
    path = resolve(cfg, gcfg["raster"])
    if not path.exists():
        raise PipelineError(
            f"gridded GDP product not found: {path}\n"
            "See config.yaml -> downloads.gridded_gdp."
        )

    gdf = load_municipios(cfg, log)
    assert_geometry_vintage(cfg, gdf, log)
    xwalk = load_crosswalk(cfg, log)
    gdf["cvegeo"] = apply_crosswalk(gdf["cvegeo"], xwalk,
                                    cfg["geometry"]["crosswalk_strategy"], log, "geometry")
    if bool(gdf["cvegeo"].duplicated().any()):
        gdf = gdf.dissolve(by="cvegeo").reset_index()
        gdf["cvegeo"] = gdf["cvegeo"].astype("string")

    from zonal import population_weighted_zonal_mean  # local helper

    log.info("      gridded GDP product = %s, band year = %s",
             gcfg.get("product"), gcfg.get("band_year"))
    res = population_weighted_zonal_mean(
        cfg=cfg, gdf=gdf, value_raster=path,
        band=gcfg.get("band_index", 1), log=log, label="gridded GDP pc",
        netcdf_variable=gcfg.get("netcdf_variable"),
        netcdf_year=gcfg.get("band_year"),
    )
    return res.rename(columns={"value": "gdppc_grid"})[["cvegeo", "gdppc_grid"]]


# ---------------------------------------------------------------------------
# comparison gate
# ---------------------------------------------------------------------------

def compare_sources(cfg, df: pd.DataFrame, log) -> dict:
    """
    Correlate the two constructions and STOP if they disagree materially.

    Both Pearson (on logs, since the levels are heavily right-skewed) and
    Spearman are reported. The gate uses the Pearson-on-logs figure, because the
    gravity model consumes logs and that is the correlation that actually
    matters for the estimated coefficient.
    """
    both = df.dropna(subset=["gdppc_ce", "gdppc_grid"])
    both = both[(both["gdppc_ce"] > 0) & (both["gdppc_grid"] > 0)]
    n = len(both)
    if n < 30:
        raise PipelineError(
            f"only {n} municipality/ies have BOTH GDP measures; cannot assess "
            "agreement. Check that both sources keyed correctly onto cvegeo."
        )

    lce = np.log(both["gdppc_ce"])
    lgr = np.log(both["gdppc_grid"])
    pearson_log = float(np.corrcoef(lce, lgr)[0, 1])
    pearson_lvl = float(np.corrcoef(both["gdppc_ce"], both["gdppc_grid"])[0, 1])
    spearman = float(both["gdppc_ce"].corr(both["gdppc_grid"], method="spearman"))
    ratio = float((both["gdppc_ce"] / both["gdppc_grid"]).median())

    log.info("=" * 70)
    log.info("GDP SOURCE COMPARISON  (n = %s municipalities with both)", f"{n:,}")
    log.info("  Pearson  r (log-log) : %.4f   <- the gate", pearson_log)
    log.info("  Pearson  r (levels)  : %.4f", pearson_lvl)
    log.info("  Spearman rho         : %.4f", spearman)
    log.info("  median ratio CE/grid : %.4f", ratio)
    log.info("  CE   : median %s int$  IQR [%s, %s]",
             f"{both['gdppc_ce'].median():,.0f}",
             f"{both['gdppc_ce'].quantile(.25):,.0f}",
             f"{both['gdppc_ce'].quantile(.75):,.0f}")
    log.info("  grid : median %s int$  IQR [%s, %s]",
             f"{both['gdppc_grid'].median():,.0f}",
             f"{both['gdppc_grid'].quantile(.25):,.0f}",
             f"{both['gdppc_grid'].quantile(.75):,.0f}")
    log.info("=" * 70)

    stats = dict(n=n, pearson_log=pearson_log, pearson_level=pearson_lvl,
                 spearman=spearman, median_ratio=ratio)

    threshold = float(cfg["gdp"]["min_cross_source_correlation"])
    if pearson_log < threshold:
        raise DecisionRequired(
            "\n" + "=" * 74 + "\n"
            "GDP SOURCES DISAGREE -- STOPPING FOR A HUMAN DECISION\n"
            + "=" * 74 + "\n\n"
            f"  Pearson r on logs : {pearson_log:.4f}\n"
            f"  Required minimum  : {threshold:.2f}\n"
            f"  Spearman rho      : {spearman:.4f}\n"
            f"  median CE/grid    : {ratio:.4f}\n"
            f"  n municipalities  : {n:,}\n\n"
            "This was configured to halt rather than pick one, because a "
            "disagreement this size means the two constructions are not measuring "
            "the same thing and the choice materially changes your results.\n\n"
            "Before overriding, check in this order:\n"
            "  1. Units. Is gdp.censos_economicos.units right? A 1000x error here\n"
            "     leaves the correlation intact but the ratio far from 1 -- if\n"
            f"     the median ratio ({ratio:.4f}) is near a power of ten, this is it.\n"
            "  2. PPP / deflator. A wrong constant shifts the level, not the\n"
            "     correlation -- so this is NOT the cause of a low r.\n"
            "  3. Coverage. Does the gridded product actually vary within Mexico,\n"
            "     or is it downscaled from state totals? If the latter, it cannot\n"
            "     validate municipal variation and is not a real cross-check.\n"
            "  4. Key alignment. Confirm both sources joined on 5-char cvegeo and\n"
            "     that neither silently lost its leading zeros.\n\n"
            "To proceed anyway, either lower gdp.min_cross_source_correlation or\n"
            "set it to 0 -- but record the decision in the README, because a\n"
            "reader of your results will want to know.\n"
            + "=" * 74
        )

    log.info("GATE PASS  r(log-log)=%.4f >= %.2f", pearson_log, threshold)
    return stats


def write_comparison_report(cfg, df: pd.DataFrame, stats: dict, log) -> Path:
    gcfg = cfg["gdp"]
    lines = [
        "# GDP per capita: source comparison",
        "",
        "**Municipal GDP per capita is an estimate, not an official statistic.**",
        "Mexico publishes no official municipal GDP series.",
        "",
        f"Canonical source in use: **`{gcfg['source']}`** (config `gdp.source`)",
        "",
    ]

    if stats.get("single_source"):
        lines += [
            "## ⚠ SINGLE SOURCE — NO CROSS-CHECK WAS PERFORMED",
            "",
            "Censos Económicos was not available, so this variable was built from",
            "the gridded product **alone** and the correlation gate did not run.",
            "",
            "Why that matters more here than it would elsewhere: Mexico publishes",
            "no official municipal GDP series. This column is a construction, and",
            "the two-source comparison is the only independent evidence that the",
            "construction is measuring what it claims. Without it, the variable",
            "rests on one downscaled product with no corroboration.",
            "",
            "Compounding it: gridded GDP products are typically downscaled from",
            "subnational accounts using night-lights or population as the key. If",
            "that is true for Mexico, within-state municipal variation is partly an",
            "artefact of the downscaling rather than independent information — and",
            "there is nothing here to reveal that.",
            "",
            "**Treat municipal GDP as indicative and say so in your write-up.**",
            "Fetch the SAIC export and re-run to restore the check; see",
            "`docs/MANUAL_DOWNLOADS.md` section 4.",
            "",
        ]
    else:
        lines += [
            "## Agreement between the two constructions",
            "",
            "| statistic | value |",
            "|---|---:|",
            f"| municipalities with both measures | {stats['n']:,} |",
            f"| Pearson r, log-log **(the gate)** | {stats['pearson_log']:.4f} |",
            f"| Pearson r, levels | {stats['pearson_level']:.4f} |",
            f"| Spearman rho | {stats['spearman']:.4f} |",
            f"| median ratio CE / gridded | {stats['median_ratio']:.4f} |",
            "",
            f"Gate threshold: `gdp.min_cross_source_correlation` = "
            f"{gcfg['min_cross_source_correlation']}. "
            f"**{'PASS' if stats['pearson_log'] >= float(gcfg['min_cross_source_correlation']) else 'FAIL'}**",
            "",
        ]

    lines += [
        "## The two constructions",
        "",
        "### A. Censos Economicos (default)",
        "",
        f"INEGI Censos Economicos {gcfg['censos_economicos']['year']}, municipal gross",
        "value added (VACB), divided by 2020 municipal population, deflated to the",
        f"PPP base year and converted at the World Bank PA.NUS.PPP factor to",
        f"{gcfg['ppp']['base_year']} international dollars.",
        "",
        "**Known bias.** Censal value added covers economic-census establishments.",
        "It systematically understates subsistence agriculture and informal",
        "activity, which are concentrated in poor rural municipalities -- the",
        "high-emigration origins this model cares most about. The measure is",
        "therefore biased *downward where it matters most*, and the bias is",
        "correlated with the dependent variable rather than random.",
        "",
        f"### B. Gridded ({gcfg['gridded']['product']})",
        "",
        f"Gridded GDP per capita, band year {gcfg['gridded']['band_year']}, zonally",
        "aggregated to municipal polygons as a **population-weighted mean** using",
        "the WorldPop 2020 grid. GDP per capita is intensive, so an unweighted",
        "polygon mean would let empty cells vote as loudly as dense ones.",
        "",
        "**Known limitation.** Gridded GDP products are typically downscaled from",
        "subnational (often state-level) accounts using night-lights or population",
        "as the disaggregation key. Where that is true, within-state municipal",
        "variation is partly an artefact of the downscaling key rather than",
        "independent information -- which weakens this as a true cross-check.",
        "",
        "## Alternatives not taken",
        "",
        "- **State-level GDP (ITAEE) assigned to all municipalities in a state.**",
        "  Official and clean, but discards all within-state variation, which is",
        "  most of the variation a municipal gravity model is trying to use.",
        "- **Night-lights as a direct income proxy.** Avoids the informality bias",
        "  but measures something else, and saturates in dense urban cores.",
        "- **CONEVAL poverty/marginalisation indices.** Better-measured and",
        "  arguably closer to the migration decision, but not a GDP series, so",
        "  coefficients are not comparable to the gravity literature.",
        "",
    ]
    p = report_path(cfg, "gdp_source_comparison.md")
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
    log.warning("Municipal GDP per capita is CONSTRUCTED, not official. "
                "See reports/gdp_source_comparison.md")

    pop = read_interim(cfg, "population", log)

    try:
        ce = load_censos_economicos(cfg, pop, log)
    except FileNotFoundError as exc:
        # Only reachable when gdp.source is NOT censos_economicos -- the loader
        # raises PipelineError instead when CE is the selected source.
        ce = None
        log.warning("=" * 70)
        log.warning("CENSOS ECONOMICOS UNAVAILABLE -- running on the gridded "
                    "source alone")
        log.warning("=" * 70)
        log.warning("  missing: %s", rel(Path(str(exc))))
        log.warning("  THE CROSS-SOURCE CORRELATION GATE WILL NOT RUN.")
        log.warning("  That gate is the only independent check on a variable "
                    "Mexico does not")
        log.warning("  publish officially. Without it, municipal GDP per capita "
                    "rests on a")
        log.warning("  single downscaled product with no corroboration.")
        log.warning("  Fetch the SAIC export when you can -- see "
                    "docs/MANUAL_DOWNLOADS.md section 4.")

    grid = load_gridded_gdp(cfg, log)

    if ce is not None:
        gdp = ce.merge(grid, on="cvegeo", how="outer")
        log.info("      merged: %s municipality/ies (CE=%s, grid=%s)",
                 f"{len(gdp):,}", f"{ce['gdppc_ce'].notna().sum():,}",
                 f"{grid['gdppc_grid'].notna().sum():,}")
        stats = compare_sources(cfg, gdp, log)
    else:
        gdp = grid.copy()
        gdp["gdppc_ce"] = np.nan
        log.info("      gridded only: %s municipality/ies", f"{len(gdp):,}")
        stats = dict(n=0, pearson_log=float("nan"), pearson_level=float("nan"),
                     spearman=float("nan"), median_ratio=float("nan"),
                     single_source=True)

    write_comparison_report(cfg, gdp, stats, log)

    # --- canonical series ----------------------------------------------------
    source = cfg["gdp"]["source"]
    col = {"censos_economicos": "gdppc_ce", "gridded": "gdppc_grid"}.get(source)
    if col is None:
        raise PipelineError(f"unknown gdp.source: {source!r}")
    gdp["gdppc"] = gdp[col]
    log.info("CHOICE canonical gdppc <- %s (%s)", col, source)

    # --- fallback for municipalities the primary source cannot price ---------
    # [DECISION] gdp.fill_missing_from_gridded (default false). When true, a
    # municipality whose canonical gdppc is MISSING (no CE data at all --
    # confidentiality suppression) or NON-POSITIVE (censal VA <= 0, so a log
    # is undefined) takes the OTHER source's value, and gdppc_source records
    # which rows are affected. This mixes measurement constructions within one
    # column -- the honest alternative is leaving them NA, which is the
    # default. It exists because a panel with no NAs was explicitly requested,
    # and dropping whole municipalities (with their real flows) to achieve
    # that would cost far more than a flagged imputation for a handful of
    # units.
    fallback_col = "gdppc_grid" if source == "censos_economicos" else "gdppc_ce"
    gdp["gdppc_source"] = np.where(gdp["gdppc"].notna(), source, "missing")
    if cfg["gdp"].get("fill_missing_from_gridded", False):
        v0 = gdp["gdppc"]
        need = (v0.isna() | (v0 <= 0)) & gdp[fallback_col].notna() & (gdp[fallback_col] > 0)
        if bool(need.any()):
            filled = gdp.loc[need, "cvegeo"].tolist()
            gdp.loc[need, "gdppc"] = gdp.loc[need, fallback_col]
            gdp.loc[need, "gdppc_source"] = f"{fallback_col}_fallback"
            log.warning("      FALLBACK: %d municipality/ies took %s because the "
                        "canonical source is missing or <= 0 there: %s",
                        int(need.sum()), fallback_col, filled)
            log.warning("      flagged in gdppc_source -- these rows mix "
                        "measurement constructions; exclude them for a "
                        "sensitivity check.")

    # --- squares and logs, named unambiguously -------------------------------
    v = gdp["gdppc"]
    nonpos = v.notna() & (v <= 0)
    if bool(nonpos.any()):
        log.warning("      %d municipality/ies have gdppc <= 0; logs set NA "
                    "(not floored -- flooring would invent a value)",
                    int(nonpos.sum()))

    gdp["gdppc_sq"] = v ** 2                                    # a square
    gdp["log_gdppc"] = np.where(v > 0, np.log(v.where(v > 0)), np.nan)
    gdp["log_gdppc_sq"] = gdp["log_gdppc"] ** 2                 # a SQUARED LOG

    log.info("      emitted gdppc, gdppc_sq (=gdppc^2), log_gdppc (=ln gdppc), "
             "log_gdppc_sq (=(ln gdppc)^2)")
    log.info("      NOT emitted: ln(gdppc^2) == 2*ln(gdppc), exactly collinear "
             "with log_gdppc")

    assert_cvegeo_valid(gdp["cvegeo"], "gdp.cvegeo")
    if bool(gdp["cvegeo"].duplicated().any()):
        raise PipelineError("duplicate cvegeo in gdp table")

    missingness_table(gdp, log)
    write_interim(cfg, gdp, "gdp", log)
    log_rows(log, "STEP TOTAL 04_gdp", len(pop), len(gdp))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_step(main, STEP))



