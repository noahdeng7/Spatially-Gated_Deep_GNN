"""
07_assemble.py -- join everything into the final dyadic panel.

Input  : every table in data/interim/
Output : data/processed/mx_dyadic.parquet
         data/processed/mx_dyadic_codebook.md
         reports/assembly_validation.md
         reports/orphans.csv (only if orphans exist)

WHY THE ZEROS ARE HERE
----------------------
The panel is the FULL cartesian product of municipalities, with zero-flow pairs
explicitly present. This roughly triples the file size and it is not optional.

Gravity models are estimated on flows with many zeros. Dropping them and running
OLS on log(flow) conditions on flow > 0, which is selection on the dependent
variable -- the resulting distance elasticity is biased toward zero, and by an
amount that depends on how many zeros were silently dropped. PPML (Santos Silva
& Tenreyro) is the standard fix and it *requires* the zeros to be present. A
panel that ships only the positive cells has made that choice for its user,
silently, and the user usually cannot tell.

    ~2,470 municipalities -> ~6.1M ordered pairs. Fine in parquet.

JOIN DISCIPLINE
---------------
Every join in this file is a LEFT join onto the cartesian spine, and every one
reports what failed to match. There are no inner joins anywhere in this pipeline
by design: an inner join deletes non-matching rows silently, and a silent inner
join is the single most likely way this pipeline would produce quiet nonsense.
Orphan codes stop the build (config `validation.fail_on_orphans`).

Idempotent.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CVEGEO_DTYPE, PipelineError, ROOT, assert_cvegeo_valid, ensure_dirs,
    get_logger, load_config, log_rows, missingness_table, read_interim,
    report_path, rel, resolve, run_step,
)

STEP = "07_assemble"


# ---------------------------------------------------------------------------
# codebook specification
# ---------------------------------------------------------------------------
# (definition, units, source, vintage, construction). Rendered to markdown at
# the end. Kept as data rather than prose so a column can never be added to the
# panel without a codebook entry -- assemble() asserts the two agree.
# Internal (join-time) name -> shipped name. The pipeline speaks orig/dest
# internally because that matches the census questions; the shipped schema
# uses src_/dst_ prefixes per the dataset owner's convention. Applied once in
# main(), immediately before the codebook completeness check.
FINAL_NAMES: dict[str, str] = {
    "orig": "src",
    "dest": "dst",
    "pop_orig": "src_pop",
    "pop_dest": "dst_pop",
    "pop_orig_2015": "src_pop_2015",
    "pop_orig_2020": "src_pop_2020",
    "pop_dest_2015": "dst_pop_2015",
    "pop_dest_2020": "dst_pop_2020",
    "pop_orig_vintage": "src_pop_vintage",
    "gdppc_orig": "src_gdppc",
    "gdppc_dest": "dst_gdppc",
    "gdppc_orig_sq": "src_gdppc_sq",
    "gdppc_dest_sq": "dst_gdppc_sq",
    "log_gdppc_orig": "src_log_gdppc",
    "log_gdppc_dest": "dst_log_gdppc",
    "log_gdppc_orig_sq": "src_log_gdppc_sq",
    "log_gdppc_dest_sq": "dst_log_gdppc_sq",
    "gdppc_source_orig": "src_gdppc_source",
    "gdppc_source_dest": "dst_gdppc_source",
    "hh_income_orig": "src_hh_income",
    "hh_income_dest": "dst_hh_income",
    "hh_income_sq_orig": "src_hh_income_sq",
    "hh_income_sq_dest": "dst_hh_income_sq",
    "log_hh_income_orig": "src_log_hh_income",
    "log_hh_income_dest": "dst_log_hh_income",
    "log_hh_income_sq_orig": "src_log_hh_income_sq",
    "log_hh_income_sq_dest": "dst_log_hh_income_sq",
    "hh_income_cv_orig": "src_hh_income_cv",
    "hh_income_cv_dest": "dst_hh_income_cv",
    "hh_income_source_orig": "src_hh_income_source",
    "hh_income_source_dest": "dst_hh_income_source",
    "temp_orig": "src_temp",
    "temp_dest": "dst_temp",
    "precip_orig": "src_precip",
    "precip_dest": "dst_precip",
    "dempres_orig": "src_dempres",
    "dempres_dest": "dst_dempres",
    "pop_youth_orig": "src_pop_youth",
    "pop_youth_dest": "dst_pop_youth",
    "pop_working_age_orig": "src_pop_working_age",
    "pop_working_age_dest": "dst_pop_working_age",
}

COLUMN_SPEC: dict[str, tuple[str, str, str, str, str]] = {
    "src": (
        "Source (origin) municipality: residence 5 years before the census.",
        "5-char code", "INEGI Marco Geoestadistico", "2020",
        "CVEGEO: 2-digit state + 3-digit municipality, zero-padded, stored as "
        "string. Harmonized to the 2020 municipal universe via the 2015-2020 "
        "crosswalk.",
    ),
    "dst": (
        "Destination municipality: residence at census date.",
        "5-char code", "INEGI Marco Geoestadistico", "2020",
        "As `src`.",
    ),
    "migrants": (
        "Migrants from src to dst over the 5-year window, scaled to "
        "population.",
        "persons", "INEGI Censo 2020, muestra ampliada (10%)", "2015-2020",
        "Sum of the census expansion factor FACTOR over person records whose "
        "current municipality is `dst` and whose municipality of residence 5 "
        "years earlier was `src`. Zero where no such record exists.",
    ),
    "migrants_unweighted": (
        "Raw count of sampled person records for this pair, before expansion.",
        "person records", "INEGI Censo 2020, muestra ampliada (10%)", "2015-2020",
        "Unweighted cell count. Retained for small-cell diagnostics: a weighted "
        "flow of 400 built from 3 records is a very different object from one "
        "built from 40.",
    ),
    "flow_observed": (
        "Whether this pair appears at all in the census sample.",
        "boolean", "derived", "2015-2020",
        "True iff migrants_unweighted > 0. False marks a cell whose zero was "
        "never directly observed.",
    ),
    "zero_class": (
        "Heuristic classification of zero cells.",
        "category", "derived", "2015-2020",
        "observed_positive | sampling_zero_plausible | structural_zero_likely. "
        "See the caveat in the Zeros section below -- this is a heuristic, not "
        "an identified quantity.",
    ),
    "src_pop": (
        "Source population (canonical; vintage set by config).",
        "persons", "INEGI census tabulados / CONAPO projections", "see src_pop_vintage",
        "Full-count municipal total. Which vintage this aliases is controlled by "
        "`population.origin_pop_year` and recorded in `src_pop_vintage`.",
    ),
    "dst_pop": (
        "Destination population (canonical).", "persons",
        "INEGI census tabulados", "2020",
        "Full-count municipal total from the 2020 census tabulados.",
    ),
    "src_pop_2015": (
        "Source population at the START of the migration window.",
        "persons", "CONAPO municipal projections", "2015",
        "The theoretically preferred exposure measure: the stock at risk of "
        "emigrating. Unlike the 2020 figure it is not mechanically depleted by "
        "the flow being modelled.",
    ),
    "src_pop_2020": (
        "Source population at the census date.", "persons",
        "INEGI census tabulados", "2020",
        "Note this is depleted by the very outflow being modelled; using it as "
        "the exposure term puts the dependent variable on both sides.",
    ),
    "dst_pop_2015": (
        "Destination population at the start of the window.", "persons",
        "CONAPO municipal projections", "2015",
        "Included for src/dst symmetry; the canonical dst_pop is the 2020 "
        "figure (the stock that received the flow).",
    ),
    "dst_pop_2020": (
        "Destination population at the census date.", "persons",
        "INEGI census tabulados", "2020", "Full-count municipal total.",
    ),
    "src_pop_vintage": (
        "Which vintage `src_pop` aliases.", "year", "config", "n/a",
        "Stamped from `population.origin_pop_year` so the panel is "
        "self-describing.",
    ),
    "dist_geodesic_km": (
        "Geodesic distance between population-weighted centroids.",
        "kilometres", "INEGI Marco Geoestadistico + WorldPop 2020", "2020",
        "Karney geodesic on the WGS84 ellipsoid (pyproj.Geod) between "
        "population-weighted centroids. Centroids are weighted by the WorldPop "
        "2020 100m grid, not geometric.",
    ),
    "dist_greatcircle_km": (
        "Great-circle (spherical) distance between the same centroids.",
        "kilometres", "as above", "2020",
        "Haversine on a sphere of radius 6371.0088 km. Retained because the "
        "literature is split on which measure it uses and the difference is "
        "large enough to matter when comparing coefficients.",
    ),
    "src_gdppc": (
        "Source GDP per capita, PPP-adjusted. **ESTIMATE, not official.**",
        "2017 international $", "INEGI Censos Economicos 2019 (or gridded)", "2019",
        "Municipal gross value added / 2020 population, deflated and converted "
        "at the World Bank PA.NUS.PPP factor. Mexico publishes no official "
        "municipal GDP. Understates informal and subsistence activity.",
    ),
    "dst_gdppc": (
        "Destination GDP per capita, PPP-adjusted. **ESTIMATE.**",
        "2017 international $", "as src_gdppc", "2019", "As src_gdppc.",
    ),
    "src_gdppc_sq": (
        "Square of source GDP per capita.", "(2017 international $)^2",
        "derived", "2019",
        "src_gdppc ** 2. A SQUARE OF A LEVEL. Distinct from src_log_gdppc_sq.",
    ),
    "dst_gdppc_sq": (
        "Square of destination GDP per capita.", "(2017 international $)^2",
        "derived", "2019", "dst_gdppc ** 2. As src_gdppc_sq.",
    ),
    "src_log_gdppc": (
        "Natural log of source GDP per capita.", "log(2017 int$)", "derived", "2019",
        "ln(src_gdppc). NA where src_gdppc <= 0 (not floored -- flooring would "
        "invent a value).",
    ),
    "dst_log_gdppc": (
        "Natural log of destination GDP per capita.", "log(2017 int$)",
        "derived", "2019", "ln(dst_gdppc).",
    ),
    "src_log_gdppc_sq": (
        "Square of the log of source GDP per capita.", "log(2017 int$)^2",
        "derived", "2019",
        "(ln src_gdppc) ** 2 -- a SQUARED LOG, the quadratic term for a "
        "log-linear specification. NOT ln(gdppc^2), which equals 2*ln(gdppc) "
        "and is exactly collinear with src_log_gdppc.",
    ),
    "dst_log_gdppc_sq": (
        "Square of the log of destination GDP per capita.", "log(2017 int$)^2",
        "derived", "2019", "(ln dst_gdppc) ** 2. As src_log_gdppc_sq.",
    ),
    "src_gdppc_source": (
        "Which construction produced src_gdppc for this origin.",
        "category", "config + fallback logic in 04_gdp.py", "2019",
        "censos_economicos | gdppc_grid_fallback | missing. The fallback value "
        "marks municipalities whose censal value added is absent "
        "(confidentiality) or non-positive, priced from the gridded product "
        "instead -- a DIFFERENT measurement construction. Exclude flagged rows "
        "as a sensitivity check.",
    ),
    "dst_gdppc_source": (
        "Which construction produced dst_gdppc for this destination.",
        "category", "as src_gdppc_source", "2019", "As src_gdppc_source.",
    ),
    "src_hh_income": (
        "Source mean household income. **MODELLED ESTIMATE, a MEAN not a median.**",
        "nominal 2020 pesos / quarter / household", "INEGI ICMM 2020", "2020",
        "Ingreso Corriente Promedio Trimestral por Hogar 2020: mean current "
        "quarterly household income, by small-area estimation from ENIGH 2020 "
        "(nueva serie) + Censo 2020 + administrative records. The census does not "
        "ask income, so this is modelled, not observed. Nominal 2020 pesos, NOT "
        "PPP-adjusted -- not comparable in level to src_gdppc (2017 int$).",
    ),
    "dst_hh_income": (
        "Destination mean household income. **MODELLED ESTIMATE, a MEAN.**",
        "nominal 2020 pesos / quarter / household", "INEGI ICMM 2020", "2020",
        "As src_hh_income.",
    ),
    "src_hh_income_sq": (
        "Square of source mean household income.", "(pesos/quarter)^2",
        "derived", "2020",
        "src_hh_income ** 2. A SQUARE OF A LEVEL. Distinct from "
        "src_log_hh_income_sq.",
    ),
    "dst_hh_income_sq": (
        "Square of destination mean household income.", "(pesos/quarter)^2",
        "derived", "2020", "dst_hh_income ** 2. As src_hh_income_sq.",
    ),
    "src_log_hh_income": (
        "Natural log of source mean household income.", "log(pesos/quarter)",
        "derived", "2020", "ln(src_hh_income). Always defined (income > 0).",
    ),
    "dst_log_hh_income": (
        "Natural log of destination mean household income.", "log(pesos/quarter)",
        "derived", "2020", "ln(dst_hh_income).",
    ),
    "src_log_hh_income_sq": (
        "Square of the log of source mean household income.",
        "log(pesos/quarter)^2", "derived", "2020",
        "(ln src_hh_income) ** 2 -- a SQUARED LOG, the quadratic term for a "
        "log-linear specification. NOT ln(hh_income^2), which equals "
        "2*ln(hh_income) and is exactly collinear with src_log_hh_income.",
    ),
    "dst_log_hh_income_sq": (
        "Square of the log of destination mean household income.",
        "log(pesos/quarter)^2", "derived", "2020",
        "(ln dst_hh_income) ** 2. As src_log_hh_income_sq.",
    ),
    "src_hh_income_cv": (
        "Coefficient of variation of the source income estimate.",
        "percent", "INEGI ICMM 2020", "2020",
        "Reliability of the small-area estimate. INEGI convention: <15% reliable, "
        "15-30% caution, >30% not recommended. In the 2020 vintage every value is "
        "below 9%. NA for a municipality harmonized from a split "
        "(hh_income_source = icmm_2020_harmonized), whose combined estimate has no "
        "clean standard error.",
    ),
    "dst_hh_income_cv": (
        "Coefficient of variation of the destination income estimate.",
        "percent", "INEGI ICMM 2020", "2020", "As src_hh_income_cv.",
    ),
    "src_hh_income_source": (
        "Provenance of src_hh_income for this origin.",
        "category", "INEGI ICMM 2020 + fold logic in 08_income.py", "2020",
        "icmm_2020 | icmm_2020_harmonized. The harmonized value marks a "
        "municipality that absorbed a post-2015 split child; its income is a "
        "population-weighted mean of the two 2020 municipalities (income is "
        "intensive, so it is averaged, not summed) and its se/ci/cv are NA.",
    ),
    "dst_hh_income_source": (
        "Provenance of dst_hh_income for this destination.",
        "category", "as src_hh_income_source", "2020", "As src_hh_income_source.",
    ),
    "src_temp": (
        "Mean annual temperature at source.",
        "degrees Celsius", "see climate_source column", "see climate_period",
        "Construction depends on climate_source. worldclim/chelsa: mean of 12 "
        "monthly normals, each a WorldPop-2020 population-weighted zonal mean. "
        "era5_land_gee: annual-mean of daily-mean ERA5-Land temperature, averaged "
        "over the climate_period years, as a WorldPop-2020 population-weighted "
        "municipal mean computed on Google Earth Engine (converted from degF). "
        "Weighted because temperature is intensive: an unweighted polygon mean "
        "lets empty territory outvote where the population actually is.",
    ),
    "dst_temp": (
        "Mean annual temperature at destination.",
        "degrees Celsius", "see climate_source column", "see climate_period",
        "As src_temp.",
    ),
    "src_precip": (
        "Mean annual total precipitation at source.",
        "millimetres/year", "see climate_source column", "see climate_period",
        "Construction depends on climate_source. worldclim/chelsa: SUM of 12 "
        "monthly normals (a total, not a mean), each a WorldPop-2020 population-"
        "weighted zonal mean. era5_land_gee: annual-SUM of daily ERA5-Land "
        "precipitation, averaged over the climate_period years, as a WorldPop-2020 "
        "population-weighted municipal mean on Google Earth Engine (converted "
        "from metres).",
    ),
    "dst_precip": (
        "Mean annual total precipitation at destination.",
        "millimetres/year", "see climate_source column", "see climate_period",
        "As src_precip.",
    ),
    "climate_source": (
        "Which climate product produced the temp and precip columns.",
        "category", "config (climate.panel_source)", "n/a",
        "worldclim | chelsa | terraclimate (population-weighted zonal normals) | "
        "era5_land_gee (ERA5-Land daily reanalysis, GEE population-weighted zonal "
        "mean). All four are population-weighted on the WorldPop 2020 grid.",
    ),
    "climate_period": (
        "Period the temp and precip columns describe.", "year range", "config", "n/a",
        "1970-2000 for WorldClim v2.1 (a long-run normal, NOT the migration "
        "window). 2016-2020 for era5_land_gee (a window mean -- conditions during "
        "the migration window).",
    ),
    "src_dempres": (
        "Youth share of the working-age population at source (DemPres).",
        "ratio (0-1)", "INEGI Censo 2020", "2020",
        "src_pop_youth / src_pop_working_age. Bands are configurable; see "
        "youth_band and working_age_band. Built at MUNICIPALITY level -- see "
        "the deviation note.",
    ),
    "dst_dempres": (
        "Youth share of the working-age population at destination.",
        "ratio (0-1)", "INEGI Censo 2020", "2020", "As src_dempres.",
    ),
    "src_pop_youth": (
        "Numerator of src_dempres.", "persons", "INEGI Censo 2020", "2020",
        "Source population within the configured youth band.",
    ),
    "dst_pop_youth": (
        "Numerator of dst_dempres.", "persons", "INEGI Censo 2020", "2020",
        "Destination population within the configured youth band.",
    ),
    "src_pop_working_age": (
        "Denominator of src_dempres.", "persons", "INEGI Censo 2020", "2020",
        "Source population within the configured working-age band.",
    ),
    "dst_pop_working_age": (
        "Denominator of dst_dempres.", "persons", "INEGI Censo 2020", "2020",
        "Destination population within the configured working-age band.",
    ),
    "youth_band": (
        "Youth age band in force.", "age range", "config", "n/a",
        "e.g. '15-29'. Stamped in so the panel is self-describing.",
    ),
    "working_age_band": (
        "Working-age band in force.", "age range", "config", "n/a", "e.g. '15-64'.",
    ),
}


# ---------------------------------------------------------------------------
# spine
# ---------------------------------------------------------------------------

def build_spine(cfg, universe: pd.Series, log) -> pd.DataFrame:
    """Full ordered cartesian product, diagonal optionally excluded."""
    codes = np.sort(universe.dropna().unique().astype(str))
    n = len(codes)
    log.info("      municipal universe: %s municipalities -> %s ordered pairs",
             f"{n:,}", f"{n * n:,}")

    o = np.repeat(codes, n)
    d = np.tile(codes, n)
    spine = pd.DataFrame({"orig": pd.array(o, dtype=CVEGEO_DTYPE),
                          "dest": pd.array(d, dtype=CVEGEO_DTYPE)})

    if not cfg["assemble"].get("include_diagonal", False):
        n_before = len(spine)
        spine = spine[spine["orig"].to_numpy() != spine["dest"].to_numpy()].reset_index(drop=True)
        log_rows(log, "drop diagonal from panel (kept in interim/flows_diagonal)",
                 n_before, len(spine))
    return spine


def left_join(spine: pd.DataFrame, right: pd.DataFrame, on, log, label: str,
              suffix: str = "") -> pd.DataFrame:
    """
    LEFT join with mandatory match reporting.

    There are no inner joins in this pipeline. Every non-match is counted and
    logged; the caller decides whether it is tolerable.
    """
    n_before = len(spine)
    right = right.copy()
    if suffix:
        right = right.rename(columns={c: f"{c}{suffix}" for c in right.columns
                                      if c not in (on if isinstance(on, list) else [on])})
    out = spine.merge(right, on=on, how="left", indicator="_m")
    matched = int((out["_m"] == "both").sum())
    out = out.drop(columns="_m")
    if len(out) != n_before:
        raise PipelineError(
            f"{label}: LEFT join changed the row count {n_before:,} -> {len(out):,}. "
            "The right-hand table has duplicate keys."
        )
    log.info("JOIN  %-34s matched=%-12s unmatched=%-12s (%.3f%% matched)",
             label, f"{matched:,}", f"{n_before - matched:,}",
             100.0 * matched / n_before if n_before else float("nan"))
    return out


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def check_orphans(cfg, flows: pd.DataFrame, universe: set[str], log) -> pd.DataFrame:
    """
    Every CVEGEO in the flow table must exist in the geometry table.

    Reported, never inner-joined away. This is the check that catches a failed
    boundary harmonization, a mis-parsed sentinel, or a leading zero lost to an
    integer cast -- all of which otherwise produce a smaller-but-plausible panel.
    """
    orphans = []
    for side in ("orig", "dest"):
        codes = set(flows[side].dropna().astype(str).unique())
        missing = sorted(codes - universe)
        for c in missing:
            n_cells = int((flows[side].astype(str) == c).sum())
            wt = float(flows.loc[flows[side].astype(str) == c, "migrants"].sum())
            orphans.append(dict(code=c, side=side, n_cells=n_cells,
                                weighted_migrants=wt))

    out = pd.DataFrame(orphans, columns=["code", "side", "n_cells", "weighted_migrants"])
    if len(out):
        log.error("=" * 70)
        log.error("ORPHAN CODES: %d code(s) in the flow table are absent from the "
                  "2020 geometry", len(out))
        log.error("=" * 70)
        for _, r in out.head(30).iterrows():
            log.error("  %s  side=%-5s cells=%-6s weighted_migrants=%s",
                      r["code"], r["side"], f"{r['n_cells']:,}",
                      f"{r['weighted_migrants']:,.0f}")
        log.error("These would be DELETED by an inner join. Likely causes, in "
                  "order of probability:")
        log.error("  1. a municipality split with no crosswalk row "
                  "(data/raw/crosswalk_municipios_2015_2020.csv)")
        log.error("  2. an unhandled sentinel in flows.special_codes")
        log.error("  3. a leading zero lost to an integer cast upstream")
        p = report_path(cfg, "orphans.csv")
        out.to_csv(p, index=False)
        log.error("WROTE %s", rel(p))
    else:
        log.info("ORPHAN CHECK  clean: every flow code exists in the 2020 geometry")
    return out


def classify_zeros(panel: pd.DataFrame, log) -> pd.DataFrame:
    """
    Distinguish, as far as the data allows, sampling zeros from structural ones.

    HONEST CAVEAT, repeated in the codebook: with a single 10% cross-section
    these are not separately identified. A pair with a true flow of 4 people has
    a ~66% chance of contributing zero sampled records, and is indistinguishable
    from a pair with a true flow of 0.

    The heuristic used: a zero is 'sampling_zero_plausible' when BOTH endpoints
    are demonstrably active in migration (the origin sent someone somewhere and
    the destination received someone from somewhere), so the machinery for a
    flow between them exists. It is 'structural_zero_likely' when at least one
    endpoint shows no migration activity at all in the sample. That is a weak
    signal and should be treated as a diagnostic, not a covariate.
    """
    panel["flow_observed"] = panel["migrants_unweighted"] > 0

    active_orig = set(panel.loc[panel["flow_observed"], "orig"].astype(str).unique())
    active_dest = set(panel.loc[panel["flow_observed"], "dest"].astype(str).unique())

    o_active = panel["orig"].astype(str).isin(active_orig).to_numpy()
    d_active = panel["dest"].astype(str).isin(active_dest).to_numpy()

    panel["zero_class"] = np.where(
        panel["flow_observed"].to_numpy(), "observed_positive",
        np.where(o_active & d_active, "sampling_zero_plausible", "structural_zero_likely"),
    )

    counts = panel["zero_class"].value_counts()
    for k, v in counts.items():
        log.info("      zero_class %-26s %s (%.2f%%)", k, f"{v:,}", 100.0 * v / len(panel))
    return panel


# ---------------------------------------------------------------------------
# positive-only export
# ---------------------------------------------------------------------------

def write_positive_only(cfg, panel: pd.DataFrame, log) -> Path:
    """
    Write a CSV containing only dyads with a positive flow.

    A companion to the full panel, not a replacement. Useful for inspection,
    mapping, network analysis, and any tool that will not hold 6.1M rows.

    NOT suitable for gravity estimation. This is stated in the log, in the
    codebook, and in a header comment inside the CSV itself -- a filtered file
    tends to outlive the conversation that produced it, and the next person to
    open it will have no idea the zeros were ever there.

    Note this filters on `migrants > 0`, NOT on `zero_class`. Dropping only the
    "structural" zeros is deliberately not offered: structural and sampling
    zeros are not separately identified from a single cross-section, so a filter
    on that heuristic would delete real flows while looking principled.
    """
    acfg = cfg["assemble"]
    out = resolve(cfg, acfg.get("positive_only_output",
                                "data/processed/mx_dyadic_positive.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)

    n_in = len(panel)
    pos = panel[panel["migrants"] > 0].reset_index(drop=True)
    log_rows(log, "filter to positive flows (companion export)", n_in, len(pos))

    if len(pos) == 0:
        raise PipelineError(
            "positive-only export would be empty -- no dyad has migrants > 0. "
            "Something is wrong upstream in 01_flows."
        )

    gzip_it = bool(acfg.get("positive_only_gzip", False))
    if gzip_it and out.suffix != ".gz":
        out = out.with_suffix(out.suffix + ".gz")

    pos.to_csv(out, index=False, encoding="utf-8",
               compression="gzip" if gzip_it else None)

    size_mb = out.stat().st_size / 1e6
    share = 100.0 * len(pos) / n_in if n_in else float("nan")
    log.info("WROTE %s  rows=%s (%.3f%% of panel)  size=%.1f MB",
             rel(out), f"{len(pos):,}", share, size_mb)
    log.info("      positive-only export is for inspection and descriptive use. "
             "Estimate gravity on the FULL panel -- see the codebook.")
    return out


# ---------------------------------------------------------------------------
# codebook
# ---------------------------------------------------------------------------

def _climate_limitation(stats: dict) -> list[str]:
    """Codebook limitation #2, written to match the active climate provider."""
    if stats.get("climate_is_window"):
        return [
            "2. **Climate is the migration-window mean, at ~11 km resolution.**",
            "   `climate_source` = era5_land_gee: ERA5-Land daily reanalysis averaged",
            "   over `climate_period` (2016-2020) -- conditions *during* the window",
            "   rather than a long-run normal, which is the right signal for a",
            "   migration model. The municipal value is a WorldPop-2020 population-",
            "   weighted mean, so a large municipality's empty interior does not",
            "   outvote the cells where people live. The remaining caveat is",
            "   resolution: ERA5-Land's 0.1 deg (~11 km) grid is much coarser than",
            "   WorldClim's 30-arcsec (~1 km), so municipalities smaller than a few",
            "   grid cells carry little within-municipality variation to weight, and",
            "   sharp orographic gradients in the sierras are smoothed away.",
        ]
    return [
        "2. **Climate normals are not window conditions.** WorldClim/CHELSA",
        "   normals describe long-run climate, a locational attribute people sort",
        "   on, not weather shocks during 2015-2020. Different question, different",
        "   identification. See `reports/climate_diagnostics.md`.",
    ]


def write_codebook(cfg, panel: pd.DataFrame, stats: dict, log) -> Path:
    lines = [
        "# `mx_dyadic.parquet` -- codebook",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"by `src/07_assemble.py`.",
        "",
        f"- Rows: **{len(panel):,}** (ordered origin-destination pairs)",
        f"- Municipalities: **{stats['n_municipios']:,}**",
        f"- Diagonal included: **{cfg['assemble'].get('include_diagonal', False)}**",
        f"- Zero-flow pairs included: **{cfg['assemble'].get('include_zero_flows', True)}**",
        f"- Census round: **{cfg['project']['census_year']}**, migration window "
        f"**{cfg['project']['census_year'] - cfg['project']['migration_window_years']}-"
        f"{cfg['project']['census_year']}**",
        "",
        "## Unit of observation",
        "",
        "One row per **ordered** (origin, destination) pair. (A, B) and (B, A) are",
        "distinct rows with distinct flows. The diagonal (A, A) is excluded from",
        "this file and available in `data/interim/flows_diagonal.parquet`.",
        "",
        "## Columns",
        "",
        "| column | definition | units | source | vintage |",
        "|---|---|---|---|---|",
    ]
    for col in panel.columns:
        spec = COLUMN_SPEC.get(col)
        if spec:
            d, u, s, v, _ = spec
            lines.append(f"| `{col}` | {d} | {u} | {s} | {v} |")
        else:
            lines.append(f"| `{col}` | *(undocumented -- add to COLUMN_SPEC)* | | | |")

    lines += ["", "## Construction detail", ""]
    for col in panel.columns:
        spec = COLUMN_SPEC.get(col)
        if not spec:
            continue
        lines += [f"### `{col}`", "", f"*{spec[0]}*", "",
                  f"- **Units:** {spec[1]}", f"- **Source:** {spec[2]}",
                  f"- **Vintage:** {spec[3]}", f"- **Construction:** {spec[4]}", ""]

    lines += ["## The zeros", ""]

    if stats.get("main_filtered"):
        n_full = stats.get("n_full_cartesian", 0)
        dropped = n_full - len(panel)
        lines += [
            "> ### ⚠ THIS PANEL HAS BEEN FILTERED TO POSITIVE FLOWS",
            ">",
            f"> `assemble.drop_zero_flows_from_main_panel` was set **true**, so",
            f"> **{dropped:,}** zero-flow dyads ({100.0 * dropped / n_full:.2f}% of the",
            f"> full cartesian product of {n_full:,}) were removed before writing.",
            ">",
            "> **Do not estimate a gravity model on this file.** It is conditioned",
            "> on the dependent variable being positive, which is selection on the",
            "> outcome. The distance elasticity will be biased toward zero by an",
            "> amount that depends on how many cells were dropped — and nearly all",
            "> of them were. PPML exists specifically to use the zeros.",
            ">",
            "> To restore the complete panel, set that flag back to `false` and",
            "> re-run `python src/07_assemble.py`. Nothing upstream needs rebuilding.",
            "",
        ]
    else:
        lines += [
            f"Of {len(panel):,} pairs, **{stats['n_positive']:,}** "
            f"({100.0 * stats['n_positive'] / len(panel):.3f}%) have a positive flow.",
            "",
            "The zeros are here on purpose. Estimating gravity on log(flow) after",
            "dropping zeros conditions on the dependent variable being positive, which",
            "biases the distance elasticity toward zero by an amount that depends on",
            "how many cells were dropped. PPML needs the zeros present.",
            "",
        ]

    if stats.get("positive_only_path"):
        lines += [
            "### Positive-only companion file",
            "",
            f"`{Path(stats['positive_only_path']).name}` contains only the dyads",
            "with a positive flow, as CSV. It is a **view** of this panel, not a",
            "separate dataset — same columns, same construction, fewer rows.",
            "",
            "Use it for inspection, mapping, network analysis, or anything that",
            "will not hold several million rows. Do not estimate gravity on it,",
            "for the reason given above.",
            "",
        ]

    lines += [
        "### `zero_class` is a heuristic, not an identified quantity",
        "",
        "Structural zeros (no flow is possible) and sampling zeros (a flow exists",
        "but the 10% sample missed it) are **not separately identified** from a",
        "single cross-section. A true flow of 4 people has roughly a 66% chance of",
        "contributing zero sampled records and is indistinguishable from a true",
        "zero. `zero_class` marks a pair as `sampling_zero_plausible` when both",
        "endpoints are demonstrably active in migration elsewhere in the data.",
        "Treat it as a diagnostic for sensitivity analysis, not as a covariate.",
        "",
        "## Known limitations",
        "",
        "1. **Municipal GDP per capita is constructed, not official.** Mexico",
        "   publishes no municipal GDP series. Censal value added understates",
        "   subsistence agriculture and informal activity, which concentrate in",
        "   poor rural origins -- so the measure is biased downward precisely where",
        "   the model is most sensitive, and the bias correlates with the outcome.",
        "   See `reports/gdp_source_comparison.md`.",
        *_climate_limitation(stats),
        "3. **Boundary harmonization folds split municipalities into parents.**",
        "   Balanced panel, coarser geography in a handful of places. See",
        "   `README.md#boundary-harmonization`.",
        "4. **`src_dempres` is municipal, not national**, deviating from the",
        "   literal specification. A country-level measure is constant here and",
        "   identifies nothing. See `reports/demography_bands.md`.",
        "5. **Flows are a 5-year stock question, not a flow count.** The census",
        "   asks where you lived 5 years ago. Return migrants who left and came",
        "   back appear as non-migrants; multiple moves within the window collapse",
        "   to one origin-destination pair; people who died or emigrated abroad",
        "   before the census are absent entirely.",
        "6. **Household income is a modelled small-area estimate, and a mean, not",
        "   a median.** `src_hh_income`/`dst_hh_income` come from INEGI's ICMM",
        "   2020, built by small-area estimation from ENIGH 2020 plus the census —",
        "   the census does not ask income, so no observed municipal income exists.",
        "   It is a mean (INEGI publishes no municipal median) in nominal 2020",
        "   quarterly pesos, NOT PPP-adjusted, so its level is not comparable to",
        "   `src_gdppc` (2017 int$). Judge reliability with `*_hh_income_cv`. See",
        "   `reports/income_summary.md`.",
        "",
        "## Reproduction",
        "",
        "```",
        "make all          # full pipeline from data/raw/",
        "make test         # unit tests",
        "```",
        "",
        f"Config in force is recorded at `{cfg.get('_config_path', 'config.yaml')}`.",
        "",
    ]
    p = resolve(cfg, cfg["assemble"]["codebook"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("WROTE codebook -> %s", rel(p))
    return p


# ---------------------------------------------------------------------------
# climate provider
# ---------------------------------------------------------------------------

def _load_climate(cfg, log):
    """
    Load the municipal climate table for the panel, normalized to degC / mm.

    Which product feeds the panel is `config.climate.panel_source`:

      worldclim_zonal -> interim/climate.parquet (05_climate.py). Population-
                         weighted zonal mean of a long-run normal (1970-2000 for
                         WorldClim). Already degC / mm.
      era5_gee        -> interim/climate_gee.parquet (src/climate_loader.py).
                         ERA5-Land daily reanalysis, 2016-2020 mean, reduced to
                         municipal polygons on Google Earth Engine as a WorldPop-
                         2020 population-weighted mean. Native degF / m, converted
                         HERE so the panel and codebook stay metric.

    Returns (df, is_window_climate). `df` carries cvegeo, temp_c, precip_mm,
    climate_source, climate_period; it is None if the configured table is absent
    (the panel then ships without climate columns, loudly). `is_window_climate`
    drives the codebook: ERA5 describes the migration window, WorldClim does not.
    """
    provider = (cfg.get("climate", {}) or {}).get("panel_source", "worldclim_zonal")

    if provider == "era5_gee":
        try:
            c = read_interim(cfg, "climate_gee", log)
        except PipelineError:
            log.warning("interim/climate_gee.parquet absent -- climate columns will "
                        "be missing. Run `python src/climate_loader.py` first.")
            return None, None
        # Period is owned by the loader (its CLIMATE_YEARS), so there is one place
        # to change the window and no risk of the stamp drifting from the data.
        from climate_loader import CLIMATE_YEARS  # noqa: E402
        period = f"{CLIMATE_YEARS[0]}-{CLIMATE_YEARS[-1]}"
        out = pd.DataFrame({
            "cvegeo": c["cvegeo"].astype(CVEGEO_DTYPE),
            "temp_c": (c["temp"].astype("float64") - 32.0) / 1.8,   # degF -> degC
            "precip_mm": c["precip"].astype("float64") * 1000.0,     # m    -> mm
        })
        out["climate_source"] = "era5_land_gee"
        out["climate_period"] = period
        log.info("CHOICE climate.panel_source = era5_gee (ERA5-Land daily, GEE zonal "
                 "mean, %s window mean). Converted degF->degC, m->mm.", period)
        return out, True

    if provider in ("worldclim_zonal", "worldclim", "zonal"):
        try:
            c = read_interim(cfg, "climate", log)
        except PipelineError:
            log.warning("interim/climate.parquet absent -- climate columns will be "
                        "missing. Run `python src/05_climate.py` first.")
            return None, None
        out = c[["cvegeo", "temp_mean_c", "precip_total_mm",
                 "climate_source", "climate_period"]].rename(
            columns={"temp_mean_c": "temp_c", "precip_total_mm": "precip_mm"})
        out["cvegeo"] = out["cvegeo"].astype(CVEGEO_DTYPE)
        log.info("CHOICE climate.panel_source = worldclim_zonal (05_climate; "
                 "population-weighted long-run normals)")
        return out, False

    raise PipelineError(
        f"unknown climate.panel_source: {provider!r} "
        "(expected 'worldclim_zonal' or 'era5_gee')")


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

    flows = read_interim(cfg, "flows_dyadic", log)
    pop = read_interim(cfg, "population", log)
    dist = read_interim(cfg, "distance", log)

    optional = {}
    for name in ("gdp", "income", "demography"):
        try:
            optional[name] = read_interim(cfg, name, log)
        except PipelineError:
            log.warning("interim/%s.parquet absent -- its columns will be missing "
                        "from the panel. Run src/0*_%s.py.", name, name)

    # Climate has two possible providers (config.climate.panel_source); load the
    # selected one, normalized to degC / mm. climate_is_window drives the codebook.
    climate, climate_is_window = _load_climate(cfg, log)

    # --- universe ------------------------------------------------------------
    try:
        universe_df = read_interim(cfg, "municipios_universe", log)
        universe_codes = universe_df["cvegeo"]
    except PipelineError:
        log.warning("municipios_universe absent; deriving the universe from the "
                    "population table instead")
        universe_codes = pop["cvegeo"]
    universe = set(universe_codes.dropna().astype(str).unique())

    # --- orphan check BEFORE any join ---------------------------------------
    orphans = check_orphans(cfg, flows, universe, log)
    if len(orphans) and cfg["validation"].get("fail_on_orphans", True):
        raise PipelineError(
            f"{len(orphans)} orphan code(s) found; see reports/orphans.csv.\n"
            "Refusing to continue: joining now would silently delete "
            f"{orphans['weighted_migrants'].sum():,.0f} weighted migrants.\n"
            "Fix the crosswalk or the sentinel handling, or set "
            "validation.fail_on_orphans=false to proceed with them dropped "
            "(and say so in your write-up)."
        )

    # --- spine ---------------------------------------------------------------
    spine = build_spine(cfg, universe_codes, log)
    n_spine = len(spine)

    # --- flows ---------------------------------------------------------------
    panel = left_join(spine, flows[["orig", "dest", "migrants", "migrants_unweighted"]],
                      ["orig", "dest"], log, "flows")
    n_positive = int(panel["migrants"].notna().sum())
    panel["migrants"] = panel["migrants"].fillna(0.0).astype("float64")
    panel["migrants_unweighted"] = panel["migrants_unweighted"].fillna(0).astype("int64")
    log.info("      %s pair(s) with a positive flow, %s explicit zeros",
             f"{n_positive:,}", f"{len(panel) - n_positive:,}")
    panel = classify_zeros(panel, log)

    # --- population ----------------------------------------------------------
    origin_year = int(cfg["population"]["origin_pop_year"])
    dest_year = int(cfg["population"]["dest_pop_year"])

    pop_o = pop[["cvegeo", "pop_origin_canonical", "pop_2015", "pop_2020"]].rename(
        columns={"cvegeo": "orig", "pop_origin_canonical": "pop_orig",
                 "pop_2015": "pop_orig_2015", "pop_2020": "pop_orig_2020"})
    panel = left_join(panel, pop_o, "orig", log, "population (origin)")

    pop_d = pop[["cvegeo", "pop_dest_canonical", "pop_2015", "pop_2020"]].rename(
        columns={"cvegeo": "dest", "pop_dest_canonical": "pop_dest",
                 "pop_2015": "pop_dest_2015", "pop_2020": "pop_dest_2020"})
    panel = left_join(panel, pop_d, "dest", log, "population (destination)")
    panel["pop_orig_vintage"] = origin_year
    log.info("CHOICE pop_orig aliases pop_%d; pop_dest aliases pop_%d",
             origin_year, dest_year)

    # --- distance ------------------------------------------------------------
    panel = left_join(panel,
                      dist[["orig", "dest", "dist_geodesic_km", "dist_greatcircle_km"]],
                      ["orig", "dest"], log, "distance")

    # --- gdp -----------------------------------------------------------------
    if "gdp" in optional:
        g = optional["gdp"]
        gcols = ["cvegeo", "gdppc", "gdppc_sq", "log_gdppc", "log_gdppc_sq"]
        if "gdppc_source" in g.columns:
            gcols.append("gdppc_source")
        panel = left_join(panel, g[gcols].rename(
            columns={"cvegeo": "orig", "gdppc": "gdppc_orig",
                     "gdppc_sq": "gdppc_orig_sq", "log_gdppc": "log_gdppc_orig",
                     "log_gdppc_sq": "log_gdppc_orig_sq",
                     "gdppc_source": "gdppc_source_orig"}),
            "orig", log, "gdp (origin)")
        panel = left_join(panel, g[gcols].rename(
            columns={"cvegeo": "dest", "gdppc": "gdppc_dest",
                     "gdppc_sq": "gdppc_dest_sq", "log_gdppc": "log_gdppc_dest",
                     "log_gdppc_sq": "log_gdppc_dest_sq",
                     "gdppc_source": "gdppc_source_dest"}),
            "dest", log, "gdp (destination)")

    # --- income (both sides) --------------------------------------------------
    if "income" in optional:
        inc = optional["income"]
        icols = ["cvegeo", "hh_income", "hh_income_sq", "log_hh_income",
                 "log_hh_income_sq", "hh_income_cv", "hh_income_source"]
        panel = left_join(panel, inc[icols].rename(
            columns={"cvegeo": "orig", "hh_income": "hh_income_orig",
                     "hh_income_sq": "hh_income_sq_orig",
                     "log_hh_income": "log_hh_income_orig",
                     "log_hh_income_sq": "log_hh_income_sq_orig",
                     "hh_income_cv": "hh_income_cv_orig",
                     "hh_income_source": "hh_income_source_orig"}),
            "orig", log, "income (origin)")
        panel = left_join(panel, inc[icols].rename(
            columns={"cvegeo": "dest", "hh_income": "hh_income_dest",
                     "hh_income_sq": "hh_income_sq_dest",
                     "log_hh_income": "log_hh_income_dest",
                     "log_hh_income_sq": "log_hh_income_sq_dest",
                     "hh_income_cv": "hh_income_cv_dest",
                     "hh_income_source": "hh_income_source_dest"}),
            "dest", log, "income (destination)")

    # --- climate (both sides) -------------------------------------------------
    if climate is not None:
        panel = left_join(panel, climate[["cvegeo", "temp_c", "precip_mm",
                                          "climate_source", "climate_period"]].rename(
            columns={"cvegeo": "orig", "temp_c": "temp_orig",
                     "precip_mm": "precip_orig"}),
            "orig", log, "climate (origin)")
        # Provenance stamps (climate_source/period) are constants across munis; one
        # copy is enough, so the destination join carries only the values.
        panel = left_join(panel, climate[["cvegeo", "temp_c",
                                          "precip_mm"]].rename(
            columns={"cvegeo": "dest", "temp_c": "temp_dest",
                     "precip_mm": "precip_dest"}),
            "dest", log, "climate (destination)")

    # --- demography (both sides) ----------------------------------------------
    if "demography" in optional:
        d = optional["demography"]
        panel = left_join(panel, d[["cvegeo", "dempres_orig", "pop_youth",
                                    "pop_working_age", "youth_band",
                                    "working_age_band"]].rename(
            columns={"cvegeo": "orig", "pop_youth": "pop_youth_orig",
                     "pop_working_age": "pop_working_age_orig"}),
            "orig", log, "demography (origin)")
        panel = left_join(panel, d[["cvegeo", "dempres_orig", "pop_youth",
                                    "pop_working_age"]].rename(
            columns={"cvegeo": "dest", "dempres_orig": "dempres_dest",
                     "pop_youth": "pop_youth_dest",
                     "pop_working_age": "pop_working_age_dest"}),
            "dest", log, "demography (destination)")

    # --- validation gauntlet -------------------------------------------------
    log.info("=" * 70)
    log.info("VALIDATION")
    log.info("=" * 70)

    assert_cvegeo_valid(panel["orig"], "panel.orig")
    assert_cvegeo_valid(panel["dest"], "panel.dest")

    dup = panel.duplicated(subset=["orig", "dest"])
    if bool(dup.any()):
        raise PipelineError(
            f"{int(dup.sum()):,} duplicate (src, dst) key(s) in the final panel"
        )
    log.info("  PASS  no duplicate (src, dst) keys")

    if not cfg["assemble"].get("include_diagonal", False):
        diag = (panel["orig"].to_numpy() == panel["dest"].to_numpy()).sum()
        if diag:
            raise PipelineError(f"{diag:,} diagonal row(s) leaked into the panel")
        log.info("  PASS  diagonal excluded")

    n_muni = len(universe)
    expected = n_muni * n_muni - (0 if cfg["assemble"].get("include_diagonal") else n_muni)
    if len(panel) != expected:
        raise PipelineError(
            f"panel has {len(panel):,} rows, expected {expected:,} "
            f"({n_muni:,}^2 minus diagonal). The cartesian product is incomplete."
        )
    log.info("  PASS  cartesian product complete: %s rows", f"{len(panel):,}")

    neg = panel["migrants"] < 0
    if bool(neg.any()):
        raise PipelineError(f"{int(neg.sum())} negative flow value(s)")
    log.info("  PASS  no negative flows")

    # --- final naming convention: src_/dst_ -----------------------------------
    # The interim tables and everything above speak orig/dest (matching the
    # census questions they came from). The SHIPPED schema uses src_/dst_
    # prefixes, per the dataset owner's naming convention. One rename, in one
    # place, after validation and before the codebook check -- so the codebook
    # documents exactly the names a reader of the file will see.
    panel = panel.rename(columns=FINAL_NAMES)
    log.info("      schema renamed to src_/dst_ convention (%d columns mapped)",
             len(FINAL_NAMES))
    # The codebook completeness check below runs on the FINAL names, so a
    # column that misses this mapping (or the codebook) still fails the build.

    # --- reconciliation ------------------------------------------------------
    total_weighted = float(panel["migrants"].sum())
    published = cfg["validation"].get("published_internal_migrants_2015_2020")
    recon_line = ""
    if published:
        disc = (total_weighted - float(published)) / float(published)
        recon_line = (f"panel total {total_weighted:,.0f} vs published "
                      f"{float(published):,.0f} -> {disc:+.3%}")
        log.info("  RECONCILIATION  %s", recon_line)
        cap = cfg["validation"].get("max_reconciliation_discrepancy")
        if cap is not None and abs(disc) > float(cap):
            raise PipelineError(
                f"reconciliation discrepancy {disc:+.3%} exceeds the configured "
                f"tolerance of {float(cap):.1%}.\n"
                "The panel total and the published tabulado disagree materially. "
                "Check bucket classification in reports/flows_reconciliation.md "
                "before trusting this panel."
            )
        log.info("  PASS  reconciliation within tolerance")
    else:
        recon_line = "NOT RUN -- validation.published_internal_migrants_2015_2020 is null"
        log.warning("  SKIP  reconciliation: no published total configured. "
                    "[TODO] set validation.published_internal_migrants_2015_2020 "
                    "from the INEGI migration tabulado.")

    # --- missingness BEFORE writing -----------------------------------------
    miss = missingness_table(panel, log)

    undocumented = [c for c in panel.columns if c not in COLUMN_SPEC]
    if undocumented:
        raise PipelineError(
            f"columns present in the panel but absent from COLUMN_SPEC: "
            f"{undocumented}. Every column must be documented in the codebook "
            "before it ships."
        )
    log.info("  PASS  every column has a codebook entry")

    # --- optionally filter the MAIN panel to positive flows ------------------
    acfg = cfg["assemble"]
    n_full = len(panel)
    main_filtered = bool(acfg.get("drop_zero_flows_from_main_panel", False))
    if main_filtered:
        log.warning("=" * 70)
        log.warning("DROPPING ZEROS FROM THE MAIN PANEL "
                    "(assemble.drop_zero_flows_from_main_panel = true)")
        log.warning("=" * 70)
        panel = panel[panel["migrants"] > 0].reset_index(drop=True)
        log_rows(log, "filter main panel to positive flows", n_full, len(panel))
        log.warning("  %s of %s dyads removed (%.2f%%).",
                    f"{n_full - len(panel):,}", f"{n_full:,}",
                    100.0 * (n_full - len(panel)) / n_full)
        log.warning("  This panel is now conditioned on the DEPENDENT VARIABLE "
                    "being positive.")
        log.warning("  It is NOT suitable for PPML gravity estimation: the "
                    "distance elasticity will be")
        log.warning("  biased toward zero by an amount that depends on how many "
                    "cells were dropped.")
        log.warning("  The choice is recorded in the codebook so a downstream "
                    "reader cannot mistake")
        log.warning("  this for a complete panel.")

    # --- write ---------------------------------------------------------------
    out_path = resolve(cfg, acfg["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False,
                     compression=acfg.get("compression", "zstd"))
    size_mb = out_path.stat().st_size / 1e6
    log.info("WROTE %s  rows=%s  cols=%d  size=%.1f MB",
             rel(out_path), f"{len(panel):,}", panel.shape[1], size_mb)

    # --- positive-only companion export --------------------------------------
    pos_path = None
    if acfg.get("export_positive_only", False):
        pos_path = write_positive_only(cfg, panel, log)

    stats = dict(n_municipios=n_muni, n_positive=n_positive,
                 n_full_cartesian=n_full, main_filtered=main_filtered,
                 positive_only_path=pos_path, climate_is_window=climate_is_window)
    write_codebook(cfg, panel, stats, log)

    # --- validation report ---------------------------------------------------
    lines = [
        "# Assembly validation",
        "",
        f"Rows: **{len(panel):,}**  |  Municipalities: **{n_muni:,}**  |  "
        f"Columns: **{panel.shape[1]}**",
        "",
        "## Checks",
        "",
        "| check | result |",
        "|---|---|",
        "| no duplicate (src, dst) keys | PASS |",
        f"| cartesian product complete ({n_muni:,}^2 - diagonal) | PASS |",
        "| diagonal excluded | PASS |",
        "| no negative flows | PASS |",
        f"| orphan CVEGEOs | {'PASS (none)' if not len(orphans) else f'{len(orphans)} FOUND'} |",
        "| every column documented in codebook | PASS |",
        "",
        "## Reconciliation against published tabulados",
        "",
        f"{recon_line}",
        "",
        "## Flow summary",
        "",
        "| statistic | value |",
        "|---|---:|",
        f"| total weighted migrants | {total_weighted:,.0f} |",
        f"| total sampled records | {int(panel['migrants_unweighted'].sum()):,} |",
        f"| pairs with positive flow | {n_positive:,} ({100.0 * n_positive / len(panel):.3f}%) |",
        f"| explicit zero pairs | {len(panel) - n_positive:,} |",
        "",
        "## Missingness by column",
        "",
        "| column | dtype | missing | % |",
        "|---|---|---:|---:|",
    ]
    for _, r in miss.iterrows():
        lines.append(f"| `{r['column']}` | {r['dtype']} | {r['n_missing']:,} | "
                     f"{r['pct_missing']:.3f}% |")
    p = report_path(cfg, "assembly_validation.md")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("WROTE %s", rel(p))

    log_rows(log, "STEP TOTAL 07_assemble", n_spine, len(panel))
    log.info("PIPELINE COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_step(main, STEP))



