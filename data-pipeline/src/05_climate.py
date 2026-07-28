"""
05_climate.py -- population-weighted temperature and precipitation by municipality.

Input  : WorldClim v2.1 / CHELSA v2.1 / TerraClimate (config: climate.source)
         WorldPop 2020 grid (weights)
         municipal geometry
Output : interim/climate.parquet
         reports/climate_diagnostics.md

VARIABLES
---------
    temp_mean_c     population-weighted mean ANNUAL MEAN temperature, degC
    precip_total_mm population-weighted mean ANNUAL TOTAL precipitation, mm

AGGREGATION ACROSS MONTHS IS NOT THE SAME OPERATION FOR BOTH
------------------------------------------------------------
Temperature is averaged over the 12 monthly normals; precipitation is SUMMED.
Annual mean temperature and annual *total* precipitation are the conventional
quantities, and summing temperature or averaging precipitation would produce a
number roughly 12x off in one case and 12x off in the other -- both large enough
to notice, neither large enough to be obviously wrong in a regression table.

Since the population-weighted mean is a linear operator, weighting each month
and then combining across months gives the identical answer to combining first
and weighting once. Doing it per-month is preferred here because it makes the
per-month min/max range check meaningful.

WEIGHTING AND REGRIDDING
------------------------
Delegated to src/zonal.py. Weights are the WorldPop 2020 grid -- the same grid
used for the population-weighted centroids in 03, so distance and climate refer
to a consistent notion of "where the people are". WorldPop is aggregated UP to
the climate grid rather than climate being resampled down; see the note in
zonal.py for why the other direction manufactures precision.

Every raster's CRS is verified explicitly before any zonal operation, and the
weighted mean is asserted to lie within the min/max of its contributing cells.

Idempotent. Monthly zonal results are cached under interim/raster_cache/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PipelineError, ROOT, assert_cvegeo_valid, ensure_dirs, get_logger,
    load_config, log_rows, missingness_table, report_path, rel, resolve, run_step,
    write_interim,
)
from geo import apply_crosswalk, assert_geometry_vintage, load_crosswalk, load_municipios  # noqa: E402
from zonal import population_weighted_zonal_mean  # noqa: E402

STEP = "05_climate"

# Physically plausible bounds for Mexico. Violations are reported, not clipped.
TEMP_BOUNDS_C = (-5.0, 35.0)
PRECIP_BOUNDS_MM = (0.0, 6000.0)


def _monthly_files(directory: Path, log, what: str) -> list[Path]:
    """Locate the 12 monthly rasters, ordered Jan..Dec."""
    if not directory.exists():
        raise PipelineError(
            f"{what} directory not found: {directory}\n"
            "See config.yaml -> downloads for what to fetch."
        )
    files = sorted(
        [p for p in directory.iterdir()
         if p.suffix.lower() in {".tif", ".tiff", ".nc"}]
    )
    if len(files) != 12:
        raise PipelineError(
            f"{what}: expected 12 monthly rasters in {directory}, found {len(files)}.\n"
            f"  {[p.name for p in files[:15]]}\n"
            "Annual aggregates built from a partial year would be silently wrong "
            "-- e.g. 9 months of precipitation still looks like a plausible "
            "rainfall total."
        )
    log.info("      %s: 12 monthly rasters, %s .. %s", what, files[0].name, files[-1].name)
    return files


def _source_spec(cfg, log) -> dict:
    """Resolve the configured climate product into directories and scaling."""
    source = cfg["climate"]["source"]
    ccfg = cfg["climate"].get(source)
    if ccfg is None:
        raise PipelineError(
            f"climate.source = {source!r} but there is no climate.{source} block "
            "in config.yaml"
        )

    if source == "worldclim":
        log.info("CHOICE climate source = WorldClim v%s %s, period %s",
                 ccfg.get("version"), ccfg.get("resolution"), ccfg.get("period"))
        log.info("       (1970-2000 normals -- a long-run climate signal, NOT "
                 "conditions during the 2015-2020 migration window)")
        return dict(
            source=source,
            tmean_files=_monthly_files(resolve(cfg, ccfg["tmean_dir"]), log, "WorldClim tavg"),
            prec_files=_monthly_files(resolve(cfg, ccfg["prec_dir"]), log, "WorldClim prec"),
            tmean_scale=1.0, tmean_offset=0.0, prec_scale=1.0,
            period=ccfg.get("period"),
        )

    if source == "chelsa":
        log.info("CHOICE climate source = CHELSA v%s, period %s",
                 ccfg.get("version"), ccfg.get("period"))
        log.info("       (finer orographic downscaling than WorldClim -- better in "
                 "the sierras, less standard in the literature)")
        return dict(
            source=source,
            tmean_files=_monthly_files(resolve(cfg, ccfg["tmean_dir"]), log, "CHELSA tas"),
            prec_files=_monthly_files(resolve(cfg, ccfg["prec_dir"]), log, "CHELSA pr"),
            # CHELSA v2.1 ships scaled integers: tas is K*10, pr is mm*100.
            tmean_scale=float(ccfg.get("tmean_scale", 0.1)),
            tmean_offset=float(ccfg.get("tmean_offset", -273.15)),
            prec_scale=float(ccfg.get("prec_scale", 0.01)),
            period=ccfg.get("period"),
        )

    if source == "terraclimate":
        log.info("CHOICE climate source = TerraClimate, period %s", ccfg.get("period"))
        log.info("       (4km, but covers %s -- adjacent to the migration window "
                 "rather than a 1970-2000 normal)", ccfg.get("period"))
        raise NotImplementedError(
            "TerraClimate ingestion is scaffolded but not implemented.\n\n"
            "It differs structurally from WorldClim/CHELSA: one NetCDF per "
            "variable per year with a 12-step time axis, and mean temperature "
            "must be derived as (tmax+tmin)/2 because TerraClimate publishes no "
            "tmean band. Implement in _source_spec() by:\n"
            "  1. opening tmax and tmin for each year in climate.terraclimate.period\n"
            "  2. averaging them to a monthly tmean\n"
            "  3. averaging monthly tmean across years to a normal\n"
            "  4. summing ppt within year, then averaging totals across years\n"
            "Step 4 order matters: mean-of-annual-totals != sum-of-monthly-means "
            "x 12 once years have differing data coverage.\n\n"
            "Use climate.source = worldclim or chelsa in the meantime."
        )

    raise PipelineError(f"unknown climate.source: {source!r}")


def _zonal_months(cfg, gdf, files, log, label, scale, offset) -> pd.DataFrame:
    """Run the population-weighted zonal mean for each of the 12 months."""
    monthly = []
    for m, path in enumerate(files, start=1):
        log.info("  %s month %02d/12  %s", label, m, path.name)
        res = population_weighted_zonal_mean(
            cfg=cfg, gdf=gdf, value_raster=path, log=log,
            label=f"{label}-{m:02d}", scale=scale, offset=offset,
        )
        res["month"] = m
        monthly.append(res)
    return pd.concat(monthly, ignore_index=True)


def main() -> int:
    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(STEP, cfg)
    log.info("=" * 78)
    log.info("STEP %s", STEP)
    log.info("=" * 78)

    spec = _source_spec(cfg, log)

    gdf = load_municipios(cfg, log)
    assert_geometry_vintage(cfg, gdf, log)
    n_in = len(gdf)
    xwalk = load_crosswalk(cfg, log)
    gdf["cvegeo"] = apply_crosswalk(gdf["cvegeo"], xwalk,
                                    cfg["geometry"]["crosswalk_strategy"], log, "geometry")
    if bool(gdf["cvegeo"].duplicated().any()):
        gdf = gdf.dissolve(by="cvegeo").reset_index()
        gdf["cvegeo"] = gdf["cvegeo"].astype("string")
        log_rows(log, "dissolve split children into parent", n_in, len(gdf))

    # --- temperature: AVERAGE the 12 monthly means --------------------------
    tmon = _zonal_months(cfg, gdf, spec["tmean_files"], log, "tavg",
                         spec["tmean_scale"], spec["tmean_offset"])
    temp = (
        tmon.groupby("cvegeo", as_index=False, observed=True)
        .agg(temp_mean_c=("value", "mean"),
             temp_month_min_c=("value", "min"),
             temp_month_max_c=("value", "max"),
             temp_n_months=("value", "count"),
             temp_method=("method", lambda s: s.mode().iat[0] if len(s.mode()) else "unknown"))
    )
    log.info("      temperature: annual mean = MEAN of 12 monthly means")

    # --- precipitation: SUM the 12 monthly totals ---------------------------
    pmon = _zonal_months(cfg, gdf, spec["prec_files"], log, "prec",
                         spec["prec_scale"], 0.0)
    precip = (
        pmon.groupby("cvegeo", as_index=False, observed=True)
        .agg(precip_total_mm=("value", "sum"),
             precip_month_min_mm=("value", "min"),
             precip_month_max_mm=("value", "max"),
             precip_n_months=("value", "count"),
             precip_method=("method", lambda s: s.mode().iat[0] if len(s.mode()) else "unknown"))
    )
    log.info("      precipitation: annual total = SUM of 12 monthly totals")

    clim = temp.merge(precip, on="cvegeo", how="outer")
    clim["climate_source"] = spec["source"]
    clim["climate_period"] = str(spec.get("period"))

    # --- completeness --------------------------------------------------------
    for col, want in (("temp_n_months", 12), ("precip_n_months", 12)):
        short = clim[col] < want
        if bool(short.any()):
            log.warning("      %d municipality/ies have fewer than %d months of "
                        "%s -- annual aggregate is incomplete and FLAGGED, not "
                        "silently rescaled", int(short.sum()), want, col)
    clim["climate_complete"] = (clim["temp_n_months"] == 12) & (clim["precip_n_months"] == 12)

    # --- plausibility (report, never clip) -----------------------------------
    diagnostics = []
    for col, (lo, hi), unit in (("temp_mean_c", TEMP_BOUNDS_C, "degC"),
                                ("precip_total_mm", PRECIP_BOUNDS_MM, "mm")):
        s = clim[col]
        out_of_range = s.notna() & ((s < lo) | (s > hi))
        log.info("      %-18s n=%s  min=%.2f  mean=%.2f  max=%.2f %s",
                 col, f"{int(s.notna().sum()):,}",
                 float(s.min()) if s.notna().any() else float("nan"),
                 float(s.mean()) if s.notna().any() else float("nan"),
                 float(s.max()) if s.notna().any() else float("nan"), unit)
        if bool(out_of_range.any()):
            log.warning("      %d value(s) of %s outside the plausible range "
                        "[%s, %s] %s: %s", int(out_of_range.sum()), col, lo, hi, unit,
                        clim.loc[out_of_range, "cvegeo"].head(10).tolist())
            log.warning("      REPORTED, NOT CLIPPED. Clipping would hide a unit "
                        "or scaling error (e.g. CHELSA's K*10 encoding).")
        diagnostics.append(dict(column=col, n=int(s.notna().sum()),
                                min=float(s.min()) if s.notna().any() else np.nan,
                                mean=float(s.mean()) if s.notna().any() else np.nan,
                                max=float(s.max()) if s.notna().any() else np.nan,
                                unit=unit, out_of_range=int(out_of_range.sum()),
                                bound_lo=lo, bound_hi=hi))

    assert_cvegeo_valid(clim["cvegeo"], "climate.cvegeo")
    if bool(clim["cvegeo"].duplicated().any()):
        raise PipelineError("duplicate cvegeo in climate table")

    missingness_table(clim, log)

    # --- report --------------------------------------------------------------
    lines = [
        "# Climate diagnostics",
        "",
        f"Source: **{spec['source']}**, period **{spec.get('period')}**",
        f"Weighting: WorldPop 2020, regrid direction "
        f"`{cfg['climate']['regrid_direction']}`",
        "",
        "## Variable construction",
        "",
        "| variable | monthly -> annual | why |",
        "|---|---|---|",
        "| `temp_mean_c` | **mean** of 12 monthly means | annual mean temperature |",
        "| `precip_total_mm` | **sum** of 12 monthly totals | annual total precipitation |",
        "",
        "These are different operations. Averaging precipitation or summing",
        "temperature yields a number ~12x off -- wrong, but not obviously wrong",
        "in a regression table.",
        "",
        "## Distributions",
        "",
        "| column | n | min | mean | max | unit | outside plausible range |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for d in diagnostics:
        lines.append(
            f"| `{d['column']}` | {d['n']:,} | {d['min']:.2f} | {d['mean']:.2f} | "
            f"{d['max']:.2f} | {d['unit']} | {d['out_of_range']} "
            f"(bounds {d['bound_lo']}..{d['bound_hi']}) |"
        )
    lines += [
        "",
        "Out-of-range values are **reported, never clipped** -- clipping would",
        "mask a unit-scaling error rather than reveal it.",
        "",
        "## Vintage caveat",
        "",
    ]
    if spec["source"] in ("worldclim", "chelsa"):
        lines += [
            f"These are long-run climate **normals** ({spec.get('period')}), not",
            "conditions during the 2015-2020 migration window. They identify the",
            "effect of *climate* (a persistent locational attribute people sort on)",
            "rather than *weather shocks* (a transitory push factor). If the",
            "research question is about drought or shock-driven displacement,",
            "switch `climate.source` to `terraclimate` and use a window-adjacent",
            "period -- the two answer different questions.",
        ]
    else:
        lines += [
            "TerraClimate covers a period adjacent to the migration window, so",
            "these values reflect conditions closer to the migration decision than",
            "a 1970-2000 normal would.",
        ]
    p = report_path(cfg, "climate_diagnostics.md")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("WROTE %s", rel(p))

    write_interim(cfg, clim, "climate", log)
    log_rows(log, "STEP TOTAL 05_climate", n_in, len(clim))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_step(main, STEP))



