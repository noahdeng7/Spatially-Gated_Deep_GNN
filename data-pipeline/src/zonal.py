"""
Population-weighted zonal statistics.

Shared by 04_gdp.py (gridded GDP cross-check) and 05_climate.py. Both need the
same operation: reduce a gridded INTENSIVE variable to a municipal value, with
people rather than area as the weight.

WHY NOT A SIMPLE POLYGON MEAN
-----------------------------
Temperature, precipitation and GDP per capita are intensive: they do not add
across space. An unweighted polygon mean answers "what is the average condition
over this territory", which lets 40,000 km2 of uninhabited Sonoran desert
outvote the town where everyone actually lives. The question a migration model
is asking is "what conditions do the people who might migrate experience", and
that requires weighting by population.

REGRIDDING DIRECTION (config: climate.regrid_direction)
-------------------------------------------------------
Default `pop_to_climate`: the FINER grid (WorldPop, 100 m) is aggregated UP onto
the COARSER climate grid (~1 km for WorldClim, ~4 km for TerraClimate).

The other direction is available but wrong by default. Resampling a 4 km climate
field down to 100 m does not create 100 m information -- it creates 1,600
identical copies of each 4 km value and lends them the *appearance* of fine
resolution. Every downstream standard error then treats invented precision as
real. Aggregating population upward instead throws away no information: the
population sum over a coarse cell is exactly the quantity the weight requires.

Population is aggregated with SUM (it is extensive -- it adds), which is why
resampling direction and resampling *method* have to be chosen together.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from common import PipelineError, ROOT, rel, resolve

__all__ = [
    "open_value_raster",
    "population_weighted_zonal_mean",
    "verify_crs",
]


def verify_crs(src, log: logging.Logger, what: str, require: str | None = None) -> str:
    """
    Explicitly verify and report a raster's CRS before any zonal operation.

    Called for EVERY raster this pipeline touches. An unlabelled CRS is refused
    rather than assumed: a wrong-but-plausible CRS produces zonal statistics
    that look entirely reasonable and describe the wrong locations.
    """
    if src.crs is None:
        raise PipelineError(
            f"{what}: raster has NO CRS. Refusing to guess.\n"
            "Assuming EPSG:4326 for an unlabelled raster is how zonal statistics "
            "end up silently sampling the wrong part of the world. Assign the CRS "
            "explicitly (gdal_edit.py -a_srs) once you have confirmed it."
        )
    crs = src.crs.to_string()
    log.info("      CRS CHECK  %-22s %-18s shape=%s res=(%.6g, %.6g) nodata=%s",
             what, crs, src.shape, src.res[0], src.res[1], src.nodata)
    if require and crs != require:
        log.warning("      %s is %s, expected %s -- will be reprojected/aligned",
                    what, crs, require)
    return crs


def open_value_raster(cfg: Mapping[str, Any], path: Path, log: logging.Logger,
                      netcdf_variable: str | None = None,
                      netcdf_year: int | None = None,
                      band: int = 1) -> Path:
    """
    Normalise any supported value source to a GeoTIFF path on disk.

    NetCDF inputs (Kummu gridded GDP, TerraClimate) are opened with rioxarray,
    the requested variable/year selected, and written to a cached GeoTIFF under
    interim/. Cached so the conversion happens once, not once per polygon.
    """
    path = Path(path)
    if path.suffix.lower() not in {".nc", ".nc4"}:
        return path

    import rioxarray  # noqa: F401
    import xarray as xr

    cache_dir = resolve(cfg, cfg["paths"]["interim"]) / "raster_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{path.stem}__{netcdf_variable or 'auto'}__{netcdf_year or 'all'}.tif"
    out = cache_dir / stem
    if out.exists():
        log.info("      using cached GeoTIFF for %s -> %s", path.name, out.name)
        return out

    log.info("      converting NetCDF %s -> GeoTIFF", path.name)
    ds = xr.open_dataset(path, decode_coords="all")

    if netcdf_variable and netcdf_variable in ds:
        da = ds[netcdf_variable]
    else:
        data_vars = [v for v in ds.data_vars
                     if ds[v].ndim >= 2 and v not in ("crs", "spatial_ref")]
        if not data_vars:
            raise PipelineError(f"{path.name}: no gridded data variables found")
        if len(data_vars) > 1 and not netcdf_variable:
            log.warning("      %s has multiple variables %s; using %r. Set the "
                        "variable explicitly in config if that is not the one.",
                        path.name, data_vars, data_vars[0])
        da = ds[data_vars[0]]

    # Collapse any time-like dimension to the requested year, or average it.
    for dim in list(da.dims):
        if dim in ("x", "y", "lon", "lat", "longitude", "latitude"):
            continue
        coord = da.coords.get(dim)
        if netcdf_year is not None and coord is not None:
            try:
                vals = pd.to_numeric(pd.Series(np.asarray(coord)), errors="coerce")
                if vals.notna().any() and (vals == netcdf_year).any():
                    da = da.sel({dim: netcdf_year})
                    log.info("      selected %s=%s", dim, netcdf_year)
                    continue
            except Exception:  # noqa: BLE001
                pass
        if dim in da.dims and da.sizes.get(dim, 1) > 1:
            log.warning("      averaging over dimension %r (size %d)", dim, da.sizes[dim])
            da = da.mean(dim=dim, skipna=True)
        elif dim in da.dims:
            da = da.squeeze(dim, drop=True)

    da = da.rename({"lon": "x", "lat": "y"}) if "lon" in da.dims else da
    if da.rio.crs is None:
        log.warning("      %s has no CRS; NetCDF climate/GDP grids of this type "
                    "are conventionally EPSG:4326 -- assigning it. VERIFY this "
                    "against the product documentation.", path.name)
        da = da.rio.write_crs("EPSG:4326")

    da.rio.to_raster(out, compress="deflate")
    log.info("      wrote %s", rel(out))
    return out


def population_weighted_zonal_mean(
    cfg: Mapping[str, Any],
    gdf,
    value_raster: Path,
    log: logging.Logger,
    label: str,
    band: int = 1,
    scale: float = 1.0,
    offset: float = 0.0,
    netcdf_variable: str | None = None,
    netcdf_year: int | None = None,
) -> pd.DataFrame:
    """
    Population-weighted mean of `value_raster` within each polygon of `gdf`.

    Returns a DataFrame keyed on cvegeo with:
        value          population-weighted mean
        value_min      minimum of contributing cells
        value_max      maximum of contributing cells
        weight_sum     total population inside the polygon
        n_cells        number of contributing cells
        method         population_weighted | unweighted_fallback

    Asserts (per config.climate.assert_weighted_mean_in_range) that the weighted
    mean lies within [min, max] of the cells that produced it. A weighted mean
    outside that interval is arithmetically impossible for non-negative weights,
    so if the assertion trips it means nodata has leaked into the weights or the
    two grids are misaligned -- both of which otherwise produce plausible-looking
    output.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.mask import mask as rio_mask
    from rasterio.vrt import WarpedVRT

    value_raster = open_value_raster(cfg, Path(value_raster), log,
                                     netcdf_variable, netcdf_year, band)
    pop_path = resolve(cfg, cfg["distance"]["worldpop_raster"])
    if not pop_path.exists():
        raise PipelineError(
            f"WorldPop raster not found: {pop_path}\n"
            "Population weighting cannot proceed. Falling back to an unweighted "
            "polygon mean would silently change what this variable measures, so "
            "this stops instead."
        )

    direction = cfg.get("climate", {}).get("regrid_direction", "pop_to_climate")
    assert_range = bool(cfg.get("climate", {}).get("assert_weighted_mean_in_range", True))

    rows = []
    with rasterio.open(value_raster) as vsrc, rasterio.open(pop_path) as psrc:
        v_crs = verify_crs(vsrc, log, f"value: {label}")
        p_crs = verify_crs(psrc, log, "weights: WorldPop 2020")

        v_area = abs(vsrc.res[0] * vsrc.res[1])
        p_area = abs(psrc.res[0] * psrc.res[1])
        log.info("      cell areas (native units^2): value=%.6g  pop=%.6g", v_area, p_area)

        if direction == "pop_to_climate":
            log.info("      REGRID  WorldPop -> value grid, Resampling.sum "
                     "(population is extensive; sum conserves people)")
            target_crs, target_transform = vsrc.crs, vsrc.transform
            target_w, target_h = vsrc.width, vsrc.height
            pop_resampling = Resampling.sum
        else:
            log.warning("      REGRID  value -> WorldPop grid (climate_to_pop). "
                        "This resamples a coarse field onto a fine grid and "
                        "manufactures apparent precision. Prefer pop_to_climate.")
            target_crs, target_transform = psrc.crs, psrc.transform
            target_w, target_h = psrc.width, psrc.height
            pop_resampling = Resampling.nearest

        with WarpedVRT(psrc, crs=target_crs, transform=target_transform,
                       width=target_w, height=target_h,
                       resampling=pop_resampling) as pvrt, \
             WarpedVRT(vsrc, crs=target_crs, transform=target_transform,
                       width=target_w, height=target_h,
                       resampling=Resampling.bilinear) as vvrt:

            work = gdf.to_crs(target_crs) if gdf.crs.to_string() != str(target_crs) else gdf
            n = len(work)
            n_fallback = 0

            for i, rec in enumerate(work.itertuples(index=False)):
                geom, cvegeo = rec.geometry, rec.cvegeo
                try:
                    # filled=False: filling with NaN directly would fail on
                    # integer-typed rasters (e.g. the Kummu uint32 GDP grid).
                    # Take the masked array and fill after casting to float.
                    varr, _ = rio_mask(vvrt, [geom], crop=True, filled=False,
                                       all_touched=True)
                    parr, _ = rio_mask(pvrt, [geom], crop=True, filled=False,
                                       all_touched=True)
                    varr = np.ma.filled(varr.astype("float64"), np.nan)
                    parr = np.ma.filled(parr.astype("float64"), np.nan)
                except ValueError:
                    rows.append(dict(cvegeo=cvegeo, value=np.nan, value_min=np.nan,
                                     value_max=np.nan, weight_sum=0.0, n_cells=0,
                                     method="no_overlap"))
                    continue

                vals = varr[band - 1].astype("float64") if varr.shape[0] >= band else varr[0].astype("float64")
                wts = parr[0].astype("float64")

                if vsrc.nodata is not None and np.isfinite(vsrc.nodata):
                    vals = np.where(vals == vsrc.nodata, np.nan, vals)
                if psrc.nodata is not None and np.isfinite(psrc.nodata):
                    wts = np.where(wts == psrc.nodata, np.nan, wts)
                wts = np.where(wts < 0, np.nan, wts)   # WorldPop nodata is -9999

                vals = vals * scale + offset

                ok = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
                n_cells = int(ok.sum())

                if n_cells and float(np.nansum(wts[ok])) > 0:
                    v_ok, w_ok = vals[ok], wts[ok]
                    mean = float(np.average(v_ok, weights=w_ok))
                    vmin, vmax = float(v_ok.min()), float(v_ok.max())
                    wsum = float(w_ok.sum())
                    method = "population_weighted"

                    if assert_range and not (vmin - 1e-9 <= mean <= vmax + 1e-9):
                        raise PipelineError(
                            f"{label} / {cvegeo}: population-weighted mean {mean:.6g} "
                            f"falls OUTSIDE the range of its own cells "
                            f"[{vmin:.6g}, {vmax:.6g}].\n"
                            "This is arithmetically impossible with non-negative "
                            "weights. Cause is almost certainly nodata leaking into "
                            "the weight array, or the value and weight grids being "
                            "misaligned. Do not disable this check to get past it."
                        )
                else:
                    # No population in the polygon at all: fall back to an
                    # unweighted mean of the value cells, and SAY SO in `method`
                    # so it can be excluded from analysis if that matters.
                    finite = np.isfinite(vals)
                    if finite.any():
                        mean = float(vals[finite].mean())
                        vmin, vmax = float(vals[finite].min()), float(vals[finite].max())
                        n_cells = int(finite.sum())
                        method = "unweighted_fallback"
                        n_fallback += 1
                    else:
                        mean = vmin = vmax = np.nan
                        n_cells = 0
                        method = "no_valid_cells"
                    wsum = 0.0

                rows.append(dict(cvegeo=cvegeo, value=mean, value_min=vmin,
                                 value_max=vmax, weight_sum=wsum,
                                 n_cells=n_cells, method=method))

                if (i + 1) % 250 == 0 or i == n - 1:
                    log.info("  %s zonal %d/%d", label, i + 1, n)

    out = pd.DataFrame(rows)
    out["cvegeo"] = out["cvegeo"].astype("string")

    n_pw = int((out["method"] == "population_weighted").sum())
    log.info("      %s: %d/%d population-weighted, %d unweighted fallback, "
             "%d with no valid cells",
             label, n_pw, len(out),
             int((out["method"] == "unweighted_fallback").sum()),
             int(out["method"].isin(["no_valid_cells", "no_overlap"]).sum()))
    if n_pw < len(out):
        log.warning("      %d municipality/ies did not get a population-weighted "
                    "value for %s -- see the `method` column, they are flagged "
                    "not dropped", len(out) - n_pw, label)
    return out


