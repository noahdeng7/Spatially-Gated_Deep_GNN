"""Per-municipality TEMPERATURE and PRECIPITATION for MEXICO via Google Earth Engine.

Mexico port of the Brazil `climate_loader.py`, trimmed to the two variables the
existing pipeline carries -- annual temperature and annual precipitation. The
Earth Engine pull and the per-metric calculation are IDENTICAL to the Brazil
loader; the extra Brazil variables (wind, wet-bulb, UV, degree-days, and the
z-from-baseline anomalies) were removed, and the window is a 2016-2020 average.

WHY GEE (and not the Copernicus CDS or the ARCO-ERA5 GCS Zarr)
-------------------------------------------------------------
The goal was to avoid the Copernicus CDS limits (queue + the per-request "cost
limit" that rejects any daily-statistics request bigger than one year). The
obvious Google option -- the ARCO-ERA5 public bucket -- does NOT help: its hourly
Zarr is chunked one *global* field per timestep, so slicing a multi-year Mexico
box still streams enormous volumes. Earth Engine is the right tool because it runs
the whole computation **server-side** and returns only the ~2,457 per-municipality
numbers -- no CDS queue, no cost cap, and essentially zero local disk.

DATASET
-------
`ECMWF/ERA5_LAND/DAILY_AGGR` (0.1 deg ~11 km daily, 1950->present).

AUTH
----
Needs Earth Engine. `_init_ee()` calls `ee.Initialize()`, using the project in
$EARTHENGINE_PROJECT / $GOOGLE_CLOUD_PROJECT if set, and falls back to
`ee.Authenticate()` once. In a container the mounted ~/.config/earthengine +
~/.config/gcloud provide the credentials.

WINDOW
------
temp and precip are the 2016-2020 average of the annual metric (CLIMATE_YEARS):
each year is reduced to one number per municipality and cached to
interim/climate_gee/, then the years are averaged. Change the single CLIMATE_YEARS
constant to shift or resize the window.

VARIABLES
---------
  temp    2016-2020 mean annual temperature (annual-mean of daily-MEAN temp).
  precip  2016-2020 mean annual precipitation (annual-sum of daily totals).

POPULATION WEIGHTING
--------------------
Both are **population-weighted** municipal means, not unweighted polygon means:

    value_muni = sum_cells(value * pop) / sum_cells(pop)

Temperature and precipitation are INTENSIVE -- they do not add across space, so
an unweighted polygon mean answers "what is the average condition over this
territory", which lets 40,000 km2 of uninhabited Sonoran desert outvote the town
where everyone actually lives. A migration model is asking what conditions the
people who might migrate experience, and that requires weighting by population.
This is the same choice, and the same WorldPop 2020 grid, as the population-
weighted centroids in 03_distance.py and the zonal means in src/zonal.py.

Weights come from `WorldPop/GP/100m/pop` (year WORLDPOP_YEAR) server-side, so
no raster leaves Earth Engine. The reduction runs at ERA5-Land's native ~11 km
scale, i.e. population is aggregated UP to the climate grid rather than climate
being resampled down onto 100 m -- see the regrid note in src/zonal.py for why
the other direction manufactures precision it does not have.

The unweighted mean is computed in the SAME pass and kept as `temp_unweighted` /
`precip_unweighted`, both because it is the fallback for municipalities that
carry no population under the ERA5 mask and because keeping it makes the size of
the weighting correction auditable rather than asserted.

UNITS CAVEAT (differs from the WorldClim path in 05_climate.py)
--------------------------------------------------------------
This is a verbatim port, so it keeps the Brazil loader's native units:
temperature in DEGREES FAHRENHEIT and precipitation in METRES (annual sum). The
existing `05_climate.py` WorldClim path emits degC and mm. To line these up,
convert temp with (F-32)/1.8 and precip with *1000 before merging -- this module
leaves them in F / m so the numbers match the calculation they came from.

CAVEATS
-------
* ERA5-Land (0.1 deg) != ERA5 (0.25 deg); values differ slightly from a CDS path.
* ERA5-Land returns null for polygons that miss the land mask (small coastal /
  island municipalities); those are filled by inverse-distance k-NN from the
  nearest municipal centroids.
* A municipality with no WorldPop weight under the ERA5 land mask falls back to
  the unweighted mean, flagged in `climate_weighting` -- never silently.

INTEGRATION NOTE
----------------
`build_all()` writes interim/climate_gee.parquet keyed on cvegeo. Wiring these into
the dyadic panel (src_/dst_ variants in 07_assemble.py + codebook entries) is a
deliberate, separate change -- this module only produces the per-municipality table.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CVEGEO_DTYPE, PipelineError, assert_cvegeo_valid, ensure_dirs, get_logger,
    load_config, resolve, run_step, write_interim,
)
from geo import apply_crosswalk, load_crosswalk, load_municipios  # noqa: E402

ERA5_LAND_DAILY = "ECMWF/ERA5_LAND/DAILY_AGGR"

# Population weights. Same product and vintage as the WorldPop raster behind the
# population-weighted centroids in 03_distance.py (config: distance.worldpop_raster),
# so distance and climate refer to one consistent notion of "where the people are".
WORLDPOP_IC = "WorldPop/GP/100m/pop"
WORLDPOP_YEAR = 2020

K = 273.15            # degC -> K offset
TARGET_YEAR = 2020    # default year for the single-year reducer (census year)

# The shared climate window: a 2016-2020 average, keyed to the tail of the
# 2015-2020 migration window. Change this single constant to shift or resize it.
CLIMATE_YEARS = list(range(2016, 2021))          # 2016..2020 inclusive

STEP = "climate_gee"


# ---------------------------------------------------------------------------
# Local reimplementations of the Brazil utils / ndvi_loader / dyad helpers.
# ---------------------------------------------------------------------------
_CFG = None
_EE_READY = False
_MUNI_CACHE = None


def _config() -> dict:
    """Load and cache config.yaml (the pipeline's single source of paths/choices)."""
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


def _intermediate() -> Path:
    """interim/climate_gee/ -- all per-year checkpoints live here."""
    cfg = _config()
    d = resolve(cfg, cfg["paths"]["interim"]) / "climate_gee"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _norm_cvegeo(s) -> pd.Series:
    """Zero-pad to the canonical 5-char CVEGEO string (never an int)."""
    t = pd.Series(s).astype(CVEGEO_DTYPE).str.strip()
    t = t.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    return t.str.zfill(5).astype(CVEGEO_DTYPE)


def _init_ee() -> None:
    """Initialize Earth Engine once (project from env if provided)."""
    global _EE_READY
    if _EE_READY:
        return
    import ee
    project = os.environ.get("EARTHENGINE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception:  # noqa: BLE001 -- first run in a fresh environment
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()
    _EE_READY = True


def _harmonized_munis(n_sample: int | None = None,
                      simplify_tol: float = 0.005):
    """The 2020 municipal universe (2,457), harmonized and simplified for GEE.

    Same recipe as 05_climate.py: load the Marco Geoestadistico layer, fold split
    children into their pre-split parent via the 2015->2020 crosswalk, dissolve,
    then simplify so each reduceRegions batch stays under GEE's payload cap.
    Cached, because reading + harmonizing the gpkg on every call is waste.
    """
    global _MUNI_CACHE
    if _MUNI_CACHE is None:
        cfg = _config()
        log = get_logger(STEP, cfg)
        gdf = load_municipios(cfg, log)
        xwalk = load_crosswalk(cfg, log)
        gdf["cvegeo"] = apply_crosswalk(
            gdf["cvegeo"], xwalk, cfg["geometry"]["crosswalk_strategy"], log, "geometry")
        if bool(gdf["cvegeo"].duplicated().any()):
            gdf = gdf.dissolve(by="cvegeo").reset_index()
            gdf["cvegeo"] = gdf["cvegeo"].astype(CVEGEO_DTYPE)
        gdf = gdf[["cvegeo", "geometry"]].copy()
        gdf["cvegeo"] = _norm_cvegeo(gdf["cvegeo"])
        gdf["geometry"] = gdf.geometry.simplify(tolerance=simplify_tol, preserve_topology=True)
        _MUNI_CACHE = gdf
    gdf = _MUNI_CACHE
    if n_sample:
        gdf = gdf.sample(n=min(n_sample, len(gdf)), random_state=0).reset_index(drop=True)
    return gdf


def _load_muni_gdf(muni_gdf=None, n_sample: int | None = None):
    """Return a caller-supplied gdf, else the cached harmonized universe."""
    if muni_gdf is not None:
        return muni_gdf if n_sample is None else \
            muni_gdf.sample(n=min(n_sample, len(muni_gdf)), random_state=0).reset_index(drop=True)
    return _harmonized_munis(n_sample)


def _reduce_over_munis(ee, image, gdf, reducer, scale, tile_scale, batch, tag):
    """reduceRegions(`reducer`) of `image` over each muni polygon -> list of prop dicts."""
    rows: list[dict] = []
    n = len(gdf)
    for start in range(0, n, batch):
        sub = gdf.iloc[start:start + batch]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry(r.geometry.__geo_interface__),
                       {"cvegeo": r["cvegeo"]})
            for _, r in sub.iterrows()
        ])
        stats = image.reduceRegions(collection=fc, reducer=reducer,
                                    scale=scale, tileScale=tile_scale)
        for f in stats.getInfo()["features"]:
            rows.append(f["properties"])
        print(f"  [gee/{tag}] munis {min(start + batch, n)}/{n}", flush=True)
    return rows


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) from one point (radians) to arrays of points (radians)."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _idw_knn_fill(values, lat, lon, k: int = 8) -> pd.Series:
    """Inverse-distance-weighted k-NN fill of NaNs from the nearest munis (haversine).

    ERA5-Land returns null for polygons that miss the land mask (small coastal /
    island municipalities), so fill those from k=8 nearest neighbours weighted by
    inverse squared distance.
    """
    vals = pd.Series(values).astype(float).to_numpy().copy()
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    miss = np.isnan(vals)
    if not miss.any():
        return pd.Series(vals)
    known = ~miss & ~np.isnan(lat) & ~np.isnan(lon)
    if not known.any():
        return pd.Series(vals)
    klat = np.radians(lat[known]); klon = np.radians(lon[known]); kval = vals[known]
    for i in np.where(miss)[0]:
        if np.isnan(lat[i]) or np.isnan(lon[i]):
            continue
        d = _haversine_km(np.radians(lat[i]), np.radians(lon[i]), klat, klon)
        idx = np.argsort(d)[:k]
        w = 1.0 / np.maximum(d[idx], 1e-6) ** 2
        vals[i] = float(np.sum(w * kval[idx]) / np.sum(w))
    return pd.Series(vals)


def _load_centroids() -> pd.DataFrame:
    """Population-weighted municipal centroids (cvegeo, lat, lon) from 03_distance."""
    cfg = _config()
    path = resolve(cfg, cfg["paths"]["processed"]) / "municipios_centroids.csv"
    if not path.exists():
        raise PipelineError(
            f"centroids not found: {path}. Run 03_distance.py first -- the IDW-KNN "
            "fill needs municipal lat/lon to fill land-mask gaps.")
    df = pd.read_csv(path, dtype={"cvegeo": str})
    df["cvegeo"] = _norm_cvegeo(df["cvegeo"])
    return df[["cvegeo", "lat", "lon"]]


def average_over_years(tag: str, fn, years, **kwargs) -> pd.DataFrame:
    """Per-muni mean of `fn`'s annual output over `years`, checkpointing each year.

    Every per-year result is cached to interim/climate_gee/{tag}_{year}.csv so a
    crash mid-build resumes and a re-run is free. The numeric columns are then
    averaged across years, keyed on cvegeo. Partial-year nulls are skipped by the
    mean; a municipality null in every year survives as NaN for the fill step.
    """
    frames = []
    for y in years:
        ckpt = _intermediate() / f"{tag}_{y}.csv"
        if ckpt.exists():
            print(f"  [cache] {ckpt.name}", flush=True)
            df = pd.read_csv(ckpt, dtype={"cvegeo": str})
        else:
            df = fn(year=y, **kwargs)
            df.to_csv(ckpt, index=False)
        df["cvegeo"] = _norm_cvegeo(df["cvegeo"])
        frames.append(df)
    allrows = pd.concat(frames, ignore_index=True)
    value_cols = [c for c in allrows.columns if c != "cvegeo"]
    return allrows.groupby("cvegeo", as_index=False)[value_cols].mean()


# ---------------------------------------------------------------------------
# Annual temp/precip reducer (one number per muni per year).
#
# Temperature is the annual mean of daily-MEAN temperature; precipitation is the
# annual SUM of daily totals -- annual mean temperature and annual *total*
# precipitation are the conventional quantities, and the two aggregations are not
# interchangeable (summing temperature or averaging precipitation is ~365x off).
#
# Both are then reduced to the municipality as a POPULATION-WEIGHTED mean. The
# weighted mean is assembled from two sums rather than an EE weighted reducer so
# that the numerator and denominator are visibly the same cells, and so that the
# denominator (`pop_weight`) comes back for inspection instead of being consumed
# inside a black box.
# ---------------------------------------------------------------------------
def _weight_image(ee):
    """WorldPop `WORLDPOP_YEAR` population count, mosaicked, unmasked to 0."""
    return (ee.ImageCollection(WORLDPOP_IC)
            .filter(ee.Filter.eq("year", WORLDPOP_YEAR))
            .mosaic()
            .unmask(0)
            .rename("pop"))


def annual_temp_precip_per_muni(year: int = TARGET_YEAR,
                                muni_gdf=None,
                                n_sample: int | None = None,
                                batch: int = 150,
                                scale: int = 11132,
                                tile_scale: int = 16) -> pd.DataFrame:
    """Per-muni population-weighted [temp (degF), precip (m)] for `year`.

    Returns temp/precip (population-weighted), temp_unweighted/precip_unweighted
    (the plain polygon mean, same pass), and pop_weight (the weight denominator).

    `pop_weight` is NOT a headcount. Reducing at ERA5-Land's ~11 km scale makes EE
    pyramid the 100 m WorldPop grid by mean, so each coarse cell carries mean
    persons-per-100m-cell. That is proportional to the cell's population (coarse
    cells are equal-area), which is all a weighted mean needs -- the constant
    factor cancels in the ratio -- but the absolute number is not people.
    """
    import ee
    _init_ee()

    gdf = _load_muni_gdf(muni_gdf, n_sample)
    land = (ee.ImageCollection(ERA5_LAND_DAILY)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))
    temp_f = land.select("temperature_2m").map(                    # daily mean -> degF
        lambda i: i.subtract(K).multiply(1.8).add(32))
    precip = land.select("total_precipitation_sum")                # daily total m
    base = (temp_f.mean().rename("temp")
            .addBands(precip.sum().rename("precip")))

    # Mask the weights to the ERA5 footprint so sum(value*pop) and sum(pop) run
    # over the identical cell set. Population under a masked climate cell must not
    # enter the denominator -- that would drag the weighted mean toward zero.
    pop = _weight_image(ee).updateMask(base.select("temp").mask())
    img = (base
           .addBands(base.select("temp").multiply(pop).rename("temp_w"))
           .addBands(base.select("precip").multiply(pop).rename("precip_w"))
           .addBands(pop))

    # sum -> the weighted numerator/denominator; mean -> the unweighted fallback,
    # both from one reduceRegions so the two never disagree about the cell set.
    reducer = ee.Reducer.sum().combine(ee.Reducer.mean(), sharedInputs=True)
    props = _reduce_over_munis(ee, img, gdf, reducer,
                               scale, tile_scale, batch, "annual")

    rows = []
    for p in props:
        w = p.get("pop_sum")
        w = float(w) if w is not None else 0.0
        rows.append({
            "cvegeo": p.get("cvegeo"),
            "temp": _safe_ratio(p.get("temp_w_sum"), w),
            "precip": _safe_ratio(p.get("precip_w_sum"), w),
            "temp_unweighted": p.get("temp_mean"),
            "precip_unweighted": p.get("precip_mean"),
            "pop_weight": w,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["cvegeo"] = _norm_cvegeo(df["cvegeo"])
    return df


def _safe_ratio(num, den) -> float:
    """num/den, or NaN when the polygon carries no population weight at all."""
    if num is None or den is None or float(den) <= 0.0:
        return float("nan")
    return float(num) / float(den)


def _resolve_weighting(df: pd.DataFrame) -> pd.DataFrame:
    """Label how each municipality's value was produced, and apply the fallback.

    A municipality with no WorldPop weight under the ERA5-Land mask has no
    population-weighted mean to compute (0/0), so it takes the unweighted polygon
    mean instead. That is a different quantity and is recorded as such in
    `climate_weighting` rather than blended in silently.
    """
    df = df.copy()
    unweighted_ok = df["temp_unweighted"].notna() | df["precip_unweighted"].notna()
    needs_fallback = df["temp"].isna() & unweighted_ok

    df["climate_weighting"] = np.where(df["temp"].notna(),
                                       "population_weighted", "no_valid_cells")
    df.loc[needs_fallback, "climate_weighting"] = "unweighted_fallback"
    df.loc[needs_fallback, "temp"] = df.loc[needs_fallback, "temp_unweighted"]
    df.loc[needs_fallback, "precip"] = df.loc[needs_fallback, "precip_unweighted"]

    n_fb = int(needs_fallback.sum())
    if n_fb:
        print(f"  [weights] {n_fb} municipality/ies carry no WorldPop weight under "
              f"the ERA5 mask -> unweighted mean, flagged `unweighted_fallback`",
              flush=True)
    return df


def _impute_temp_precip(df: pd.DataFrame) -> pd.DataFrame:
    """IDW-KNN fill temp/precip for municipalities the ERA5-Land land mask missed."""
    centroids = _load_centroids()
    df = df.merge(centroids, on="cvegeo", how="left")
    n_filled = 0
    for c in ("temp", "precip"):
        if df[c].isna().any():
            before = int(df[c].isna().sum())
            df[c] = _idw_knn_fill(df[c], df["lat"], df["lon"]).to_numpy()
            n_filled += before - int(df[c].isna().sum())
    if n_filled:
        print(f"  [impute] IDW-KNN filled {n_filled} temp/precip cell(s)", flush=True)
    filled = (df["climate_weighting"] == "no_valid_cells") & df["temp"].notna()
    df.loc[filled, "climate_weighting"] = "idw_knn_fill"
    return df.drop(columns=["lat", "lon"])


# ---------------------------------------------------------------------------
# Assemble the 2016-2020 average and write the per-municipality climate table.
# ---------------------------------------------------------------------------
def build_all(muni_gdf=None) -> pd.DataFrame:
    """2016-2020 population-weighted temp/precip per muni -> interim/climate_gee.parquet."""
    cfg = _config()
    ensure_dirs(cfg)
    log = get_logger(STEP, cfg)
    log.info("=" * 78)
    log.info("STEP %s  (GEE / ERA5-Land, window %s-%s mean)",
             STEP, CLIMATE_YEARS[0], CLIMATE_YEARS[-1])
    log.info("=" * 78)
    log.info("CHOICE population-weighted municipal means, weights = %s year %d. "
             "Temperature and precipitation are intensive; an unweighted polygon "
             "mean would let empty territory outvote where people live.",
             WORLDPOP_IC, WORLDPOP_YEAR)

    t0 = time.time()
    # Cache tag carries `popw`: the pre-weighting checkpoints under the old
    # `temp_precip_raw_*` name hold unweighted means and a different column set,
    # so they must not be picked up by the resume path.
    out = average_over_years("temp_precip_popw_raw", annual_temp_precip_per_muni,
                             CLIMATE_YEARS, muni_gdf=muni_gdf)
    out["cvegeo"] = _norm_cvegeo(out["cvegeo"])
    out = _resolve_weighting(out)
    out = _impute_temp_precip(out)

    assert_cvegeo_valid(out["cvegeo"], "climate_gee.cvegeo")
    if bool(out["cvegeo"].duplicated().any()):
        raise PipelineError("duplicate cvegeo in climate_gee table")

    n_pw = int((out["climate_weighting"] == "population_weighted").sum())
    log.info("      weighting: %d/%d population-weighted (WorldPop %d, %s), %s",
             n_pw, len(out), WORLDPOP_YEAR, WORLDPOP_IC,
             ", ".join(f"{int(v)} {k}" for k, v in
                       out["climate_weighting"].value_counts().items()
                       if k != "population_weighted") or "no fallbacks")
    if n_pw < len(out):
        log.warning("      %d municipality/ies did not get a population-weighted "
                    "value -- see the `climate_weighting` column; they are FLAGGED, "
                    "not dropped", len(out) - n_pw)

    log.info("      temp   (degF): n=%s  min=%.1f  mean=%.1f  max=%.1f",
             f"{int(out['temp'].notna().sum()):,}",
             out["temp"].min(), out["temp"].mean(), out["temp"].max())
    log.info("      precip (m):    n=%s  min=%.3f  mean=%.3f  max=%.3f",
             f"{int(out['precip'].notna().sum()):,}",
             out["precip"].min(), out["precip"].mean(), out["precip"].max())

    # How much the weighting actually moved things. Population-weighted climate
    # differs from the polygon mean precisely where population is concentrated
    # away from the territorial average -- the large, empty, high-out-migration
    # northern municipalities. Reported so the change is auditable, not asserted.
    both = out["temp"].notna() & out["temp_unweighted"].notna()
    if bool(both.any()):
        dt = (out.loc[both, "temp"] - out.loc[both, "temp_unweighted"])
        dp = (out.loc[both, "precip"] - out.loc[both, "precip_unweighted"])
        log.info("      weighted - unweighted:  temp mean %+.3f degF (max |%.2f|), "
                 "precip mean %+.4f m (max |%.3f|)",
                 dt.mean(), dt.abs().max(), dp.mean(), dp.abs().max())

    write_interim(cfg, out, "climate_gee", log)
    log.info("DONE  climate_gee: %d municipalities, temp + precip, %.1f min",
             len(out), (time.time() - t0) / 60.0)
    return out


def main() -> int:
    build_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_step(main, STEP))
