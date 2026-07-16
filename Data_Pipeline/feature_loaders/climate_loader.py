"""Per-muni climate features, all CDS-free via Google Earth Engine.

WHY GEE (and not the Copernicus CDS or the ARCO-ERA5 GCS Zarr)
-------------------------------------------------------------
The goal was to avoid the Copernicus CDS limits (queue + the per-request "cost
limit" that rejects any daily-statistics request bigger than one year). The
obvious Google option — the ARCO-ERA5 public bucket — does NOT help: its hourly
Zarr is chunked one *global* field per timestep, so slicing a 30-year Brazil box
still streams ~1 TB. Earth Engine is the right tool because it runs the whole
computation **server-side** and returns only the ~5,570 per-municipality numbers
— no CDS queue, no cost cap, and essentially zero local disk.

DATASET
-------
`ECMWF/ERA5_LAND/DAILY_AGGR` (0.1° ~11 km daily, 1950→present) for everything
except UV, which is only in `ECMWF/ERA5/HOURLY` (0.25°) and is aggregated
hourly→daily server-side.

AUTH
----
Needs Earth Engine — the same auth NDVI uses (`_init_ee` from ndvi_loader).
In the container the mounted ~/.config/earthengine + ~/.config/gcloud provide it.

VARIABLES PRODUCED (this is the whole surface; everything else was trimmed)
--------------------------------------------------------------------------
Registered as five loaders in features.CANONICAL_LOADERS. Every one is the
2005–2010 average of the annual metric (utils.CLIMATE_YEARS), not a single-year
snapshot — the `*_avg` wrappers average the per-year single-year reducers,
checkpointing each year to intermediate/. z_from_baseline is an average by
construction (2005–2010 mean vs a 1980–2000 baseline).

  wind        → wind_per_muni_avg
    wind_mean                Annual-mean 10 m v-wind (m/s), 2005–2010 mean.

  wet_bulb    → wet_bulb_per_muni_avg
    wet_bulb_F               Wet-bulb temperature (°F): heat-stress proxy from
                             combined heat + humidity. MetPy on annual-mean
                             pressure/temperature/dewpoint. Tw is nonlinear in
                             its inputs, so Tw(annual-mean inputs) ≈ (not =) the
                             mean of daily Tw — fine for an annual summary.

  uv_annual   → uv_log_per_muni_avg
    uv_log_mean              Per-muni mean of the annual-mean log1p daily UV
                             (downward UV at the surface). Per-year raw checkpoints
                             use column uv_log_mean_annual (suffix pinned to
                             `_annual` so every year aligns); the 6-year average is
                             renamed to uv_log_mean.

  degreedays_extra → degreedays_extra_per_muni_avg
    num_degreedays           Mean annual count of days with daily max > 86 °F.
    degreedays_streak        Mean annual longest run of consecutive > 86 °F days
                             (per-muni max of the per-pixel streaks).

  climate_baseline → climate_baseline_per_muni  (the heavy one — 27 GEE years)
    temp                     2005–2010 mean annual temperature (annual-mean of
                             daily-MEAN temp, °F) — the z_from_baseline target
                             mean, emitted directly. Drives temp_diff in the dyad.
    precip                   2005–2010 mean annual precipitation (annual-sum, m)
                             — the z_from_baseline target mean. Drives precip_diff.
    z_from_baseline_temp     Z-score of the 2005–2010 mean temp vs the per-muni
                             1980–2000 baseline mean/std.
    z_from_baseline_precip   Same for precip.

The 1980–2000 + 2005–2010 annual temp/precip panel behind climate_baseline is the
expensive step (27 GEE years); it is checkpointed to
intermediate/climate_zbaseline_panel_wide.csv so a crash mid-build resumes, and
temp/precip fall out of it for free (they are its 2005–2010 target means).

CAVEATS
-------
* ERA5-Land (0.1°) ≠ ERA5 (0.25°); values differ slightly from the CDS path.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from utils import (INTERMEDIATE, MUNI_GPKG, normalize_code,
                   CLIMATE_YEARS, average_over_years)
from ndvi_loader import _init_ee   # reuse the exact EE init NDVI already uses

ERA5_LAND_DAILY = "ECMWF/ERA5_LAND/DAILY_AGGR"
ERA5_HOURLY     = "ECMWF/ERA5/HOURLY"
UV_HOURLY_BAND  = "downward_uv_radiation_at_the_surface"

K = 273.15            # °C → K offset
THRESHOLD_F = 86.0    # degree-day / hot-day threshold
TARGET_YEAR = 2010    # default year for the single-year reducers (below)

# z_from_baseline: standardize the 2005–2010 average against a 1980–2000 baseline.
Z_BASELINE_YEARS = list(range(1980, 2001))   # 1980..2000 inclusive
Z_TARGET_YEARS   = CLIMATE_YEARS             # 2005..2010 (the shared climate window)
WIDE_PATH        = INTERMEDIATE / "climate_zbaseline_panel_wide.csv"


# ---------------------------------------------------------------------------
# Shared GEE plumbing (one gdf loader + one batched reduceRegions helper reused
# by every reducer below).
# ---------------------------------------------------------------------------
def _load_muni_gdf(muni_shapefile=None, n_sample=None):
    """Read + reproject + simplify the 2010 muni polygons for GEE reduceRegions."""
    import geopandas as gpd
    if muni_shapefile is None:
        from utils import MUNI_SHAPEFILE
        muni_shapefile = MUNI_SHAPEFILE
    gdf = gpd.read_file(muni_shapefile)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4674")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    key = "code_muni" if "code_muni" in gdf.columns else "CD_MUN"
    gdf = gdf.copy()
    gdf["code_muni"] = normalize_code(gdf[key])
    # Simplify dense Brazilian borders so each batch stays under GEE's payload cap.
    gdf["geometry"] = gdf.geometry.simplify(tolerance=0.005, preserve_topology=True)
    if n_sample:
        gdf = gdf.sample(n=min(n_sample, len(gdf)), random_state=0).reset_index(drop=True)
    return gdf


def _reduce_over_munis(ee, image, gdf, reducer, scale, tile_scale, batch, tag):
    """reduceRegions(`reducer`) of `image` over each muni polygon → list of prop dicts."""
    rows: list[dict] = []
    n = len(gdf)
    for start in range(0, n, batch):
        sub = gdf.iloc[start:start + batch]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry(r.geometry.__geo_interface__),
                       {"code_muni": r["code_muni"]})
            for _, r in sub.iterrows()
        ])
        stats = image.reduceRegions(collection=fc, reducer=reducer,
                                    scale=scale, tileScale=tile_scale)
        for f in stats.getInfo()["features"]:
            rows.append(f["properties"])
        print(f"  [gee/{tag}] munis {min(start + batch, n)}/{n}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# 1. Mean 10 m v-wind (m/s).
# ---------------------------------------------------------------------------
def wind_per_muni(year: int = TARGET_YEAR,
                  muni_shapefile=None,
                  n_sample: int | None = None,
                  batch: int = 150,
                  scale: int = 11132,
                  tile_scale: int = 16) -> pd.DataFrame:
    """Per-muni annual-mean 10 m v-wind → DataFrame[code_muni, wind_mean]."""
    import ee
    _init_ee()

    gdf = _load_muni_gdf(muni_shapefile, n_sample)
    land = (ee.ImageCollection(ERA5_LAND_DAILY)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))
    img = land.select("v_component_of_wind_10m").mean().rename("wind")  # per-pixel annual mean

    props = _reduce_over_munis(ee, img, gdf, ee.Reducer.mean(),
                               scale, tile_scale, batch, "wind")
    df = pd.DataFrame([{"code_muni": p.get("code_muni"),
                        "wind_mean": p.get("wind")} for p in props])
    if not df.empty:
        df["code_muni"] = normalize_code(df["code_muni"])
    return df


# ---------------------------------------------------------------------------
# 2. Wet-bulb temperature (heat-stress proxy). GEE reduces annual-mean
#    surface_pressure / temperature_2m / dewpoint_2m to one scalar per muni;
#    MetPy computes Tw client-side on those three means.
# ---------------------------------------------------------------------------
def wet_bulb_per_muni(year: int = TARGET_YEAR,
                      muni_shapefile=None,
                      n_sample: int | None = None,
                      batch: int = 150,
                      scale: int = 11132,
                      tile_scale: int = 16) -> pd.DataFrame:
    """Per-muni annual-mean wet-bulb temperature → DataFrame[code_muni, wet_bulb_F]."""
    import ee
    from metpy.calc import wet_bulb_temperature
    from metpy.units import units
    _init_ee()

    gdf = _load_muni_gdf(muni_shapefile, n_sample)
    land = (ee.ImageCollection(ERA5_LAND_DAILY)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))
    img = (land.select("surface_pressure").mean().rename("pressure")
           .addBands(land.select("temperature_2m").mean().rename("temperature"))
           .addBands(land.select("dewpoint_temperature_2m").mean().rename("dewpoint")))

    props = _reduce_over_munis(ee, img, gdf, ee.Reducer.mean(),
                               scale, tile_scale, batch, "wetbulb")
    df = pd.DataFrame([{
        "code_muni":     p.get("code_muni"),
        "pressure_Pa":   p.get("pressure"),
        "temperature_K": p.get("temperature"),
        "dewpoint_K":    p.get("dewpoint"),
    } for p in props])
    if df.empty:
        return df
    df["code_muni"] = normalize_code(df["code_muni"])

    valid = df[["pressure_Pa", "temperature_K", "dewpoint_K"]].notna().all(axis=1)
    tw_c = np.full(len(df), np.nan)
    if valid.any():
        P  = df.loc[valid, "pressure_Pa"].to_numpy()   * units.pascal
        T  = df.loc[valid, "temperature_K"].to_numpy() * units.kelvin
        Td = df.loc[valid, "dewpoint_K"].to_numpy()    * units.kelvin
        tw_c[valid.to_numpy()] = wet_bulb_temperature(P, T, Td).to(units.degC).magnitude
    return pd.DataFrame({"code_muni": df["code_muni"],
                         "wet_bulb_F": tw_c * 1.8 + 32.0})


# ---------------------------------------------------------------------------
# 3. UV: only in ERA5/HOURLY (0.25°). Aggregate hourly→daily as a streaming
#    collection (each daily image = a cheap 24-hour mean), then per-pixel mean of
#    log1p daily UV, then per-muni spatial mean.
# ---------------------------------------------------------------------------
def _uv_daily_collection(ee, year: int):
    """365 daily-mean UV images (each = mean of that day's 24 hourly values)."""
    hourly = (ee.ImageCollection(ERA5_HOURLY).select(UV_HOURLY_BAND)
              .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))
    start = ee.Date(f"{year}-01-01")
    def daily(d):
        s = start.advance(ee.Number(d), "day")
        return hourly.filterDate(s, s.advance(1, "day")).mean().rename("uv")
    return ee.ImageCollection(ee.List.sequence(0, 364).map(daily))


def uv_log_per_muni(year: int = TARGET_YEAR,
                    suffix: str | None = None,
                    muni_shapefile=None,
                    n_sample: int | None = None,
                    batch: int = 150,
                    scale: int = 27830,          # 0.25° native (ERA5/HOURLY)
                    tile_scale: int = 16) -> pd.DataFrame:
    """Per-muni mean log1p daily UV → DataFrame[code_muni, uv_log_mean{suffix}].

    `suffix` overrides the column suffix (default `_annual_{year}`); the 2005–2010
    averaging wrapper pins it to `_annual` (no year) so every year's column aligns.
    """
    import ee
    _init_ee()
    sfx = suffix if suffix is not None else f"_annual_{year}"

    gdf = _load_muni_gdf(muni_shapefile, n_sample)
    uv_daily = _uv_daily_collection(ee, year).map(
        lambda i: i.max(0).add(1).log())          # log1p (GEE has no .log1p)
    img = uv_daily.mean().rename("uvl")

    props = _reduce_over_munis(ee, img, gdf, ee.Reducer.mean(),
                               scale, tile_scale, batch, "uv")
    df = pd.DataFrame([{"code_muni": p.get("code_muni"),
                        f"uv_log_mean{sfx}": p.get("uvl")} for p in props])
    if not df.empty:
        df["code_muni"] = normalize_code(df["code_muni"])
    return df


# ---------------------------------------------------------------------------
# 4. Hot-day count + longest hot streak (daily max > 86 °F).
# ---------------------------------------------------------------------------
def degreedays_extra_per_muni(year: int = TARGET_YEAR,
                              muni_shapefile=None,
                              n_sample: int | None = None,
                              batch: int = 150,
                              scale: int = 11132,
                              tile_scale: int = 16) -> pd.DataFrame:
    """Per-muni [num_degreedays, degreedays_streak] for daily max > 86 °F in `year`."""
    import ee
    _init_ee()

    gdf = _load_muni_gdf(muni_shapefile, n_sample)
    land = (ee.ImageCollection(ERA5_LAND_DAILY)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select("temperature_2m_max"))
    # Per-day 0/1 hot mask: K → °C → °F → (> 86).
    hot = land.map(
        lambda i: i.subtract(K).multiply(1.8).add(32).gt(THRESHOLD_F).rename("hot"))
    num_dd = hot.sum().rename("num_dd")                       # per-pixel count of hot days

    # Per-pixel longest consecutive-hot run. State carries cur=current run, mx=max
    # seen; multiplying (cur+1) by the 0/1 mask resets cur to 0 on cold days.
    initial = ee.Image.constant([0, 0]).rename(["cur", "mx"]).toFloat()

    def step(img, prev):
        prev = ee.Image(prev)
        h = ee.Image(img).select("hot")
        new_cur = prev.select("cur").add(1).multiply(h)
        new_mx = new_cur.max(prev.select("mx"))
        return new_cur.rename("cur").addBands(new_mx.rename("mx"))

    streak = ee.Image(hot.iterate(step, initial)).select("mx").rename("streak")

    # One pass: mean gives the muni's typical hot-day COUNT (like DegreeDays);
    # max gives the longest STREAK observed at any pixel (like the source script).
    image = num_dd.addBands(streak)
    reducer = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True)
    props = _reduce_over_munis(ee, image, gdf, reducer,
                               scale, tile_scale, batch, "dd")
    df = pd.DataFrame([{
        "code_muni":         p.get("code_muni"),
        "num_degreedays":    p.get("num_dd_mean"),
        "degreedays_streak": p.get("streak_max"),
    } for p in props])
    if not df.empty:
        df["code_muni"] = normalize_code(df["code_muni"])
    return df


# ---------------------------------------------------------------------------
# 5. Annual temp/precip reducer (one number per muni per year) — the building
#    block of the 1980–2010 panel and the source of temp / precip.
# ---------------------------------------------------------------------------
def _annual_temp_precip_per_muni(year: int,
                                 muni_shapefile=None,
                                 batch: int = 150,
                                 scale: int = 11132,
                                 tile_scale: int = 16) -> pd.DataFrame:
    """Per-muni [temp (annual-mean daily-MEAN °F), precip (annual-sum m)] for `year`."""
    import ee
    _init_ee()

    gdf = _load_muni_gdf(muni_shapefile)
    land = (ee.ImageCollection(ERA5_LAND_DAILY)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))
    temp_f = land.select("temperature_2m").map(                    # daily mean → °F
        lambda i: i.subtract(K).multiply(1.8).add(32))
    precip = land.select("total_precipitation_sum")                # daily total m
    img = (temp_f.mean().rename("temp")
           .addBands(precip.sum().rename("precip")))

    props = _reduce_over_munis(ee, img, gdf, ee.Reducer.mean(),
                               scale, tile_scale, batch, "annual")
    df = pd.DataFrame([{"code_muni": p.get("code_muni"),
                        "temp":   p.get("temp"),
                        "precip": p.get("precip")} for p in props])
    if not df.empty:
        df["code_muni"] = normalize_code(df["code_muni"])
    return df


def _build_wide_panel(years, muni_shapefile=None, batch=150,
                      scale=11132, tile_scale=16) -> pd.DataFrame:
    """Per-(muni, year) temp_{y}/precip_{y} panel for `years`, cached to WIDE_PATH.

    Only the years not already in the cached panel are computed; overlapping years
    are reused from disk. So shifting the target window (e.g. 2005–2010 → 2004–2009)
    reuses every year the old panel already holds and only computes the new one(s),
    then rewrites the panel with them merged in.
    """
    wide = None
    if WIDE_PATH.exists():
        print(f"  [cache] reading {WIDE_PATH.name}", flush=True)
        wide = pd.read_csv(WIDE_PATH)
        wide["code_muni"] = normalize_code(wide["code_muni"])

    have = set() if wide is None else {int(c[len("temp_"):])
                                       for c in wide.columns if c.startswith("temp_")}
    missing = [y for y in years if y not in have]
    if wide is not None and not missing:
        return wide

    t0 = time.time()
    for i, y in enumerate(missing, 1):
        ty = time.time()
        print(f"\n  === panel year {y}  ({i}/{len(missing)}) ===", flush=True)
        yr = (_annual_temp_precip_per_muni(y, muni_shapefile, batch, scale, tile_scale)
              .rename(columns={"temp": f"temp_{y}", "precip": f"precip_{y}"}))
        wide = yr if wide is None else wide.merge(yr, on="code_muni", how="outer")
        elapsed = time.time() - t0
        eta = elapsed / i * (len(missing) - i)
        print(f"  === year {y} done in {time.time() - ty:.0f}s "
              f"| total {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min ===", flush=True)

    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    wide.to_csv(WIDE_PATH, index=False)
    return wide


def _impute_panel(wide: pd.DataFrame) -> pd.DataFrame:
    """IDW-KNN fill each per-year temp/precip column from neighbouring munis.

    Mirrors dyad.impute's climate path: ERA5-Land returns null for polygons that
    miss the land mask (e.g. Fernando de Noronha), so fill those from k=8 nearest
    neighbours weighted by inverse squared haversine distance.
    """
    from dyad import _idw_knn_fill
    from utils import load_centroids

    centroids = load_centroids(MUNI_GPKG)
    wide = wide.merge(centroids, on="code_muni", how="left")
    year_cols = [c for c in wide.columns
                 if c.startswith("temp_") or c.startswith("precip_")]
    n_filled = 0
    for c in year_cols:
        if wide[c].isna().any():
            before = wide[c].isna().sum()
            wide[c] = _idw_knn_fill(wide[c], wide["lat"], wide["lon"])
            n_filled += before - wide[c].isna().sum()
    print(f"  [impute] IDW-KNN filled {n_filled} (muni × year) cells across "
          f"{len(year_cols)} columns", flush=True)
    return wide.drop(columns=["lat", "lon"])


def climate_baseline_per_muni(baseline_years=None,
                              target_years=None,
                              muni_shapefile=None,
                              batch: int = 150,
                              scale: int = 11132,
                              tile_scale: int = 16) -> pd.DataFrame:
    """temp, precip, z_from_baseline_temp, z_from_baseline_precip in one panel.

    Builds the (baseline_years ∪ target_years) annual temp/precip panel (heavy: one
    GEE pass per year, checkpointed), IDW-KNN imputes it, then per muni:
      temp   = mean_{target} annual temp        (the 2005–2010 average temperature)
      precip = mean_{target} annual precip       (the 2005–2010 average precipitation)
      z_from_baseline_* = (mean_{target} − mean_{baseline}) / std_{baseline}
    Defaults: baseline = 1980–2000, target = 2005–2010 (the shared climate window),
    so temp/precip are the recent climate normal and the z-anomaly compares it to
    the early-baseline normal. temp/precip come out of the same panel for free.
    """
    baseline_years = list(baseline_years if baseline_years is not None else Z_BASELINE_YEARS)
    target_years   = list(target_years   if target_years   is not None else Z_TARGET_YEARS)
    years = sorted(set(baseline_years) | set(target_years))

    wide = _build_wide_panel(years, muni_shapefile, batch, scale, tile_scale)
    wide = _impute_panel(wide)

    def _stats(prefix, cols_years):
        M = wide[[f"{prefix}_{y}" for y in cols_years]].to_numpy(dtype=float)
        return np.nanmean(M, axis=1), np.nanstd(M, axis=1, ddof=0)

    t_base_mean, t_base_std = _stats("temp",   baseline_years)
    p_base_mean, p_base_std = _stats("precip", baseline_years)
    t_tgt_mean, _ = _stats("temp",   target_years)     # 2005–2010 mean, per muni
    p_tgt_mean, _ = _stats("precip", target_years)

    return pd.DataFrame({
        "code_muni":              wide["code_muni"].values,
        "temp":                   t_tgt_mean,
        "precip":                 p_tgt_mean,
        "z_from_baseline_temp":   np.where(t_base_std > 0,
                                           (t_tgt_mean - t_base_mean) / t_base_std, np.nan),
        "z_from_baseline_precip": np.where(p_base_std > 0,
                                           (p_tgt_mean - p_base_mean) / p_base_std, np.nan),
    })


# ---------------------------------------------------------------------------
# 6. 2005–2010 averaged variants — the pipeline's registered loaders. Each is the
#    per-municipality mean of the annual value over CLIMATE_YEARS, every per-year
#    result checkpointed to intermediate/ (see utils.average_over_years).
#    climate_baseline is already an average by construction and is registered raw.
#    Column names are unchanged; only the values become 6-year means.
# ---------------------------------------------------------------------------
def wind_per_muni_avg(**kwargs) -> pd.DataFrame:
    """2005–2010 average of wind_per_muni → DataFrame[code_muni, wind_mean]."""
    return average_over_years("wind_raw", wind_per_muni, CLIMATE_YEARS, **kwargs)


def wet_bulb_per_muni_avg(**kwargs) -> pd.DataFrame:
    """2005–2010 average of wet_bulb_per_muni → DataFrame[code_muni, wet_bulb_F]."""
    return average_over_years("wet_bulb_raw", wet_bulb_per_muni, CLIMATE_YEARS, **kwargs)


def uv_log_per_muni_avg(**kwargs) -> pd.DataFrame:
    """2005–2010 average of uv_log_per_muni → [code_muni, uv_log_mean].

    The per-year raw checkpoints are written with suffix '_annual' (column
    uv_log_mean_annual); the averaged output is renamed to uv_log_mean (the name
    in variables_temp.txt), without disturbing the caches.
    """
    df = average_over_years("uv_annual_raw", uv_log_per_muni, CLIMATE_YEARS,
                            suffix="_annual", **kwargs)
    return df.rename(columns={"uv_log_mean_annual": "uv_log_mean"})


def degreedays_extra_per_muni_avg(**kwargs) -> pd.DataFrame:
    """2005–2010 average of degreedays_extra_per_muni → [num_degreedays, degreedays_streak]."""
    return average_over_years("degreedays_raw", degreedays_extra_per_muni, CLIMATE_YEARS, **kwargs)
