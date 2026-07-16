"""Shared helpers: paths, normalization, censobr readers, centroids, math.

Every other module in this pipeline imports from here. Kept deliberately small
so that the actual feature transformations live alongside their data, not
behind indirection.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RAW = HERE / "raw_data"
INTERMEDIATE = HERE / "intermediate"
FINAL = HERE / "final_data"

# Pinned to 2010 IBGE municipal boundaries (5,565 munis) — the dataset is for
# 2010, so all per-muni grouping, area computations, and zonal stats use 2010
# polygons. The 7 munis split off after 2010 (e.g., Mojuí dos Campos from
# Santarém in 2014) do NOT appear in this shapefile, matching censobr 2010's
# 5,565 codes exactly. Downloaded by `download_data.py` via geobr.
MUNI_SHAPEFILE = RAW / "shapefile" / "BR_Municipios_2010.shp"
MUNI_GPKG      = RAW / "shapefile" / "br_municipios_2010.gpkg"

for d in (INTERMEDIATE, FINAL): # create directories if they do not exist akready
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Climate averaging window
# ---------------------------------------------------------------------------
# Every climate feature is the 2005–2010 mean of its ANNUAL value, not a single
# one-year snapshot. We average the per-year results (rather than pooling 6 years
# of daily data) so sums/counts — num_degreedays, degreedays_streak, … — stay on
# their natural per-year scale instead of coming out ~6× too large.
CLIMATE_YEARS = list(range(2005, 2011))   # 2005..2010 inclusive


def average_over_years(name: str, loader, years, **kwargs) -> "pd.DataFrame":
    """Run a per-year climate `loader(year=y, **kwargs)` over `years`, average per muni.

    Each year is checkpointed to intermediate/{name}_{year}.parquet so a crash
    mid-sweep resumes (mirrors the pipeline's checkpoint-everything philosophy).
    Averaging is a per-column NaN-skipping mean grouped on code_muni, so every
    metric keeps its annual scale and a muni missing in one year still averages
    over the years it has.
    """
    frames = []
    for y in years:
        cache = INTERMEDIATE / f"{name}_{y}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            print(f"  CACHE {name}_{y}", flush=True)
        else:
            print(f"  RUN   {name}_{y} ...", flush=True)
            df = loader(year=y, **kwargs)
            if df is not None and not df.empty:
                INTERMEDIATE.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache, index=False)
        if df is not None and not df.empty:
            df = df.copy()
            df["code_muni"] = normalize_code(df["code_muni"])
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["code_muni"])
    allrows = pd.concat(frames, ignore_index=True)
    return allrows.groupby("code_muni", as_index=False).mean(numeric_only=True)


# ---------------------------------------------------------------------------
# IBGE municipality code
# ---------------------------------------------------------------------------
def normalize_code(s) -> pd.Series:
    """Force any int/float/str representation of an IBGE muni code to 7-digit string."""
    return pd.Series(s).astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(7)


def state_code(muni_code) -> pd.Series:
    return normalize_code(muni_code).str[:2]


# ---------------------------------------------------------------------------
# Censobr 2010 parquet readers
# ---------------------------------------------------------------------------
def _read_columns(parquet_path: Path, cols: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(parquet_path, format="parquet")
    have = {f.name for f in dataset.schema}
    keep = [c for c in cols if c in have]
    return dataset.to_table(columns=keep).to_pandas()


def read_population(cols: list[str]) -> pd.DataFrame:
    """Census 2010 person-level parquet with a code_muni column added."""
    # V0001: State codes
    # V0002: Muni codes
    # V0010: Sample Weights
    base = ["V0001", "V0002", "V0010"] 
    df = _read_columns(RAW / "censobr" / "population_2010.parquet", sorted(set(base + cols)))
    
    # normalize codes and make a column called code_muni for the cleaned muni codes
    df["code_muni"] = normalize_code(
        df["V0001"].astype(str).str.zfill(2) + df["V0002"].astype(str).str.zfill(5)
    )
    return df

# basically the same function as read_population but for household level data
def read_household(cols: list[str]) -> pd.DataFrame:
    base = ["V0001", "V0002", "V0010"]
    df = _read_columns(RAW / "censobr" / "household_2010.parquet", sorted(set(base + cols)))
    df["code_muni"] = normalize_code(
        df["V0001"].astype(str).str.zfill(2) + df["V0002"].astype(str).str.zfill(5)
    )
    return df


# ---------------------------------------------------------------------------
# Geographic helpers
# ---------------------------------------------------------------------------

# (TODO: Must check over this logic and make sure it works with imputation)
def load_centroids(gpkg: Path) -> pd.DataFrame:
    """Per-muni centroids (lat/lon) from the 2010 IBGE municipal boundaries."""
    import geopandas as gpd
    try:
        gdf = gpd.read_file(gpkg, layer="municipios")
    except Exception:
        gdf = gpd.read_file(gpkg)
    key = "code_muni" if "code_muni" in gdf.columns else "CD_MUN"
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4674")  # geobr default (SIRGAS 2000)
    # Compute centroids in a projected CRS (Brazil Polyconic, the same EPSG:5880
    # used for KNN distance) so the centroid is geometrically correct, then
    # convert the points back to lat/lon. Computing `.centroid` directly on a
    # geographic CRS both warns and distorts the result in degree-space.
    cent = gdf.geometry.to_crs("EPSG:5880").centroid.to_crs("EPSG:4326")
    return pd.DataFrame({
        "code_muni": normalize_code(gdf[key]).values,
        "lat":       cent.y.values,
        "lon":       cent.x.values,
    })


def load_areas(gpkg: Path) -> pd.DataFrame:
    """Per-muni polygon area in km^2 from the 2010 IBGE municipal boundaries.

    Area is computed in EPSG:5880 (SIRGAS 2000 / Brazil Polyconic), IBGE's
    official equal-area projection, so km^2 are geometrically correct — the same
    projection load_centroids uses for its centroids.
    """
    import geopandas as gpd
    try:
        gdf = gpd.read_file(gpkg, layer="municipios")
    except Exception:
        gdf = gpd.read_file(gpkg)
    key = "code_muni" if "code_muni" in gdf.columns else "CD_MUN"
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4674")  # geobr default (SIRGAS 2000)
    area_km2 = gdf.geometry.to_crs("EPSG:5880").area / 1_000_000.0
    return pd.DataFrame({
        "code_muni": normalize_code(gdf[key]).values,
        "area_km2":  area_km2.values,
    })


def load_centroids_projected(shapefile: Path) -> pd.DataFrame:
    """Same shape as load_centroids but in meters (SIRGAS Polyconic) for KNN distance."""
    import geopandas as gpd
    gdf = gpd.read_file(shapefile)
    # geobr writes `code_muni` (lowercase); IBGE's own shapefiles use `CD_MUN`.
    key = "code_muni" if "code_muni" in gdf.columns else "CD_MUN"
    gdf["code_muni"] = normalize_code(gdf[key])
    proj = gdf.to_crs("EPSG:5880")
    pts = proj.geometry.centroid
    return pd.DataFrame({"code_muni": gdf["code_muni"], "x": pts.x.values, "y": pts.y.values})


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Weighted aggregation (used by census features)
# ---------------------------------------------------------------------------
def weighted_share(values: pd.Series, weights: pd.Series, condition: pd.Series) -> float:
    w = weights.astype(float)
    return float((w * condition.astype(float)).sum() / w.sum()) if w.sum() else np.nan


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.astype(float)
    v = pd.to_numeric(values, errors="coerce")
    mask = v.notna() & w.notna()
    return float((v[mask] * w[mask]).sum() / w[mask].sum()) if w[mask].sum() else np.nan


def weighted_pct(condition: pd.Series, weights: pd.Series) -> float:
    """Weighted % of rows satisfying `condition` (0/1)."""
    w = pd.to_numeric(weights, errors="coerce")
    c = condition.astype(float)
    mask = c.notna() & w.notna()
    return float(100.0 * (w[mask] * c[mask]).sum() / w[mask].sum()) if w[mask].sum() else np.nan


def weighted_median(values, weights) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return np.nan
    order = np.argsort(v[mask])
    vs, ws = v[mask][order], w[mask][order]
    csum = np.cumsum(ws)
    return float(vs[np.searchsorted(csum, 0.5 * csum[-1])])
