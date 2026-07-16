"""NDVI per municipality from MODIS MOD13Q1 v061 (NASA via Google Earth Engine).

Auth (one-time):
  pip install earthengine-api geemap
  earthengine authenticate     # browser flow, host-side
  # For Docker, set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON
  # and mount that file into the container.

Returns 1 column: ndvi (the September NDVI mean) — keyed by code_muni.
Missing munis are filled by the standard IDW imputer applied in dyad.impute().
(The GEE reducer still computes min/max/std/median server-side for free, but
only the mean is surfaced; it's the only NDVI stat the final dataset keeps.)

The pipeline registers `ndvi_per_muni_avg`, which is the 2005–2010 average of
the single-year September reducer (mirrors the climate `*_avg` loaders: per-year
results are checkpointed to intermediate/ and averaged per muni). Column name
stays `ndvi`; only the value becomes a 6-year mean.
"""
from __future__ import annotations

import os
import pandas as pd

from utils import RAW, normalize_code, average_over_years, CLIMATE_YEARS

# September-NDVI averaging window: the same 2005–2010 span as the climate
# variables (utils.CLIMATE_YEARS), so all multi-year means share one window.
NDVI_YEARS = CLIMATE_YEARS   # 2005..2010 inclusive


def _init_ee() -> None:
    """Initialize Earth Engine — service-account if creds set, else user creds.

    Recent earthengine-api requires a `project` parameter. We read it from
    ~/.config/earthengine/credentials (written by `earthengine authenticate`)
    if present; otherwise the GEE_PROJECT env var.
    """
    import ee, json
    sa_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_json and os.path.exists(sa_json):
        creds = json.load(open(sa_json))
        if creds.get("type") == "service_account":
            credentials = ee.ServiceAccountCredentials(creds["client_email"], sa_json)
            ee.Initialize(credentials, project=creds.get("project_id") or creds.get("quota_project_id"))
            return

    # Default: read project ID from the earthengine credentials file (modern flow);
    # actual OAuth token comes from ~/.config/gcloud Application Default Credentials.
    project = os.environ.get("GEE_PROJECT")
    ee_creds_path = os.path.expanduser("~/.config/earthengine/credentials")
    if not project and os.path.exists(ee_creds_path):
        try:
            project = json.load(open(ee_creds_path)).get("project")
        except Exception:
            pass
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()


def ndvi_per_muni(year: int = 2010,
                  muni_shapefile: str | None = None) -> pd.DataFrame:
    """Pull MOD13Q1 NDVI for September of `year`, zonal stats per muni polygon.

    Method:
      • MOD13Q1 v061, filtered to Sept 1 — Oct 1 of `year`
      • Scale factor 0.0001 applied
      • Median composite across the ~2 September 16-day windows
      • reduceRegions with a mean reducer on each muni polygon
        (NOT centroid point sample) → the single `ndvi` column
    """
    import ee
    import geopandas as gpd
    _init_ee()

    if muni_shapefile is None:
        from utils import MUNI_SHAPEFILE
        muni_shapefile = MUNI_SHAPEFILE

    gdf = gpd.read_file(muni_shapefile)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4674")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    key = "code_muni" if "code_muni" in gdf.columns else "CD_MUN"
    gdf["code_muni"] = normalize_code(gdf[key])

    # September composite, scale factor applied.
    composite = (ee.ImageCollection("MODIS/061/MOD13Q1")
                 .select("NDVI")
                 .filterDate(f"{year}-09-01", f"{year}-10-01")
                 .map(lambda img: img.multiply(0.0001))
                 .median())

    reducer = ee.Reducer.mean()

    # Simplify polygons to ~0.005° (~500m, below the 300m raster resolution we
    # reduce against). Brazilian munis have very dense coastlines/borders that
    # otherwise blow past GEE's 10 MB request payload limit.
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(tolerance=0.005, preserve_topology=True)

    # GEE caps getInfo at 5000 features AND request payload at 10 MB. With
    # simplified geometries, ~100 polygons/batch stays comfortably under both.
    BATCH = 100
    all_rows: list[dict] = []
    for start in range(0, len(gdf), BATCH):
        sub = gdf.iloc[start:start + BATCH]
        feats = []
        for _, row in sub.iterrows():
            geom_json = row.geometry.__geo_interface__
            feats.append(ee.Feature(ee.Geometry(geom_json),
                                    {"code_muni": row["code_muni"]}))
        fc = ee.FeatureCollection(feats)
        stats = composite.reduceRegions(
            collection=fc,
            reducer=reducer,
            scale=300,
            tileScale=4,
        )
        for f in stats.getInfo()["features"]:
            p = f["properties"]
            all_rows.append({
                "code_muni": p.get("code_muni"),
                "ndvi":      p.get("mean"),
            })

    df = pd.DataFrame(all_rows)
    if df.empty:
        return pd.DataFrame(columns=["code_muni", "ndvi"])
    df["code_muni"] = normalize_code(df["code_muni"])
    return df[["code_muni", "ndvi"]]


def ndvi_per_muni_avg(**kwargs) -> pd.DataFrame:
    """2005–2010 average of ndvi_per_muni → DataFrame[code_muni, ndvi].

    Runs the single-year September reducer once per year in NDVI_YEARS,
    checkpointing each to intermediate/ndvi_raw_{year}.parquet, then takes the
    per-muni NaN-skipping mean (see utils.average_over_years).
    """
    return average_over_years("ndvi_raw", ndvi_per_muni, NDVI_YEARS, **kwargs)
