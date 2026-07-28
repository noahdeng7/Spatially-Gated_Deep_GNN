"""
fetch_mgn_arcgis.py -- fetch municipal geometry from INEGI's ArcGIS Feature Service.

A FALLBACK for the Marco Geoestadistico, not the preferred source. Use this when
the INEGI portal download is unavailable -- as of 2026-07-21 the published direct
link is dead and the portal is JavaScript-rendered, so there is no scriptable
route to the canonical file.

WHAT THIS FETCHES
-----------------
An Esri Feature Service published by an INEGI/Esri account, carrying the
municipal layer (service name `00mun`) with CVEGEO, CVE_ENT, CVE_MUN and NOMGEO.

    https://services5.arcgis.com/0TAhhrymXRLOcVMe/arcgis/rest/services/00mun/FeatureServer

*** TWO CAVEATS, BOTH IMPORTANT ***

1. The service is titled "... diciembre 2022 ... (Prueba)". "Prueba" means TEST.
   It is not billed as a production product. Treat the geometry as provisional
   and prefer the official download when you can get it.

2. It is the DECEMBER 2022 vintage, not the census-2020 one. It carries 2,475
   municipalities against the 2020 census's 2,469. The six extras are post-census
   creations, and the crosswalk folds them back into their parents -- dissolving
   child into parent unions the polygons and reconstructs the 2020 boundary. The
   vintage guard in geo.assert_geometry_vintage() verifies this held.

Run:
    python src/fetch_mgn_arcgis.py
    python src/fetch_mgn_arcgis.py --out data/raw/mgn2022
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PipelineError, ensure_dirs, get_logger, load_config, rel, resolve, run_step,
)

STEP = "fetch_mgn_arcgis"

SERVICE = ("https://services5.arcgis.com/0TAhhrymXRLOcVMe/arcgis/rest/services/"
           "00mun/FeatureServer/0")
PAGE = 500          # well under the service's maxRecordCount of 2000
EXPECTED_FIELDS = ["CVEGEO", "CVE_ENT", "CVE_MUN", "NOMGEO"]


def fetch_all(log) -> "object":
    """Page through the service and return a GeoDataFrame in EPSG:4326."""
    import geopandas as gpd
    import requests
    from shapely.geometry import shape

    # Ask for WGS84 directly. The service stores Web Mercator (102100); letting
    # the server reproject avoids a second transform on our side.
    params = {
        "where": "1=1",
        "outFields": ",".join(EXPECTED_FIELDS),
        "returnGeometry": "true",
        "outSR": 4326,
        "f": "geojson",
    }

    count = requests.get(f"{SERVICE}/query",
                         params={"where": "1=1", "returnCountOnly": "true",
                                 "f": "json"}, timeout=60).json().get("count")
    log.info("service reports %s features", f"{count:,}")

    records, offset = [], 0
    while True:
        r = requests.get(f"{SERVICE}/query",
                         params={**params, "resultOffset": offset,
                                 "resultRecordCount": PAGE}, timeout=180)
        r.raise_for_status()
        feats = r.json().get("features", [])
        for f in feats:
            props = f.get("properties") or {}
            geom = f.get("geometry")
            if geom is None:
                log.warning("      feature %s has no geometry; skipped",
                            props.get("CVEGEO"))
                continue
            records.append({**{k: props.get(k) for k in EXPECTED_FIELDS},
                            "geometry": shape(geom)})
        log.info("  fetched %s / %s", f"{len(records):,}", f"{count:,}")
        if len(feats) < PAGE:
            break
        offset += PAGE

    if count and len(records) != count:
        raise PipelineError(
            f"fetched {len(records):,} features but the service reports "
            f"{count:,}. A partial download would silently shrink the municipal "
            "universe -- refusing to write it."
        )

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf["CVEGEO"] = gdf["CVEGEO"].astype("string").str.strip().str.zfill(5)
    return gdf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/mgn2022",
                    help="destination directory")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(STEP, cfg)
    log.info("=" * 78)
    log.info("FETCH municipal geometry from ArcGIS Feature Service")
    log.info("=" * 78)
    log.warning("FALLBACK SOURCE. The service is labelled 'Prueba' (test) and is "
                "the December 2022 vintage, not census 2020.")
    log.warning("Prefer the official INEGI download when it is obtainable; see "
                "docs/MANUAL_DOWNLOADS.md")

    gdf = fetch_all(log)

    bad = gdf["CVEGEO"].isna() | (gdf["CVEGEO"].str.len() != 5)
    if bool(bad.any()):
        raise PipelineError(f"{int(bad.sum())} feature(s) have an unusable CVEGEO")
    if bool(gdf["CVEGEO"].duplicated().any()):
        n = int(gdf["CVEGEO"].duplicated().sum())
        raise PipelineError(f"{n} duplicate CVEGEO value(s) in the service output")

    # GeoPackage rather than shapefile: shapefiles truncate field names to 10
    # characters, cannot hold UTF-8 reliably, and scatter a layer across five
    # sidecar files any one of which can go missing.
    out_dir = resolve(cfg, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "00mun.gpkg"
    gdf.to_file(out, layer="municipios", driver="GPKG")

    log.info("WROTE %s  features=%s  crs=%s  size=%.1f MB",
             rel(out), f"{len(gdf):,}", gdf.crs, out.stat().st_size / 1e6)
    log.info("")
    log.info("NEXT: point config.yaml at it --")
    log.info("  geometry.municipal_layer: %s", f"{args.out}/00mun.gpkg")
    log.info("Then run any geometry step; assert_geometry_vintage() will verify "
             "the harmonized universe matches the %s census.",
             cfg["project"]["census_year"])
    return 0


if __name__ == "__main__":
    raise SystemExit(run_step(main, STEP))
