"""Per-muni rail/road/water densities from HOTOSM via the HDX API.

Public function: all_densities() → 2 cols:
  railway_density, road_density (meters per km²)

Sources (no auth):
  - https://data.humdata.org/dataset/hotosm_bra_roads
  - https://data.humdata.org/dataset/hotosm_bra_railways
  - https://data.humdata.org/dataset/hotosm_bra_waterways

The HDX package_show API returns each dataset's resource list with direct
download URLs (GeoPackage preferred, smaller than shapefile bundles).
"""
from __future__ import annotations

import datetime
import json
import urllib.request
from pathlib import Path

import pandas as pd
import requests

from utils import RAW, normalize_code, INTERMEDIATE

HDX_API = "https://data.humdata.org/api/3/action/package_show?id="
HOTOSM_DIR = RAW / "hotosm"
CACHE = INTERMEDIATE   # data/intermediate — shared with the pipeline's checkpoints

LAYERS = [
    ("hotosm_bra_roads",      "road_density"),
    ("hotosm_bra_railways",   "railway_density"),
    # waterways (water_density) intentionally omitted — not in the final variable set.
]


def _resolve_gpkg_url(dataset: str) -> tuple[str, str]:
    """Find the GeoPackage download URL on the HDX dataset page.

    Returns (url, hdx_last_modified) so the caller can record which monthly
    HOTOSM snapshot was used (these layers refresh and the density columns
    drift between runs).
    """
    meta = requests.get(HDX_API + dataset, timeout=60).json()
    if not meta.get("success"):
        raise RuntimeError(f"HDX returned no success for {dataset}: {meta}")

    def _lm(res) -> str:
        return res.get("last_modified") or res.get("created") or ""

    for res in meta["result"]["resources"]:
        fmt = (res.get("format") or "").lower()
        url = res.get("url") or res.get("download_url") or ""
        if fmt in ("geopackage", "gpkg") and url.endswith(".gpkg"):
            return url, _lm(res)
        if url.endswith(".gpkg.zip"):
            return url, _lm(res)
    # Fall back to first shapefile if no GPKG.
    for res in meta["result"]["resources"]:
        if (res.get("format") or "").lower() in ("shp", "shapefile"):
            return res["url"], _lm(res)
    raise RuntimeError(f"No vector resource on HDX page for {dataset}.")


def _record_snapshot(dataset: str, url: str, last_modified: str, dest: Path) -> None:
    """Record this layer's HOTOSM snapshot provenance to a manifest.

    Writes raw_data/hotosm/hotosm_manifest.json with the resolved download URL,
    the HDX resource's last-modified date, and when/what we fetched — so the
    transport-density columns are reproducible despite the live HDX pull.
    """
    manifest = HOTOSM_DIR / "hotosm_manifest.json"
    entries: dict = {}
    if manifest.exists():
        try:
            entries = json.loads(manifest.read_text())
        except Exception:
            entries = {}
    entries[dataset] = {
        "dataset":           dataset,
        "url":               url,
        "hdx_last_modified": last_modified,
        "downloaded_at_utc": datetime.datetime.now(datetime.timezone.utc)
                                     .isoformat(timespec="seconds"),
        "local_file":        dest.name,
        "size_bytes":        dest.stat().st_size if dest.exists() else None,
    }
    manifest.write_text(json.dumps(entries, indent=2, ensure_ascii=False))


def _download(dataset: str) -> Path:
    """Cache the HOTOSM file locally and return its path; record its provenance."""
    HOTOSM_DIR.mkdir(parents=True, exist_ok=True)
    url, last_modified = _resolve_gpkg_url(dataset)
    ext = ".gpkg.zip" if url.endswith(".gpkg.zip") else Path(url).suffix
    dest = HOTOSM_DIR / f"{dataset}{ext}"
    if not (dest.exists() and dest.stat().st_size > 0):
        urllib.request.urlretrieve(url, dest)
    _record_snapshot(dataset, url, last_modified, dest)
    return dest


def _line_density(file_path: Path, gdf_munis) -> pd.Series:
    """Within-muni line meters per km² of muni area.

    HOTOSM zips bundle three layers per dataset: `*_points`, `*_polygons`,
    and `*_lines`. `gpd.read_file()` picks the first by default — usually the
    points layer — which has zero length and yields a meaningless density.
    Explicitly select the `*_lines` layer.

    Each line is **clipped to muni boundaries** (`overlay(how="intersection")`):
    a line that crosses several munis is split, so only the portion *inside* a
    muni counts toward that muni's length — no cross-boundary over-counting of
    long highways/rivers. (This is more accurate than a plain `intersects`
    spatial join, which would add the line's full length to every muni it
    touches.)

    The roads layer is 7.4M LineStrings; we read geometry-only once (~60 MB) and
    reproject + clip in 1M-row chunks so peak memory stays bounded. Clipping is
    heavier than a spatial join, so the roads layer takes noticeably longer.
    """
    import geopandas as gpd
    from pyogrio import list_layers

    src = f"zip://{file_path}" if str(file_path).endswith(".zip") else str(file_path)
    layer_names = [row[0] for row in list_layers(src)]
    line_layer = next((ln for ln in layer_names if ln.endswith("_lines")), None)
    if line_layer is None:
        raise RuntimeError(
            f"No `*_lines` layer in {file_path}; got layers {layer_names}. "
            "HOTOSM may have changed their GeoPackage schema."
        )

    # Geometry-only read: 7.4M roads → ~60 MB in RAM.
    lines = gpd.read_file(src, layer=line_layer, columns=[])
    munis_proj = gdf_munis[["code_muni", "geometry"]]

    chunk = 1_000_000
    total_m_per_muni: dict[str, float] = {}
    for start in range(0, len(lines), chunk):
        sub = lines.iloc[start:start + chunk]
        sub_proj = sub.to_crs("EPSG:5880")
        # Clip lines to munis: each output piece is the part of a line inside one
        # muni, so its length is counted only there. keep_geom_type drops the
        # point/degenerate intersections (a line merely touching a border).
        clipped = gpd.overlay(sub_proj, munis_proj, how="intersection",
                              keep_geom_type=True)
        clipped["length_m"] = clipped.geometry.length
        per_muni = clipped.groupby("code_muni")["length_m"].sum()
        for code, m in per_muni.items():
            total_m_per_muni[code] = total_m_per_muni.get(code, 0.0) + float(m)
        del sub_proj, clipped

    total_m = pd.Series(total_m_per_muni, name="length_m")
    area_km2 = pd.Series((gdf_munis.geometry.area / 1e6).values,
                         index=gdf_munis["code_muni"].values)
    area_km2.index.name = "code_muni"
    area_km2 = area_km2[~area_km2.index.duplicated(keep="first")]
    return (total_m.reindex(area_km2.index).fillna(0) / area_km2).rename("density")


def all_densities() -> pd.DataFrame:
    """Per-muni road_density, railway_density (m per km²), cached to data/intermediate/.

    The road layer is ~7.4M LineStrings and its clip/overlay dominates runtime,
    so the computed result is cached. Delete data/intermediate/transport_density.parquet
    to force a refresh against the current HOTOSM snapshot.
    """
    cache = CACHE / "transport_density.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    import geopandas as gpd
    from utils import MUNI_SHAPEFILE

    gdf = gpd.read_file(MUNI_SHAPEFILE).to_crs("EPSG:5880")
    key = "code_muni" if "code_muni" in gdf.columns else "CD_MUN"
    gdf["code_muni"] = normalize_code(gdf[key])

    out = pd.DataFrame({"code_muni": gdf["code_muni"].drop_duplicates().values})
    for dataset, col in LAYERS:
        path = _download(dataset)
        density = _line_density(path, gdf).rename(col).reset_index()
        out = out.merge(density, on="code_muni", how="left")
    out = out.fillna(0)

    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out
