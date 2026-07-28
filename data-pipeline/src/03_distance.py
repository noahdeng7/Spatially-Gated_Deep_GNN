"""
03_distance.py -- population-weighted centroids and geodesic O-D distance.

Input  : municipal geometry (Marco Geoestadistico 2020)
         WorldPop 2020 Mexico 100m gridded population
Output : interim/centroids.parquet        one row per municipality
         interim/distance.parquet         one row per ordered O-D pair
         reports/centroid_containment.csv  centroids outside their own polygon

WHY POPULATION-WEIGHTED CENTROIDS
---------------------------------
A geometric centroid answers "where is the middle of this shape". A gravity
model wants "where are the people". In the sparse northern municipalities those
are wildly different places: Ensenada is ~52,000 km2 with its population pinned
to the northwest corner, and its geometric centroid sits in empty desert
~150 km from where anyone lives. Using geometric centroids there does not add
noise, it adds *bias*, and it biases the distance decay term specifically in the
municipalities with the largest out-migration.

WHY GEODESIC, NOT PROJECTED EUCLIDEAN
-------------------------------------
Mexico spans ~30 degrees of longitude and ~17 of latitude. No single projection
holds distance accurate across that extent; a Euclidean distance in any one of
them accumulates several percent error at the corners, systematically. Geodesic
distance on the WGS84 ellipsoid (Karney's algorithm, via pyproj.Geod) is exact
to sub-millimetre and has no projection choice to defend.

Idempotent. The centroid pass is the slow part; it is cached to
interim/centroids.parquet and skipped on re-run unless --recompute-centroids.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PipelineError, ROOT, assert_cvegeo_valid, ensure_dirs, get_logger,
    load_config, log_rows, report_path, rel, resolve, run_step, write_interim,
)
from geo import apply_crosswalk, assert_geometry_vintage, load_crosswalk, load_municipios  # noqa: E402

STEP = "03_distance"

# WGS84 ellipsoid parameters, used by the geodesic solver.
WGS84 = dict(ellps="WGS84")


# ---------------------------------------------------------------------------
# the distance function (pure, testable -- see tests/test_distance.py)
# ---------------------------------------------------------------------------

def geodesic_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Geodesic distance in kilometres on the WGS84 ellipsoid.

    Vectorised over numpy arrays or scalars. Uses pyproj.Geod, which implements
    Karney's algorithm -- accurate to nanometres, and correct at antipodes where
    the haversine formula degrades and the spherical law of cosines collapses.

    Args are in decimal degrees. Returns kilometres.

        >>> round(float(geodesic_km(0.0, 0.0, 0.0, 1.0)), 3)
        111.319
    """
    from pyproj import Geod

    g = Geod(**WGS84)
    lat1 = np.asarray(lat1, dtype="float64")
    lon1 = np.asarray(lon1, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")

    # pyproj takes (lon, lat) order -- transposing these is the classic bug, and
    # it produces plausible-looking wrong numbers rather than an error.
    _, _, dist_m = g.inv(lon1, lat1, lon2, lat2)
    return np.asarray(dist_m, dtype="float64") / 1000.0


def great_circle_km(lat1, lon1, lat2, lon2, radius_km: float = 6371.0088) -> np.ndarray:
    """
    Great-circle (spherical) distance in km, via the haversine formula.

    Retained alongside the geodesic measure because the gravity literature is
    split on which it uses, and the difference (up to ~0.3% at Mexico's
    latitudes) is large enough to matter when comparing coefficients against a
    published paper. Default radius is the IUGG mean Earth radius.
    """
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype="float64"))
                              for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def internal_distance_km(area_km2, coefficient: float = 0.67) -> np.ndarray:
    """
    Head-Mayer internal ("self") distance: coefficient * sqrt(area / pi).

    This is the average distance from a random point in a disc of equal area to
    its centre, scaled by the conventional 0.67. Used for the diagonal, which is
    excluded from the main panel but written to interim/ in case it is wanted
    later (e.g. for a home-bias specification).

        >>> round(float(internal_distance_km(math.pi * 100.0)), 4)   # r = 10 km
        6.7
    """
    area = np.asarray(area_km2, dtype="float64")
    if np.any(area < 0):
        raise ValueError("negative area passed to internal_distance_km")
    return coefficient * np.sqrt(area / math.pi)


# ---------------------------------------------------------------------------
# population-weighted centroids
# ---------------------------------------------------------------------------

def _check_raster_crs(src, expected: str, log, what: str) -> None:
    """
    Verify a raster's CRS explicitly before any zonal operation.

    An unlabelled or mismatched CRS is the single most common cause of zonal
    statistics that are quietly wrong -- the numbers come out plausible, just
    for the wrong places.
    """
    if src.crs is None:
        raise PipelineError(
            f"{what} has NO CRS defined. Refusing to assume one. "
            "Zonal statistics against an unlabelled raster silently sample the "
            "wrong cells."
        )
    log.info("      %s CRS = %s  shape=%s  res=%s",
             what, src.crs.to_string(), src.shape,
             tuple(round(r, 8) for r in src.res))
    if src.crs.to_string() != expected:
        log.warning("      %s CRS (%s) != storage CRS (%s); geometries will be "
                    "reprojected to the raster before masking",
                    what, src.crs.to_string(), expected)


def population_weighted_centroids(cfg, gdf, log) -> pd.DataFrame:
    """
    Compute the population-weighted centroid of every municipality.

    For each polygon: mask the WorldPop grid to it, take the cell centre
    coordinates and the cell population values, and compute the weighted mean
    coordinate.

    Falls back to the geometric representative point where a polygon contains no
    population at all (a handful of tiny or entirely-nodata units). Every
    fallback is logged and flagged in the output, never silent.
    """
    import rasterio
    from rasterio.mask import mask as rio_mask

    raster_path = resolve(cfg, cfg["distance"]["worldpop_raster"])
    if not raster_path.exists():
        raise PipelineError(
            f"WorldPop raster not found: {raster_path}\n"
            "Run `python src/00_download.py --only worldpop_2020` "
            "(fill in the URL first). Population-weighted centroids are the whole "
            "point of this step -- falling back to geometric centroids would "
            "reintroduce exactly the bias this step exists to remove."
        )

    storage_crs = cfg["crs"]["storage"]
    rows = []

    with rasterio.open(raster_path) as src:
        _check_raster_crs(src, storage_crs, log, "WorldPop 2020")
        nodata = src.nodata

        work = gdf.to_crs(src.crs) if gdf.crs.to_string() != src.crs.to_string() else gdf
        n = len(work)

        for i, rec in enumerate(work.itertuples(index=False)):
            geom = rec.geometry
            cvegeo = rec.cvegeo
            try:
                arr, transform = rio_mask(src, [geom], crop=True, filled=True,
                                          nodata=np.nan, all_touched=False)
            except ValueError:
                # polygon does not overlap the raster at all
                arr, transform = None, None

            lon = lat = np.nan
            pop_sum = 0.0
            method = "population_weighted"

            if arr is not None:
                band = arr[0].astype("float64")
                if nodata is not None and not math.isnan(float(nodata)):
                    band = np.where(band == nodata, np.nan, band)
                band = np.where(band < 0, np.nan, band)   # WorldPop uses -9999
                valid = np.isfinite(band) & (band > 0)
                pop_sum = float(np.nansum(band[valid])) if valid.any() else 0.0

                if valid.any() and pop_sum > 0:
                    r_idx, c_idx = np.nonzero(valid)
                    # cell CENTRES, not corners: the +0.5 offset matters at 100m
                    xs, ys = rasterio.transform.xy(transform, r_idx, c_idx, offset="center")
                    w = band[r_idx, c_idx]
                    lon = float(np.average(np.asarray(xs), weights=w))
                    lat = float(np.average(np.asarray(ys), weights=w))

            if not math.isfinite(lon) or not math.isfinite(lat):
                pt = geom.representative_point()
                lon, lat = float(pt.x), float(pt.y)
                method = "representative_point_fallback"
                log.warning("      %s: no population in polygon; fell back to "
                            "representative point", cvegeo)

            rows.append(dict(cvegeo=cvegeo, lon=lon, lat=lat,
                             pop_raster_sum=pop_sum, centroid_method=method))

            if (i + 1) % 250 == 0 or i == n - 1:
                log.info("  centroids %d/%d", i + 1, n)

        cent = pd.DataFrame(rows)

        # Reproject centroids back to storage CRS if the raster was in another.
        if src.crs.to_string() != storage_crs:
            from pyproj import Transformer
            tr = Transformer.from_crs(src.crs, storage_crs, always_xy=True)
            lons, lats = tr.transform(cent["lon"].to_numpy(), cent["lat"].to_numpy())
            cent["lon"], cent["lat"] = lons, lats

    n_fallback = int((cent["centroid_method"] != "population_weighted").sum())
    log.info("      %d/%d centroids population-weighted, %d fallback",
             len(cent) - n_fallback, len(cent), n_fallback)
    return cent


def check_containment(cfg, cent: pd.DataFrame, gdf, log) -> pd.DataFrame:
    """
    Verify each population-weighted centroid falls inside its own polygon.

    It legitimately may not: a crescent-shaped or multipart municipality with
    population clustered at both ends puts the weighted mean in the gap. That is
    not a bug, but it IS something the distance measure should not silently
    absorb, so every case is reported and (per config) snapped to the nearest
    valid interior point.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    if not cfg["distance"].get("centroid_containment_check", True):
        log.info("      containment check disabled in config")
        return cent

    merged = gdf[["cvegeo", "geometry"]].merge(cent, on="cvegeo", how="left")
    pts = gpd.GeoSeries([Point(xy) for xy in zip(merged["lon"], merged["lat"])],
                        crs=cfg["crs"]["storage"])
    # shapely.within is element-wise on plain arrays; GeoSeries.within would
    # align on index (or, in geopandas >= 1.1, reject a raw GeometryArray).
    import shapely
    inside = shapely.within(pts.values, merged["geometry"].values)
    merged["centroid_inside_polygon"] = inside

    outside = merged[~merged["centroid_inside_polygon"]]
    log.info("      containment: %d/%d centroids inside their own polygon",
             int(merged["centroid_inside_polygon"].sum()), len(merged))

    if len(outside):
        log.warning("      %d centroid(s) fall OUTSIDE their polygon "
                    "(concave/multipart geometry): %s",
                    len(outside), outside["cvegeo"].head(15).tolist())
        p = report_path(cfg, "centroid_containment.csv")
        outside[["cvegeo", "lon", "lat", "pop_raster_sum", "centroid_method"]].to_csv(
            p, index=False)
        log.warning("      flagged -> %s", rel(p))

        if cfg["distance"].get("centroid_fallback") == "representative_point":
            idx = merged.index[~merged["centroid_inside_polygon"]]
            reps = merged.loc[idx, "geometry"].representative_point()
            merged.loc[idx, "lon"] = reps.x.to_numpy()
            merged.loc[idx, "lat"] = reps.y.to_numpy()
            merged.loc[idx, "centroid_method"] = "representative_point_containment"
            log.warning("      snapped %d centroid(s) to representative point "
                        "(config: distance.centroid_fallback)", len(idx))

    return merged.drop(columns="geometry")


def compute_areas(cfg, gdf, log) -> pd.DataFrame:
    """Municipal area in km2, computed in an equal-area projection."""
    ea = cfg["crs"]["equal_area"]
    log.info("      computing areas in %s (equal-area; areas from a geographic "
             "CRS would be meaningless)", ea)
    proj = gdf.to_crs(ea)
    return pd.DataFrame({
        "cvegeo": gdf["cvegeo"].to_numpy(),
        "area_km2": proj.geometry.area.to_numpy() / 1e6,
    })


# ---------------------------------------------------------------------------
# pairwise distance
# ---------------------------------------------------------------------------

def build_distance_matrix(cent: pd.DataFrame, cfg, log) -> pd.DataFrame:
    """
    Full ordered-pair distance table, including the diagonal.

    Built in blocks so peak memory stays bounded: 2,470^2 pairs x 8 bytes x a
    few columns is manageable, but materialising the cartesian product as an
    object-dtype frame in one shot is not.
    """
    cent = cent.sort_values("cvegeo", ignore_index=True)
    codes = cent["cvegeo"].to_numpy()
    lat = cent["lat"].to_numpy(dtype="float64")
    lon = cent["lon"].to_numpy(dtype="float64")
    area = cent["area_km2"].to_numpy(dtype="float64")
    n = len(cent)
    log.info("      building %d x %d = %s ordered pairs", n, n, f"{n * n:,}")

    coeff = float(cfg["distance"]["internal_distance_coefficient"])
    internal = internal_distance_km(area, coeff)

    blocks = []
    block = max(1, 4_000_000 // max(n, 1))
    for start in range(0, n, block):
        stop = min(start + block, n)
        m = stop - start

        o_idx = np.repeat(np.arange(start, stop), n)
        d_idx = np.tile(np.arange(n), m)

        geo = geodesic_km(lat[o_idx], lon[o_idx], lat[d_idx], lon[d_idx])
        gc = great_circle_km(lat[o_idx], lon[o_idx], lat[d_idx], lon[d_idx])

        # Diagonal: replace the (zero) self-distance with Head-Mayer internal.
        same = o_idx == d_idx
        geo = np.where(same, internal[o_idx], geo)
        gc = np.where(same, internal[o_idx], gc)

        blocks.append(pd.DataFrame({
            "orig": pd.array(codes[o_idx], dtype="string"),
            "dest": pd.array(codes[d_idx], dtype="string"),
            "dist_geodesic_km": geo,
            "dist_greatcircle_km": gc,
            "is_diagonal": same,
        }))
        log.info("  distance block %d-%d / %d", start, stop, n)

    out = pd.concat(blocks, ignore_index=True)
    log.info("      distance: n=%s  min=%.3f  median=%.1f  max=%.1f km",
             f"{len(out):,}", out["dist_geodesic_km"].min(),
             out["dist_geodesic_km"].median(), out["dist_geodesic_km"].max())

    off = out.loc[~out["is_diagonal"], "dist_geodesic_km"]
    max_dev = float((out.loc[~out["is_diagonal"], "dist_greatcircle_km"] - off).abs().max())
    log.info("      max |great-circle - geodesic| off-diagonal = %.3f km "
             "(both retained)", max_dev)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recompute-centroids", action="store_true",
                    help="ignore the cached centroid table and redo the raster pass")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(STEP, cfg)
    log.info("=" * 78)
    log.info("STEP %s", STEP)
    log.info("=" * 78)

    gdf = load_municipios(cfg, log)
    assert_geometry_vintage(cfg, gdf, log)
    n_geo_in = len(gdf)

    # Harmonize geometry to the same municipal universe as the flows, by
    # dissolving split children into their parent.
    xwalk = load_crosswalk(cfg, log)
    strategy = cfg["geometry"]["crosswalk_strategy"]
    gdf["cvegeo"] = apply_crosswalk(gdf["cvegeo"], xwalk, strategy, log, "geometry")
    if bool(gdf["cvegeo"].duplicated().any()):
        n_before = len(gdf)
        gdf = gdf.dissolve(by="cvegeo").reset_index()
        gdf["cvegeo"] = gdf["cvegeo"].astype("string")
        log_rows(log, "dissolve split children into parent", n_before, len(gdf))
    assert_cvegeo_valid(gdf["cvegeo"], "geometry.cvegeo")

    cache = resolve(cfg, cfg["paths"]["interim"]) / "centroids.parquet"
    if cache.exists() and not args.recompute_centroids:
        cent = pd.read_parquet(cache)
        cent["cvegeo"] = cent["cvegeo"].astype("string")
        log.info("READ  cached centroids (%s rows). Use --recompute-centroids to "
                 "redo the raster pass.", f"{len(cent):,}")
    else:
        cent = population_weighted_centroids(cfg, gdf, log)
        cent = check_containment(cfg, cent, gdf, log)
        cent = cent.merge(compute_areas(cfg, gdf, log), on="cvegeo", how="left")
        cent["internal_dist_km"] = internal_distance_km(
            cent["area_km2"], float(cfg["distance"]["internal_distance_coefficient"]))
        write_interim(cfg, cent, "centroids", log)

    if "area_km2" not in cent.columns:
        cent = cent.merge(compute_areas(cfg, gdf, log), on="cvegeo", how="left")

    # --- sanity: centroids must be inside Mexico's bounding box --------------
    lon_ok = cent["lon"].between(-118.5, -86.0)
    lat_ok = cent["lat"].between(14.0, 33.0)
    bad = ~(lon_ok & lat_ok)
    if bool(bad.any()):
        raise PipelineError(
            f"{int(bad.sum())} centroid(s) fall outside Mexico's bounding box, "
            f"e.g. {cent.loc[bad, ['cvegeo', 'lon', 'lat']].head().to_dict('records')}. "
            "This is the signature of a lon/lat transposition or a CRS mismatch."
        )
    log.info("      all %s centroids inside Mexico bbox", f"{len(cent):,}")

    dist = build_distance_matrix(cent, cfg, log)
    write_interim(cfg, dist, "distance", log)

    log_rows(log, "STEP TOTAL 03_distance", n_geo_in, len(dist))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_step(main, STEP))



