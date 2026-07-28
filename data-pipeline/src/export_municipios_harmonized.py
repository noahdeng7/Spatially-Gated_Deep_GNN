"""
export_municipios_harmonized.py -- write the municipal geometry that MATCHES
the panel's src/dst codes.

The raw layer (data/raw/mgn2022/00mun.gpkg) is the December-2022 vintage with
2,475 municipalities. The panel is built on the census-2020 harmonized
universe: the crosswalk folds post-census children back into their parents,
which dissolves to 2,464 polygons. Mapping panel results onto the RAW layer
silently loses the six post-census municipalities plus the five census-window
splits -- so this export exists, and it verifies itself against the shipped
panel before writing.

Uses the same load -> crosswalk -> dissolve path as 03_distance.py, so the
geometry here cannot drift from what the pipeline actually used.

Output:
    data/processed/municipios_harmonized_2020.gpkg   (layer: municipios)
    data/processed/municipios_harmonized_2020_shp/   (Esri shapefile, for
        tools that cannot read GeoPackage; names longer than 10 chars are
        truncated by the format itself)

Run:
    python src/export_municipios_harmonized.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PipelineError, assert_cvegeo_valid, ensure_dirs, get_logger, load_config,
    rel, resolve, run_step,
)
from geo import apply_crosswalk, load_crosswalk, load_municipios  # noqa: E402

STEP = "export_municipios_harmonized"


def main() -> int:
    import pandas as pd

    cfg = load_config()
    ensure_dirs(cfg)
    log = get_logger(STEP, cfg)
    log.info("=" * 78)
    log.info("EXPORT harmonized municipal geometry (matches panel src/dst codes)")
    log.info("=" * 78)

    gdf = load_municipios(cfg, log)

    # Keep a name lookup from rows whose code is already final (the parents
    # exist in the raw layer too). A plain dissolve would take the FIRST name
    # in each group, which can be a child's name, not the parent's.
    names = None
    if "nomgeo" in gdf.columns:
        names = gdf[["cvegeo", "nomgeo"]].copy()

    xwalk = load_crosswalk(cfg, log)
    strategy = cfg["geometry"]["crosswalk_strategy"]
    gdf["cvegeo"] = apply_crosswalk(gdf["cvegeo"], xwalk, strategy, log, "geometry")
    if bool(gdf["cvegeo"].duplicated().any()):
        n_before = len(gdf)
        gdf = gdf.dissolve(by="cvegeo").reset_index()
        gdf["cvegeo"] = gdf["cvegeo"].astype("string")
        log.info("      dissolved split children into parents: %d -> %d features",
                 n_before, len(gdf))
    assert_cvegeo_valid(gdf["cvegeo"], "geometry.cvegeo")

    if names is not None:
        lookup = names[names["cvegeo"].isin(set(gdf["cvegeo"]))]
        lookup = lookup.drop_duplicates("cvegeo").set_index("cvegeo")["nomgeo"]
        gdf["nomgeo"] = gdf["cvegeo"].map(lookup)

    # --- verify against the panel actually shipped ---------------------------
    panel_path = resolve(cfg, cfg["assemble"]["output"])
    if panel_path.exists():
        src = set(pd.read_parquet(panel_path, columns=["src"])["src"].unique())
        geo = set(gdf["cvegeo"])
        if src != geo:
            raise PipelineError(
                f"geometry/panel mismatch: {len(geo - src)} code(s) only in "
                f"geometry {sorted(geo - src)[:5]}, {len(src - geo)} only in "
                f"panel {sorted(src - geo)[:5]}. Refusing to write a layer "
                "that will not join cleanly."
            )
        log.info("      VERIFIED: geometry codes == panel src universe "
                 "(%s municipios)", f"{len(geo):,}")
    else:
        log.warning("panel not found at %s -- skipping the join check",
                    rel(panel_path))

    out_dir = resolve(cfg, cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    gpkg = out_dir / "municipios_harmonized_2020.gpkg"
    gdf.to_file(gpkg, layer="municipios", driver="GPKG")
    log.info("WROTE %s  features=%s  size=%.1f MB",
             rel(gpkg), f"{len(gdf):,}", gpkg.stat().st_size / 1e6)

    shp_dir = out_dir / "municipios_harmonized_2020_shp"
    shp_dir.mkdir(exist_ok=True)
    shp = shp_dir / "municipios_harmonized_2020.shp"
    gdf.to_file(shp)
    log.info("WROTE %s (+ sidecars)", rel(shp))
    log.info("      shapefile caveats: 10-char field names, dbf encoding; "
             "prefer the GeoPackage where possible")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_step(main, STEP))
