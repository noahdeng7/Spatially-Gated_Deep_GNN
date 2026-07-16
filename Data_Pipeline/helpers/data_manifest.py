"""Single source of truth for every raw input the pipeline stages into raw_data/.

`download_data.py` iterates this manifest to fetch and verify each dataset. The
data files themselves are gitignored (large / derived), but THIS manifest IS
committed — so `python download_data.py` reproducibly regenerates raw_data/ from
pinned sources, with SHA-256 verification to catch upstream drift.

Each `Dataset` declares one file (or, for `geobr`/`inpe`, one logical dataset
fetched via a package/helper). `sha256=None` means the bytes can't be pinned
(dynamically generated, or upstream re-encodes on every request) — those get a
presence check only; pinned entries fail loudly on mismatch.

Not in the manifest, by design:
  * HOTOSM roads/rail/water — the download URL is resolved through the HDX API at
    run time (`features_transport._download`), so there is no static URL to pin.
    `download_data.download_hotosm()` stages these; the versions float with HDX.
  * Google Earth Engine (ERA5 climate + MODIS NDVI) — requires interactive auth,
    so it can't be staged unattended. Ship the cached intermediate/*.parquet
    instead. See `download_data.show_manual_setup()`.

Fields:
  id        stable slug (unique)
  group     which download_data.py section stages it / `--only` selector
  method    http | gdrive | geobr | inpe
  dest      path relative to utils.RAW
  url       for method in {http, inpe} — the source (inpe: the per-year zip)
  gdrive_id for method=gdrive
  geobr     {"year", "driver"} for method=geobr
  sha256    pinned lowercase hex digest, or None (presence-check only)
  required  True if a missing file degrades the FINAL dataset (informational)
  note      human context
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    id: str
    group: str
    method: str
    dest: str
    url: str | None = None
    gdrive_id: str | None = None
    geobr: dict | None = None
    sha256: str | None = None
    required: bool = False
    note: str = ""


# --- Pinned source coordinates ------------------------------------------------

# censobr GitHub release — immutable release assets (safe to pin a version tag).
CENSOBR_RELEASE = "v0.5.0"
_CENSOBR_BASE = (
    f"https://github.com/ipeaGIT/censobr/releases/download/{CENSOBR_RELEASE}"
)

# MapBiomas Collection 10.1 coverage (biomes/states/municipalities), 1985-2024,
# area in hectares per land-cover class. Hosted on Google Drive off the portal
# https://brasil.mapbiomas.org/en/estatisticas/ . The file is stable within a
# collection, so we pin both the Drive id AND the SHA-256 — a mismatch means
# MapBiomas re-released and the land_use numbers may shift.
MAPBIOMAS_COVERAGE_GDRIVE_ID = "1p0dLrhvKymPhrSithsRBMRgHlFHq0FUf"
MAPBIOMAS_COVERAGE_SHA256 = (
    "96504305146d55d48c316887aaa65c66e89b08d1c54db58442511c75aad5594d"
)

# INPE BDQueimadas active-fire focos — one all-Brazil zip per year. Fetched by
# features_biomas._download_inpe_focos, which extracts the CSV. The 2006-2010
# window matches features_biomas.MAPBIOMAS_YEARS (the fire_pct averaging window).
_INPE_FOCOS_URL = (
    "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/anual/"
    "Brasil_sat_ref/focos_br_ref_{year}.zip"
)
FIRE_YEARS = list(range(2006, 2011))  # 2006..2010 inclusive

# IBGE SIDRA crop pivot tables — table 1612 (temporary) + 1613 (permanent).
# The HTML export is regenerated per request (no stable byte checksum), so these
# are presence-checked only; the query string pins table/variables/years.
_SIDRA_TEMPORARY = (
    "https://sidra.ibge.gov.br/geratabela?format=us.csv&name=tabela1612.csv"
    "&terr=NCN&rank=-&query=t/1612/n3/all/v/109,214,215,216,112/p/2005,2006,2007,2008,2009,2010"
    "/c81/allxt/d/v109%200,v214%200,v215%200,v216%200,v112%200/l/v,p%2Bc81,t"
)
_SIDRA_PERMANENT = (
    "https://sidra.ibge.gov.br/geratabela?format=us.csv&name=tabela1613.csv"
    "&terr=NCN&rank=-&query=t/1613/n3/all/v/2313,214,215,216,112/p/2005,2006,2007,2008,2009,2010"
    "/c82/allxt/d/v2313%200,v214%200,v215%200,v216%200,v112%200/l/v,p%2Bc82,t"
)


# --- The manifest -------------------------------------------------------------
MANIFEST: list[Dataset] = [
    # Census (feeds every census_* feature).
    Dataset("censobr_population_2010", "censobr", "http",
            "censobr/population_2010.parquet",
            url=f"{_CENSOBR_BASE}/2010_population_{CENSOBR_RELEASE}.parquet",
            required=True, note="2010 person parquet (censobr release)"),
    Dataset("censobr_household_2010", "censobr", "http",
            "censobr/household_2010.parquet",
            url=f"{_CENSOBR_BASE}/2010_households_{CENSOBR_RELEASE}.parquet",
            required=True, note="2010 household parquet (censobr release)"),

    # Municipality geometry (join key + areas for land_use/fire/transport/etc).
    # geobr writes the file locally, so the bytes aren't reproducible — pinned by
    # year instead (the 2010 vintage: 5,565 munis matching censobr 2010).
    Dataset("shapefile_shp", "shapefile", "geobr",
            "shapefile/BR_Municipios_2010.shp",
            geobr={"year": 2010, "driver": "ESRI Shapefile"},
            required=True, note="IBGE 2010 muni polygons via geobr"),
    Dataset("shapefile_gpkg", "shapefile", "geobr",
            "shapefile/br_municipios_2010.gpkg",
            geobr={"year": 2010, "driver": "GPKG"},
            note="same munis, GeoPackage form"),

    # IBGE crop economics (crop_pct / crop_economics).
    Dataset("ibge_crops_temporary", "ibge_crops", "http",
            "ibge_crops/temporary_crops_2005_2010.csv",
            url=_SIDRA_TEMPORARY, note="SIDRA table 1612 (temporary crops)"),
    Dataset("ibge_crops_permanent", "ibge_crops", "http",
            "ibge_crops/permanent_crops_2005_2010.csv",
            url=_SIDRA_PERMANENT, note="SIDRA table 1613 (permanent crops)"),

    # MapBiomas land use — the previously-manual step, now automated + pinned.
    Dataset("mapbiomas_coverage", "mapbiomas", "gdrive",
            "mapbiomas/MAPBIOMAS_BRAZIL-COVERAGE_STATISTICS-COL.10.1-MUNICIPALITIES_STATES_BIOMES.xlsx",
            gdrive_id=MAPBIOMAS_COVERAGE_GDRIVE_ID,
            sha256=MAPBIOMAS_COVERAGE_SHA256,
            required=True,
            note="MapBiomas Collection 10.1 coverage (land_use); ~76 MB"),

    # INPE fire focos, pre-staged for the 2006-2010 fire_pct window.
    *[Dataset(f"fire_focos_{y}", "fire", "inpe",
              f"fire/focos_br_ref_{y}.csv",
              url=_INPE_FOCOS_URL.format(year=y),
              note=f"INPE BDQueimadas focos {y} (fire_pct)")
      for y in FIRE_YEARS],
]

# Groups in the order download_data.py stages them ("hotosm" is delegated to
# features_transport and "manual" prints setup notes — neither is in MANIFEST).
GROUPS = ["censobr", "shapefile", "ibge_crops", "mapbiomas", "fire", "hotosm", "manual"]


def by_group(group: str) -> list[Dataset]:
    return [d for d in MANIFEST if d.group == group]


def pinned() -> list[Dataset]:
    """Datasets with a SHA-256 to verify (the reproducibility-critical ones)."""
    return [d for d in MANIFEST if d.sha256]
