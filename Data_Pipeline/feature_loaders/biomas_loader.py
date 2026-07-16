"""Land-use (MapBiomas) + fire (INPE) per municipality.

Public functions:
  land_use_pct_per_muni(years) → deforestation_pct, mining_pct, pasture_pct, urban_area_pct
  fire_pct_per_muni(years)     → fire_pct

Both default to a 2006-2010 average (MAPBIOMAS_YEARS): each output column is the
mean of its per-year values across those five years, which smooths single-year
noise. Pass a single int (e.g. 2010) or a shorter list to change the window.
Because every output is a percent of the (constant) muni area, averaging the
yearly percents is identical to taking the percent of the averaged area.

Land-use input (manually downloaded from https://brasil.mapbiomas.org/en/estatisticas/):
  raw_data/mapbiomas/MAPBIOMAS_BRAZIL-COVERAGE_STATISTICS-COL.10.1-MUNICIPALITIES_STATES_BIOMES.xlsx
      "Land cover and use - Biome and Political Division" → "Municipality" download.
      Provides per-muni per-year per-class areas in hectares (sheet "COVERAGE_10.1").

Fire input (auto-downloaded per year): INPE BDQueimadas active-fire focos — one
all-Brazil zip per year (see _download_inpe_focos). No MapBiomas Fogo file is used.

Class codes used (MapBiomas legend, https://brasil.mapbiomas.org/codigos-de-legenda):
  3  = Forest formation   (deforestation_pct = remaining coverage % [ref-compatible])
  15 = Pasture            → pasture_pct
  24 = Urban Area         → urban_area_pct
  30 = Mining             → mining_pct
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import RAW, normalize_code, INTERMEDIATE

MAPBIOMAS_DIR = RAW / "mapbiomas"
COVERAGE_FILE = MAPBIOMAS_DIR / "MAPBIOMAS_BRAZIL-COVERAGE_STATISTICS-COL.10.1-MUNICIPALITIES_STATES_BIOMES.xlsx"
CACHE = INTERMEDIATE   # data/intermediate — shared with the pipeline's checkpoints

CLASS_PASTURE    = 15
CLASS_URBAN      = 24
CLASS_MINING     = 30
CLASS_FOREST     = 3

# Default averaging window for both land-use and fire: 2006-2010 inclusive.
MAPBIOMAS_YEARS = list(range(2006, 2011))


def _as_year_list(years: list[int] | int | None) -> list[int]:
    """Normalize the `years` argument to a list (None → MAPBIOMAS_YEARS, int → [int])."""
    if years is None:
        return list(MAPBIOMAS_YEARS)
    if isinstance(years, int):
        return [years]
    return list(years)


def _cache_key(years: list[int]) -> str:
    return "-".join(str(y) for y in years)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"MapBiomas file missing: {path}\n"
            "  See raw_data/mapbiomas/README.md for download instructions."
        )
    return path


# ---------------------------------------------------------------------------
# Muni name → IBGE 7-digit code lookup (the COVERAGE sheet has no muni code,
# only (state_acronym, municipality) strings).
# ---------------------------------------------------------------------------
def _muni_name_to_code() -> dict[tuple[str, str], str]:
    import geopandas as gpd
    from utils import MUNI_SHAPEFILE
    gdf = gpd.read_file(MUNI_SHAPEFILE)
    # Tolerate either geobr ("name_muni", "abbrev_state") or IBGE ("NM_MUN", "SIGLA_UF") schemas.
    name_col = next((c for c in ("name_muni", "NM_MUN", "nm_mun") if c in gdf.columns), None)
    uf_col   = next((c for c in ("abbrev_state", "abbrev_sta", "SIGLA_UF", "sigla_uf", "SIGLA") if c in gdf.columns), None)
    code_col = next((c for c in ("code_muni", "CD_MUN") if c in gdf.columns), None)
    if not (name_col and uf_col and code_col):
        raise RuntimeError(f"Cannot find name+UF+code columns in {MUNI_SHAPEFILE}; got {gdf.columns.tolist()[:10]}")
    out: dict[tuple[str, str], str] = {}
    for _, r in gdf[[name_col, uf_col, code_col]].iterrows():
        key = (str(r[uf_col]).strip().upper(), str(r[name_col]).strip().lower())
        out[key] = normalize_code(pd.Series([r[code_col]])).iloc[0]
    return out


def _muni_area_km2() -> pd.Series:
    """Per-muni area (km²) indexed by code_muni, from utils.load_areas (2010 gpkg)."""
    from utils import load_areas, MUNI_GPKG
    areas = load_areas(MUNI_GPKG)
    area = pd.Series(areas["area_km2"].values, index=normalize_code(areas["code_muni"]).values)
    area = area[~area.index.duplicated(keep="first")]
    area.index.name = "code_muni"   # so class_pct's `x / area` keeps code_muni on reset_index
    return area


# ---------------------------------------------------------------------------
# Land use
# ---------------------------------------------------------------------------
def land_use_pct_per_muni(years: list[int] | int | None = None) -> pd.DataFrame:
    years = _as_year_list(years)
    cache = CACHE / f"biomas_landuse_{_cache_key(years)}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    _require(COVERAGE_FILE)
    df = pd.read_excel(COVERAGE_FILE, sheet_name="COVERAGE_10.1")

    # Map (state_acronym, municipality) → code_muni.
    lookup = _muni_name_to_code()
    keys = list(zip(df["state_acronym"].astype(str).str.strip().str.upper(),
                    df["municipality"].astype(str).str.strip().str.lower()))
    df["code_muni"] = [lookup.get(k) for k in keys]
    df = df.dropna(subset=["code_muni"])

    area_km2 = _muni_area_km2()
    out = pd.DataFrame({"code_muni": area_km2.index})

    def class_pct(class_id: int) -> pd.Series:
        # Sum each year's class area per muni, then average the area across
        # `years`. Percent-of-constant-area ⇒ mean(yearly pct) == pct(mean area).
        sub = df[df["class_id"] == class_id].groupby("code_muni")[years].sum()
        avg_ha = sub.mean(axis=1)
        # MapBiomas values are in hectares; muni area is km². 1 ha = 0.01 km².
        return (100.0 * (avg_ha * 0.01) / area_km2).rename(f"class_{class_id}_pct")

    out = out.merge(class_pct(CLASS_PASTURE).rename("pasture_pct").reset_index(),    on="code_muni", how="left")
    out = out.merge(class_pct(CLASS_URBAN).rename("urban_area_pct").reset_index(),   on="code_muni", how="left")
    out = out.merge(class_pct(CLASS_MINING).rename("mining_pct").reset_index(),      on="code_muni", how="left")

    # deforestation_pct — REFERENCE-COMPATIBLE. The reference's "Deforestation_pct"
    # is actually the remaining forest-formation COVERAGE % (~21% mean): a stock of
    # surviving forest, despite the misleading name. We reproduce it verbatim (now
    # as the `years` mean) so the column matches the reference dataset.
    forest_y = class_pct(CLASS_FOREST)
    out = out.merge(forest_y.rename("deforestation_pct").reset_index(),
                    on="code_muni", how="left")

    # Imputation step: MapBiomas occasionally classifies the same pixel into
    # multiple land-use categories across time slices within the same year.
    # When that happens, our muni-level Σ(class_area) can exceed muni area,
    # producing a tiny number of `pasture_pct` values > 100 (max observed:
    # 151.06 on 3 munis). Cap to the mathematical bound of 100 so downstream
    # consumers can trust the column as a true percent.
    pct_cols = ["deforestation_pct", "mining_pct", "pasture_pct", "urban_area_pct"]
    out = out[["code_muni"] + pct_cols].fillna(0)
    for c in pct_cols:
        out[c] = out[c].clip(lower=0.0, upper=100.0)

    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


# ---------------------------------------------------------------------------
# Fire — INPE BDQueimadas active-fire-detection focos.
#
# Counts hotspot detections per muni, approximates fire area as count ×
# 0.01 km² (each MODIS hotspot covers roughly 1 ha), then divides by muni area.
#
# Source: INPE open data — one zip per year for all of Brazil:
#   https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/anual/
#     Brasil_sat_ref/focos_br_ref_{year}.zip
# Schema: id_bdq,foco_id,lat,lon,data_pas,pais,estado,municipio,bioma
#   estado    = uppercase Portuguese state name (e.g., "ACRE", "SÃO PAULO")
#   municipio = uppercase Portuguese muni name (e.g., "ACRELÂNDIA")
# ---------------------------------------------------------------------------
INPE_FOCOS_URL = (
    "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/anual/"
    "Brasil_sat_ref/focos_br_ref_{year}.zip"
)
INPE_FIRE_DIR = RAW / "fire"

# Full-name → 2-letter acronym (INPE uses uppercase Portuguese with accents).
UF_NAME_TO_ACRONYM_UPPER = {
    "ACRE": "AC", "ALAGOAS": "AL", "AMAPÁ": "AP", "AMAZONAS": "AM", "BAHIA": "BA",
    "CEARÁ": "CE", "DISTRITO FEDERAL": "DF", "ESPÍRITO SANTO": "ES", "GOIÁS": "GO",
    "MARANHÃO": "MA", "MATO GROSSO": "MT", "MATO GROSSO DO SUL": "MS",
    "MINAS GERAIS": "MG", "PARÁ": "PA", "PARAÍBA": "PB", "PARANÁ": "PR",
    "PERNAMBUCO": "PE", "PIAUÍ": "PI", "RIO DE JANEIRO": "RJ",
    "RIO GRANDE DO NORTE": "RN", "RIO GRANDE DO SUL": "RS", "RONDÔNIA": "RO",
    "RORAIMA": "RR", "SANTA CATARINA": "SC", "SÃO PAULO": "SP",
    "SERGIPE": "SE", "TOCANTINS": "TO",
}


def _download_inpe_focos(year: int) -> Path:
    """Download INPE all-Brazil focos zip for `year` if not cached. Returns CSV path."""
    import urllib.request
    import zipfile

    INPE_FIRE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = INPE_FIRE_DIR / f"focos_br_ref_{year}.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return csv_path

    zip_path = INPE_FIRE_DIR / f"focos_br_ref_{year}.zip"
    if not (zip_path.exists() and zip_path.stat().st_size > 0):
        url = INPE_FOCOS_URL.format(year=year)
        print(f"  INPE focos {year}: downloading {url}")
        with urllib.request.urlopen(url, timeout=180) as r, open(zip_path, "wb") as f:
            f.write(r.read())

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(INPE_FIRE_DIR)
    return csv_path


def _focos_counts(year: int) -> pd.Series:
    """Per-(state_acronym, muni_key) INPE hotspot count for `year`.

    Downloads/caches the INPE focos CSV and normalizes the join keys (state full
    name → acronym; muni name → trimmed lowercase). Returns a Series named after
    the year so several years concat cleanly for averaging.
    """
    csv_path = _download_inpe_focos(year)
    focos = pd.read_csv(csv_path, skipinitialspace=True, low_memory=False)
    focos["state_acronym"] = (focos["estado"].astype(str).str.strip().str.upper()
                                              .map(UF_NAME_TO_ACRONYM_UPPER))
    focos["muni_key"]      = focos["municipio"].astype(str).str.strip().str.lower()
    focos = focos.dropna(subset=["state_acronym", "muni_key"])
    return focos.groupby(["state_acronym", "muni_key"]).size().rename(str(year))


def fire_pct_per_muni(years: list[int] | int | None = None) -> pd.DataFrame:
    """Per-muni `fire_pct` = (avg #hotspots × 0.01 km²) / muni_area_km² × 100.

    Hotspot counts come from INPE BDQueimadas focos (MODIS Aqua reference
    satellite), averaged across `years` (default 2006-2010) — one focos zip is
    downloaded per year. Joined to munis by (estado, municipio) name → IBGE
    7-digit code via the 2010 shapefile. Result cached to data/intermediate/.
    """
    years = _as_year_list(years)
    cache = CACHE / f"biomas_fire_{_cache_key(years)}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    # Average the annual hotspot count across `years`. A year with no detections
    # in a muni is a missing row → filled with 0, so absences correctly pull the
    # mean down rather than being dropped.
    per_year = [_focos_counts(y) for y in years]
    counts = pd.concat(per_year, axis=1).fillna(0).mean(axis=1).rename("focos")

    import geopandas as gpd
    from utils import MUNI_SHAPEFILE
    gdf = gpd.read_file(MUNI_SHAPEFILE).to_crs("EPSG:5880")

    code_col = "code_muni" if "code_muni" in gdf.columns else "CD_MUN"
    name_col = next((c for c in ("name_muni", "NM_MUN", "nm_mun") if c in gdf.columns), None)
    uf_col   = next((c for c in ("abbrev_state", "abbrev_sta", "SIGLA_UF", "sigla_uf") if c in gdf.columns), None)
    if not (name_col and uf_col):
        raise RuntimeError(f"Cannot find name+UF cols in {MUNI_SHAPEFILE}; got {gdf.columns.tolist()[:10]}")

    gdf["code_muni"]     = normalize_code(gdf[code_col])
    gdf["state_acronym"] = gdf[uf_col].astype(str).str.strip().str.upper()
    gdf["muni_key"]      = gdf[name_col].astype(str).str.strip().str.lower()
    gdf["area_km2"]      = gdf.geometry.area / 1e6
    gdf = gdf.drop_duplicates("code_muni")

    merged = gdf[["code_muni", "state_acronym", "muni_key", "area_km2"]] \
                .merge(counts.reset_index(), on=["state_acronym", "muni_key"], how="left")
    merged["focos"] = merged["focos"].fillna(0)
    # 0.01 km² per hotspot ≈ 1 ha (each MODIS hotspot covers roughly 1 ha).
    merged["fire_pct"] = 100.0 * (merged["focos"] * 0.01) / merged["area_km2"]
    out = merged[["code_muni", "fire_pct"]]

    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out
