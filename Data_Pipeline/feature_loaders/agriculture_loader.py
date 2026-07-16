"""Per-muni crop area shares from IBGE PAM via the SIDRA JSON API.

Public function: crop_pct_per_muni(year) → 6 cols:
  coffee_pct, common_bean_pct, corn_pct,
  cotton_pct, soybean_pct, sugarcane_pct

Source: IBGE SIDRA API (https://servicodados.ibge.gov.br/api/docs/agregados?versao=3).
  Table 1612 = lavouras temporárias, classification 81.
  Table 1613 = lavouras permanentes, classification 82.
  Variable 109 = "Área plantada" (planted area, hectares).

Codes were verified against the live SIDRA metadata endpoint
(https://servicodados.ibge.gov.br/api/v3/agregados/1612/metadados etc.).
"""
from __future__ import annotations

from pathlib import Path

import requests
import pandas as pd

from utils import normalize_code, load_areas, MUNI_GPKG

SIDRA = "https://servicodados.ibge.gov.br/api/v3/agregados"
CACHE = Path(__file__).resolve().parent.parent / "data_cache"

# Variable holding "planted-equivalent" area, by table:
#   1612 (temporárias) → 109 (Área plantada)
#   1613 (permanentes) → 2313 (Área destinada à colheita; perennials aren't replanted)
#   1002 (feijão safras) / 839 (milho safras) → 109 (Área plantada)
AREA_VAR = {1612: 109, 1613: 2313, 1002: 109, 839: 109}

# (table, classification_id, category_id) for each crop.
# Beans and corn use 1ª-safra-only tables because PAM publishes them split by
# harvest; IBGE's annual-total table 1612 sums all safras → ~2-3× too high in
# states with a strong 2ª safra, so use the per-safra tables and pick 1ª safra:
#   table 1002 cat 117991 = "Feijão (em grão) - 1ª safra"
#   table 839  cat 114253 = "Milho (em grão) - 1ª safra"
CROPS = {
    "coffee":       (1613, 82, 2723),    # Café (em grão) Total
    "common_bean":  (1002, 81, 117991),  # Feijão (em grão) - 1ª safra only
    "corn":         (839,  81, 114253),  # Milho (em grão) - 1ª safra only
    "cotton":       (1612, 81, 2689),    # Algodão herbáceo (em caroço)
    "soybean":      (1612, 81, 2713),    # Soja (em grão)
    "sugarcane":    (1612, 81, 2696),    # Cana-de-açúcar
}

_MISSING = {None, "...", "-", "X", "..", ""}


def _fetch(table: int, cls: int, cats: list[int], year: int) -> dict[int, pd.Series]:
    """One SIDRA call for several categories at once → {category_id: per-muni area Series}.

    All crops sharing a (table, classification) are pulled in a single request;
    SIDRA returns one `resultado` per category, tagged by its `classificacoes`.
    """
    url = (f"{SIDRA}/{table}/periodos/{year}/variaveis/{AREA_VAR[table]}"
           f"?localidades=N6[all]&classificacao={cls}[{','.join(map(str, cats))}]")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return {}

    out: dict[int, pd.Series] = {}
    for var_block in data:
        if not isinstance(var_block, dict):
            continue
        for result in var_block.get("resultados", []):
            cat = _result_category(result, cats)
            if cat is None:
                continue
            pairs = []
            for series in result.get("series", []):
                val = series.get("serie", {}).get(str(year))
                if val in _MISSING:
                    continue
                try:
                    pairs.append((series["localidade"]["id"], float(val)))
                except (TypeError, ValueError):
                    continue
            if pairs:
                s = pd.Series(dict(pairs), dtype=float)
                s.index = normalize_code(s.index)
                out[cat] = s
    return out


def _result_category(result: dict, cats: list[int]) -> int | None:
    """Which requested category a SIDRA `resultado` block belongs to."""
    for c in result.get("classificacoes", []):
        for cid in c.get("categoria", {}):
            if int(cid) in cats:
                return int(cid)
    return None


def _planted_ha(year: int) -> pd.DataFrame:
    """Per-muni planted area (hectares) for every crop, indexed by code_muni.

    One SIDRA request per (table, classification) group, then cached to parquet
    so re-runs never re-hit the API.
    """
    cache = CACHE / f"agri_planted_ha_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache).set_index("code_muni")

    groups: dict[tuple[int, int], list[int]] = {}
    for table, cls, cat in CROPS.values():
        groups.setdefault((table, cls), []).append(cat)

    by_cat: dict[int, pd.Series] = {}
    for (table, cls), cats in groups.items():
        by_cat.update(_fetch(table, cls, cats, year))

    df = pd.DataFrame({crop: by_cat.get(cat) for crop, (_, _, cat) in CROPS.items()})
    df.index.name = "code_muni"
    CACHE.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_parquet(cache, index=False)
    return df


def crop_pct_per_muni(year: int = 2010) -> pd.DataFrame:
    """% of muni land area planted to each crop. Zero-fill missing, clip to [0, 100]."""
    ha = _planted_ha(year)
    areas = load_areas(MUNI_GPKG)
    area_km2 = pd.Series(areas["area_km2"].values, index=normalize_code(areas["code_muni"]))
    area_km2 = area_km2[~area_km2.index.duplicated(keep="first")]

    out = pd.DataFrame({"code_muni": area_km2.index})
    for crop in CROPS:
        # 1 km² = 100 ha, so pct = planted_ha / (area_km2 * 100) * 100 = planted_ha / area_km2.
        pct = (ha[crop] / area_km2).reindex(out["code_muni"]).fillna(0.0)
        # IBGE PAM occasionally reports planted area > muni area for intensively-
        # rotated crops (sugar cane in São Paulo — same field rotated twice in one
        # calendar year; one muni came out at 124.3%). Clip to the mathematical
        # bound so downstream consumers can trust the column as a true percent.
        out[f"{crop}_pct"] = pct.clip(0.0, 100.0).values
    return out
