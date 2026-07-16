"""Per-municipality features whose source is the censobr 2010 parquet or
the IBGE crops CSVs. Every other feature family lives in features_<domain>.py.

The pipeline is fully self-contained — no dependency on any external folder;
it can be cloned alone and run against fresh institutional data.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils import RAW, normalize_code

from utils import MUNI_SHAPEFILE

PATHS = {
    "ibge_perm_crops":   RAW / "ibge_crops" / "permanent_crops_2005_2010.csv",
    "ibge_temp_crops":   RAW / "ibge_crops" / "temporary_crops_2005_2010.csv",
    "shapefile":         MUNI_SHAPEFILE,
}


# ---------------------------------------------------------------------------
# Migration totals — per-muni population, inflow, outflow, net flow and pcts,
# computed from the censobr 2010 person parquet.
# ---------------------------------------------------------------------------
def migration_totals_from_parquet() -> pd.DataFrame:
    """Per-muni population + inflow + outflow + net_flow + pct columns."""
    from utils import read_population
    df = read_population(["V6264"])
    df["w"] = pd.to_numeric(df["V0010"], errors="coerce")
    pop = df.groupby("code_muni")["w"].sum().rename("population")

    migs = df.dropna(subset=["V6264"]).copy()
    migs["source_code"] = normalize_code(migs["V6264"])
    migs["dest_code"]   = migs["code_muni"]
    migs = migs[(migs["source_code"].str.len() == 7)
                & (migs["source_code"] != migs["dest_code"])]

    # inflow/outflow = sum(V0010), the sample-expansion weights — yielding
    # expanded population totals. Using .size() here would be an unweighted
    # record count (≈10× too small). The `flow` column IS an unweighted count,
    # kept in flows_edgelist.
    inflow  = migs.groupby("dest_code")["w"].sum().rename("inflow")
    outflow = migs.groupby("source_code")["w"].sum().rename("outflow")

    out = pd.DataFrame({"code_muni": pop.index})
    out["population"] = pop.values
    out = out.merge(inflow.reset_index().rename(columns={"dest_code": "code_muni"}),
                    on="code_muni", how="left")
    out = out.merge(outflow.reset_index().rename(columns={"source_code": "code_muni"}),
                    on="code_muni", how="left")
    out[["inflow", "outflow"]] = out[["inflow", "outflow"]].fillna(0)
    out["net_flow"] = out["inflow"] - out["outflow"]
    denom = out["population"].replace(0, np.nan)
    out["inflow_pct"]   = 100.0 * out["inflow"]   / denom
    out["outflow_pct"]  = 100.0 * out["outflow"]  / denom
    out["net_flow_pct"] = 100.0 * out["net_flow"] / denom
    return out


# ---------------------------------------------------------------------------
# Migration edgelist — unweighted V6264 count (integer O–D flows).
# ---------------------------------------------------------------------------
def flows_edgelist_from_parquet() -> pd.DataFrame:
    """O–D edgelist (source_code, dest_code, flow) — integer counts from V6264."""
    from utils import read_population
    df = read_population(["V6264"])
    migs = df.dropna(subset=["V6264"]).copy()
    migs["source_code"] = normalize_code(migs["V6264"])
    migs["dest_code"]   = migs["code_muni"]
    migs = migs[(migs["source_code"].str.len() == 7)
                & (migs["source_code"] != migs["dest_code"])]
    edges = (migs.groupby(["source_code", "dest_code"])
                 .size().reset_index(name="flow"))
    return edges[["source_code", "dest_code", "flow"]]


# ---------------------------------------------------------------------------
# State-level agricultural economic indicators — broadcast to muni level via
# state_code = code_muni[:2].
# ---------------------------------------------------------------------------
VARIABLE_MAP = {
    "Área plantada":             "area_planted",
    "Área destinada à colheita": "area_planted",
    "Área colhida":              "area_harvested",
    "Quantidade produzida":      "production_qty",
    "Valor da produção":         "production_value",
}
# Columns the rest of crop_economics() indexes into; guaranteed to exist even
# when a source block is absent (e.g. a table without "Área plantada").
_VAR_KEYS = ("area_planted", "area_harvested", "production_qty", "production_value")
_YEAR_RE = re.compile(r"^\d{4}$")


def _parse_sidra_crop_csv(path: Path) -> pd.DataFrame:
    """Parse an IBGE SIDRA `geratabela` CSV (table 1612/1613) into long format.

    These exports are *not* the tidy "Linha por observação" layout — they stack
    one wide cross-tab per variable ("Área colhida", "Quantidade produzida",
    "Valor da produção", …). Each block looks like::

        "Tabela 1612 - …"                              ← title banner
        "Variável - Área colhida (Hectares)"           ← variable marker
        "Cód.","Unidade da Federação","Ano x Produto…"  ← layout banner (3 cols)
        "Cód.","Unidade da Federação","2005",,, … "2010" ← year row (sparse)
        "Cód.","Unidade da Federação","Abacaxi*", …      ← product row
        "11","Rondônia",515,…                            ← one data row per UF
        …
        "Fonte: IBGE …"                                  ← footer

    We walk the rows, slice out each block, forward-fill the sparse year row
    across its product columns, and melt back to
    (state_code, state, year, crop, variable, value).
    """
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    records: list[dict] = []
    i, n = 0, len(rows)
    while i < n:
        row = rows[i]
        if not (row and row[0].startswith("Variável")):
            i += 1
            continue
        # "Variável - Área colhida (Hectares)" -> "Área colhida"
        variable = row[0].split(" - ", 1)[-1].rsplit(" (", 1)[0].strip()
        i += 1

        # The year and product header rows both start with "Cód."; a 3-column
        # "Cód." row is just the layout banner and is skipped. The year row is
        # the one whose first data cell is a 4-digit year.
        year_row = product_row = None
        while i < n and rows[i] and rows[i][0] == "Cód.":
            r = rows[i]
            i += 1
            if len(r) <= 3:
                continue
            if year_row is None and _YEAR_RE.match(r[2].strip()):
                year_row = r
            elif product_row is None:
                product_row = r
                break
        if year_row is None or product_row is None:
            continue

        products = [p.strip() for p in product_row[2:]]
        year_cells = year_row[2:]
        years: list[int | None] = []
        cur: int | None = None
        for k in range(len(products)):
            if k < len(year_cells) and _YEAR_RE.match(year_cells[k].strip()):
                cur = int(year_cells[k].strip())
            years.append(cur)

        # Data rows run until the "Fonte:" footer, a blank line, or next block.
        while i < n:
            r = rows[i]
            if not r or not r[0].strip() or not r[0].strip()[0].isdigit():
                break
            state_code, state = r[0].strip(), r[1].strip()
            for k, crop in enumerate(products):
                records.append({
                    "state_code": state_code, "state": state,
                    "year": years[k], "crop": crop, "variable": variable,
                    "value": r[2 + k] if 2 + k < len(r) else "",
                })
            i += 1

    return pd.DataFrame.from_records(
        records,
        columns=["state_code", "state", "year", "crop", "variable", "value"],
    )


def _load_crop_pivot() -> pd.DataFrame:
    raw = pd.concat(
        [_parse_sidra_crop_csv(PATHS["ibge_temp_crops"]),
         _parse_sidra_crop_csv(PATHS["ibge_perm_crops"])],
        ignore_index=True,
    )
    # SIDRA us.csv uses "," as a thousands separator and "-"/"..."/"X" for
    # missing; coercion turns every non-numeric token into NaN.
    raw["value"] = pd.to_numeric(
        raw["value"].str.strip().str.replace(",", "", regex=False), errors="coerce",
    )
    raw = raw.dropna(subset=["value", "year"])
    raw["year"] = raw["year"].astype(int)
    raw["crop"] = raw["crop"].str.rstrip("*").str.strip()
    # Drop SIDRA sub-aggregate roll-ups ("<crop> Total", e.g. "Café (em grão)
    # Total" = Arábica + Canephora) so summing products doesn't double-count.
    raw = raw[~raw["crop"].str.endswith("Total")].copy()
    raw["var_key"] = raw["variable"].map(VARIABLE_MAP)
    raw = raw.dropna(subset=["var_key"])

    pivot = (raw.pivot_table(
                index=["state", "state_code", "year", "crop"],
                columns="var_key", values="value", aggfunc="first")
                .reset_index().rename_axis(None, axis=1))
    for col in _VAR_KEYS:
        if col not in pivot.columns:
            pivot[col] = np.nan
    return pivot


def crop_economics() -> pd.DataFrame:
    """State-level ag indicators → broadcast to every muni in the state."""
    pivot = _load_crop_pivot()
    c2010 = pivot[(pivot["year"] == 2010) & (pivot["production_value"] > 0)]

    gdp = (c2010.groupby("state_code")["production_value"].sum()
                 .rename("agri_gdp_mil_brl_2010").reset_index())

    totals = c2010.groupby("state_code")["production_value"].sum().rename("state_total")
    sh = c2010.merge(totals, on="state_code")
    sh["share_sq"] = (100 * sh["production_value"] / sh["state_total"]) ** 2
    hhi = sh.groupby("state_code")["share_sq"].sum().rename("agri_hhi_2010_2010").reset_index()

    by_yr = pivot.groupby(["state_code", "year"])["production_value"].sum().reset_index()
    v05 = by_yr[by_yr["year"] == 2005].set_index("state_code")["production_value"]
    v10 = by_yr[by_yr["year"] == 2010].set_index("state_code")["production_value"]
    cagr = ((v10 / v05) ** (1 / 5) - 1) * 100
    cagr = cagr.rename("agri_value_cagr_pct_2010").reset_index()

    # Left-join on gdp (the full set of states with production value) so a
    # missing sub-metric yields NaN instead of dropping states.
    state_df = (gdp.merge(hhi, on="state_code", how="left")
                   .merge(cagr, on="state_code", how="left"))
    state_df["state_code"] = state_df["state_code"].astype(int)

    import geopandas as gpd
    gdf = gpd.read_file(PATHS["shapefile"])
    key = "code_muni" if "code_muni" in gdf.columns else "CD_MUN"
    code = normalize_code(gdf[key])
    munis = pd.DataFrame({
        "code_muni":  code,
        "state_code": code.str[:2].astype(int),
    })
    return munis.merge(state_df, on="state_code", how="left").drop(columns=["state_code"])


# ---------------------------------------------------------------------------
# GDP-sector shares from IBGE SIDRA table 5938 (Municipal GDP by sector).
# ---------------------------------------------------------------------------
def gdp_sector_shares(year: int = 2010) -> pd.DataFrame:
    """Each muni's share of national agricultural GVA, in %.

    Variable (verified live against SIDRA `/api/v3/agregados/5938/metadados`):
      513 = VAB Agropecuária (agriculture)
    """
    import requests
    SIDRA = "https://servicodados.ibge.gov.br/api/v3/agregados"

    def _fetch(var_id: int) -> pd.Series:
        url = f"{SIDRA}/5938/periodos/{year}/variaveis/{var_id}?localidades=N6[all]"
        data = requests.get(url, timeout=180).json()
        pairs = []
        for block in data:
            for res in block.get("resultados", []):
                for series in res.get("series", []):
                    val = series["serie"].get(str(year))
                    if val in (None, "...", "-", "X", ".."):
                        continue
                    pairs.append((series["localidade"]["id"], float(val)))
        s = pd.Series(dict(pairs))
        s.index = normalize_code(s.index)
        return s

    agri = _fetch(513)

    def _share_of_national(s: pd.Series, col: str) -> pd.DataFrame:
        national = s.sum() or np.nan
        return pd.DataFrame({"code_muni": s.index, col: 100.0 * s.values / national})

    return _share_of_national(agri, "agri_nat_pct_2010").fillna(0)


# ---------------------------------------------------------------------------
# Total municipal GDP (for GDP per capita)
# ---------------------------------------------------------------------------
def gdp_total(year: int = 2010) -> pd.DataFrame:
    """Each muni's total GDP (PIB a preços correntes), in BRL.

    Same source/table as gdp_sector_shares — SIDRA 5938, variable 37
    ("Produto Interno Bruto a preços correntes"), published in thousands of BRL,
    so it is scaled by 1000 here to land in BRL. Divided by census population
    (total_pop_2010) downstream to form GDP_per_capita.
    """
    import requests
    SIDRA = "https://servicodados.ibge.gov.br/api/v3/agregados"
    url = f"{SIDRA}/5938/periodos/{year}/variaveis/37?localidades=N6[all]"
    data = requests.get(url, timeout=180).json()

    pairs = []
    for block in data:
        for res in block.get("resultados", []):
            for series in res.get("series", []):
                val = series["serie"].get(str(year))
                if val in (None, "...", "-", "X", ".."):
                    continue
                pairs.append((series["localidade"]["id"], float(val) * 1000.0))
    s = pd.Series(dict(pairs))
    s.index = normalize_code(s.index)
    return pd.DataFrame({"code_muni": s.index, "gdp_brl_2010": s.values})


# ---------------------------------------------------------------------------
# Loader registry — pipeline.py iterates over this in canonical mode.
# ---------------------------------------------------------------------------
# Feature loaders live in ../feature_loaders (on sys.path via run_pipeline.py).
# Aliased to their historical names so the registry below reads unchanged.
import ndvi_loader as features_ndvi
import climate_loader as features_climate   # all per-muni climate from Google Earth Engine (no Copernicus CDS)
import biomas_loader as features_biomas
import agriculture_loader as features_agriculture
import transport_loader as features_transport
import vulnerability_loader as features_vulnerability

CANONICAL_LOADERS = {
    # migration_totals_from_parquet is intentionally NOT registered: its columns
    # (population/inflow/outflow/net_flow + pcts) are not part of the final
    # variable set. The unweighted O–D edgelist below is what the dyad needs.
    "crop_economics":    crop_economics,
    "gdp_sector_shares": gdp_sector_shares,
    "gdp_total":         gdp_total,
    "crop_pct":          features_agriculture.crop_pct_per_muni,
    "land_use":          features_biomas.land_use_pct_per_muni,
    "fire":              features_biomas.fire_pct_per_muni,
    "transport":         features_transport.all_densities,
    "vulnerability":     features_vulnerability.all_three,
    "ndvi":              features_ndvi.ndvi_per_muni_avg,   # 2005–2010 Sept-NDVI mean
    # All per-muni climate from Google Earth Engine (ERA5-Land 0.1° daily + ERA5/
    # HOURLY for UV) — no Copernicus CDS. Every climate loader is the 2005–2010
    # average of the annual metric (values are 6-year means; column names unchanged).
    # See utils.average_over_years and features_climate for the full variable list.
    #   wind             → wind_mean
    #   wet_bulb         → wet_bulb_F
    #   uv_annual        → uv_log_mean
    #   degreedays_extra → num_degreedays, degreedays_streak
    #   climate_baseline → temp, precip, z_from_baseline_temp, z_from_baseline_precip
    #                      (2005–2010 mean vs a 1980–2000 baseline — the slowest step)
    "wind":              features_climate.wind_per_muni_avg,
    "wet_bulb":          features_climate.wet_bulb_per_muni_avg,
    "uv_annual":         features_climate.uv_log_per_muni_avg,
    "degreedays_extra":  features_climate.degreedays_extra_per_muni_avg,
    "climate_baseline":  features_climate.climate_baseline_per_muni,
}
ALL_LOADERS = CANONICAL_LOADERS
