"""Dyad construction — layers 2 through 5 collapsed.

  merge_features(feature_tables)   master per-muni feature table
  impute(df)                        KNN-neighbor fill for selected column families
  encode(df)                        one-hot expand top_crop_2010 → crop_* (no-op: crop_economics() no longer emits top_crop_2010)
  build_dyad(features, flows)       mirror src_/dst_, distance, log_flow, *_diffs, same_state

This is the only file that contains pipeline orchestration logic for the join
chain. The 5 separate sub-modules in data_pipeline/pipeline/{layer2..layer5}/
collapse here because they're 95% the same shape: read parquet → join → write.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (
    FINAL,
    RAW,
    haversine_km,
    load_centroids,
    normalize_code,
)


# ---------------------------------------------------------------------------
# Layer 2 + 3 — merge per-muni feature tables
# ---------------------------------------------------------------------------
def merge_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join all per-muni feature tables on code_muni.

    `tables` is {name: DataFrame}. Each table must have a `code_muni` column.
    Columns that collide get suffixed with `__{name}` to keep things explicit.
    """
    master: pd.DataFrame | None = None
    seen: set[str] = {"code_muni"}
    for name, df in tables.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df["code_muni"] = normalize_code(df["code_muni"])
        rename = {c: f"{c}__{name}" for c in df.columns if c in seen and c != "code_muni"}
        if rename:
            df = df.rename(columns=rename)
        seen.update(c for c in df.columns if c != "code_muni")
        master = df if master is None else master.merge(df, on="code_muni", how="outer")
    return master if master is not None else pd.DataFrame(columns=["code_muni"])


# ---------------------------------------------------------------------------
# Layer 3b — population accessibility (gravity potential)
#
#   accessibility_i = Σ_j  pop_j / (haversine_km(i, j) + 1)   −  pop_i
#
# A gravity-style potential: every muni's population discounted by its distance
# to i, with the self term (d = 0 → pop_i / 1) removed so a muni does not count
# its own population. The +1 km softening keeps near-zero-distance terms finite.
# Needs the whole country at once, so it runs on the merged per-muni master.
# ---------------------------------------------------------------------------
def add_accessibility(df: pd.DataFrame,
                      pop_col: str = "total_pop_2010") -> pd.DataFrame:
    if pop_col not in df.columns:
        print(f"  [accessibility] {pop_col} missing — skipping")
        return df

    from utils import MUNI_GPKG
    centroids = load_centroids(MUNI_GPKG)
    m = df.merge(centroids, on="code_muni", how="left")
    lat = m["lat"].to_numpy(dtype=float)
    lon = m["lon"].to_numpy(dtype=float)
    # Missing population contributes zero mass to everyone's potential.
    pop = pd.to_numeric(m[pop_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    n = len(m)
    acc = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(lat[i]) or np.isnan(lon[i]):
            continue  # no centroid → accessibility undefined for this muni
        d = haversine_km(lat[i], lon[i], lat, lon)
        acc[i] = float(np.nansum(pop / (d + 1.0)) - pop[i])

    out = df.copy()
    out["accessibility"] = acc
    return out


# ---------------------------------------------------------------------------
# Layer 3c — population density (persons per km^2)
#
#   population_density_i = pop_i / area_km2_i
#
# Per-muni feature, so build_dyad mirrors it into src_population_density /
# dst_population_density. Areas come from the same 2010 IBGE polygons (EPSG:5880)
# the pipeline already uses for centroids.
# ---------------------------------------------------------------------------
def add_pop_density(df: pd.DataFrame,
                    pop_col: str = "total_pop_2010") -> pd.DataFrame:
    if pop_col not in df.columns:
        print(f"  [pop_density] {pop_col} missing — skipping")
        return df

    from utils import MUNI_GPKG, load_areas
    areas = load_areas(MUNI_GPKG)
    out = df.merge(areas, on="code_muni", how="left")
    pop = pd.to_numeric(out[pop_col], errors="coerce")
    out["population_density"] = pop / out["area_km2"].replace(0, np.nan)
    return out.drop(columns=["area_km2"])


# ---------------------------------------------------------------------------
# Layer 3d — GDP per capita (persons denominator = census population)
#
#   GDP_per_capita = gdp_brl_2010 / total_pop_2010
#
# Total GDP comes from features.gdp_total (SIDRA 5938 var 37). The raw total is
# consumed here and dropped, leaving only the per-capita column, which build_dyad
# mirrors into src_/dst_.
# ---------------------------------------------------------------------------
def add_gdp_per_capita(df: pd.DataFrame,
                       gdp_col: str = "gdp_brl_2010",
                       pop_col: str = "total_pop_2010") -> pd.DataFrame:
    if gdp_col not in df.columns or pop_col not in df.columns:
        print(f"  [gdp_per_capita] {gdp_col}/{pop_col} missing — skipping")
        return df

    out = df.copy()
    pop = pd.to_numeric(out[pop_col], errors="coerce").replace(0, np.nan)
    out["GDP_per_capita"] = pd.to_numeric(out[gdp_col], errors="coerce") / pop
    return out.drop(columns=[gdp_col])


# ---------------------------------------------------------------------------
# Layer 4 — per-column imputation
#
# Strategy assignments:
#   idw_knn      → ndvi_*           (V3/data_filling_ndvi.ipynb)
#   state_mean   → urban_area_pct   (V4/V4.ipynb cell 20)
#   median       → 43 climate cols  (DataEngineeringV1.ipynb cell 17)
#   global_median→ total_flow       (DataEngineeringV1.ipynb cell 8) — not
#                                     produced by this pipeline but kept for
#                                     parity if a downstream caller adds it
#   zero         → everything else  (DataEngineeringV1.ipynb cell 18 catch-all)
# ---------------------------------------------------------------------------
NDVI_COLS = ["ndvi"]

STATE_MEAN_COLS = ["urban_area_pct"]

# The per-muni climate columns (all from climate_loader). Median-fill any
# straggler NaN rather than zero — 0 °F / 0-day-streak / 0 mm would be a wrong
# value, not "missing". Anything not present in `df` is silently skipped. The
# climate_baseline columns (temp/precip/z_from_baseline_*) are already IDW-KNN
# imputed in-loader; median here is only a safety net (e.g. a z-score where the
# baseline std is 0). temp_diff/precip_diff are derived post-mirror, not here.
MEDIAN_COLS = [
    "wind_mean",
    "temp", "precip",
    "uv_log_mean",
    "wet_bulb_F", "num_degreedays", "degreedays_streak",
    "z_from_baseline_temp", "z_from_baseline_precip",
]

GLOBAL_MEDIAN_COLS = ["total_flow"]


def _strategy_for(col: str) -> str:
    if col in NDVI_COLS:           return "idw_knn"
    if col in STATE_MEAN_COLS:     return "state_mean"
    if col in MEDIAN_COLS:         return "median"
    if col in GLOBAL_MEDIAN_COLS:  return "global_median"
    return "zero"


def imputation_strategy_for_columns(cols) -> dict[str, str]:
    """Public introspection: map each column name to its imputation strategy."""
    return {c: _strategy_for(c) for c in cols}


def _idw_knn_fill(values: pd.Series, lat: pd.Series, lon: pd.Series,
                  k: int = 8, power: int = 2, eps: float = 1e-12) -> pd.Series:
    """Port of V3/data_filling_ndvi.ipynb (BallTree haversine, k=8, power=2)."""
    from sklearn.neighbors import BallTree

    s = values.copy()
    coords_ok = lat.notna() & lon.notna()
    donors  = coords_ok & s.notna()
    targets = coords_ok & s.isna()

    if targets.sum() == 0 or donors.sum() < 2:
        return s

    donors_xy  = np.deg2rad(np.column_stack([lat[donors].to_numpy(),  lon[donors].to_numpy()]))
    targets_xy = np.deg2rad(np.column_stack([lat[targets].to_numpy(), lon[targets].to_numpy()]))

    tree = BallTree(donors_xy, metric="haversine")
    k_use = min(k, int(donors.sum()))
    dist, ind = tree.query(targets_xy, k=k_use)

    donor_vals = s[donors].to_numpy(dtype=float)
    w = 1.0 / np.maximum(dist, eps) ** power
    pred = (w * donor_vals[ind]).sum(axis=1) / w.sum(axis=1)

    s.loc[targets] = pred
    return s


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column imputation.

    Strategies (see module docstring): IDW-KNN for NDVI, state-mean for
    urban_area_pct, per-column median for the 43 climate columns, and a
    catch-all zero-fill for everything else still NaN — this mirrors
    `DataEngineeringV1.ipynb` cell 18 (`DE_V1 = DE_V1.fillna(0)`).
    """
    df = df.copy()

    # 1. NDVI: IDW-KNN haversine
    ndvi_present = [c for c in NDVI_COLS if c in df.columns and df[c].isna().any()]
    if ndvi_present:
        from utils import MUNI_GPKG
        centroids_ll = load_centroids(MUNI_GPKG)
        df = df.merge(centroids_ll, on="code_muni", how="left")
        lat, lon = df["lat"], df["lon"]
        for col in ndvi_present:
            df[col] = _idw_knn_fill(df[col], lat, lon)
        df = df.drop(columns=["lat", "lon"])

    # 2. State-mean (urban_area_pct). State code is first 2 digits of code_muni.
    state = df["code_muni"].astype(str).str[:2]
    for col in STATE_MEAN_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df.groupby(state)[col].transform("mean"))

    # 3. Per-column median for the 43 raw climate columns
    for col in MEDIAN_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # 4. Global median for total_flow (parity with DataEngineeringV1.ipynb c8)
    for col in GLOBAL_MEDIAN_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # 5. Catch-all zero-fill for any remaining NaN.
    for col in df.columns:
        if col == "code_muni":
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().any():
            df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Layer 4b — one-hot encode top_crop_2010
# ---------------------------------------------------------------------------
EXPECTED_TOP_CROPS = [
    "Arroz (em casca)",
    "Banana (cacho)",
    "Café (em grão) Total",
    "Cana-de-açúcar",
    "Fumo (em folha)",
    "Mandioca",
    "Milho (em grão)",
    "Soja (em grão)",
]


def encode(df: pd.DataFrame, drop_original: bool = False) -> pd.DataFrame:
    if "top_crop_2010" not in df.columns:
        return df
    # Build all one-hot columns at once and concat once. Assigning them one by
    # one into the already-wide frame fragments pandas' block manager and emits
    # a PerformanceWarning; this is value-identical but avoids the churn.
    crops = pd.DataFrame(
        {f"crop_{cat}": (df["top_crop_2010"] == cat).astype(int)
         for cat in EXPECTED_TOP_CROPS},
        index=df.index,
    )
    df = pd.concat([df, crops], axis=1)
    if drop_original:
        df = df.drop(columns=["top_crop_2010"])
    return df


# ---------------------------------------------------------------------------
# Layer 5 — dyad construction
# ---------------------------------------------------------------------------
def _mirror(features: pd.DataFrame, key: str = "code_muni") -> tuple[pd.DataFrame, pd.DataFrame]:
    feat_cols = [c for c in features.columns if c != key]
    src = features.rename(columns={c: f"src_{c}" for c in feat_cols}) \
                  .rename(columns={key: "source_code"})
    dst = features.rename(columns={c: f"dst_{c}" for c in feat_cols}) \
                  .rename(columns={key: "dest_code"})
    return src, dst


def _add_pair_features(dyad: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    dyad["log_flow"] = np.log1p(dyad["flow"])

    sc = centroids.rename(columns={"code_muni": "source_code", "lat": "slat", "lon": "slon"})
    dc = centroids.rename(columns={"code_muni": "dest_code",   "lat": "dlat", "lon": "dlon"})
    dyad = dyad.merge(sc, on="source_code", how="left") \
               .merge(dc, on="dest_code",   how="left")
    dyad["distance_km"]     = haversine_km(dyad["slat"], dyad["slon"], dyad["dlat"], dyad["dlon"])
    dyad["log_distance_km"] = np.log1p(dyad["distance_km"])
    dyad = dyad.drop(columns=["slat", "slon", "dlat", "dlon"])

    def has(a, b): return a in dyad.columns and b in dyad.columns

    if has("src_total_pop_2010", "dst_total_pop_2010"):
        dyad["pop_ratio"] = (dyad["src_total_pop_2010"]
                              / dyad["dst_total_pop_2010"].replace(0, np.nan))
    if has("src_mean_income_brl_2010", "dst_mean_income_brl_2010"):
        dyad["income_diff"] = dyad["dst_mean_income_brl_2010"] - dyad["src_mean_income_brl_2010"]
    if has("src_temp", "dst_temp"):
        dyad["temp_diff"] = dyad["dst_temp"] - dyad["src_temp"]
    if has("src_precip", "dst_precip"):
        dyad["precip_diff"] = dyad["dst_precip"] - dyad["src_precip"]
    if has("src_ndvi", "dst_ndvi"):
        dyad["ndvi_diff"] = dyad["dst_ndvi"] - dyad["src_ndvi"]

    dyad["same_state"] = (
        dyad["source_code"].astype(str).str[:2]
        == dyad["dest_code"].astype(str).str[:2]
    ).astype(int)
    return dyad


def build_dyad(features: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features["code_muni"] = normalize_code(features["code_muni"])
    flows = flows.copy()
    flows["source_code"] = normalize_code(flows["source_code"])
    flows["dest_code"]   = normalize_code(flows["dest_code"])

    src, dst = _mirror(features)
    dyad = flows.merge(src, on="source_code", how="left") \
                .merge(dst, on="dest_code",   how="left")

    from utils import MUNI_GPKG
    centroids = load_centroids(MUNI_GPKG)
    return _add_pair_features(dyad, centroids)


# ---------------------------------------------------------------------------
# Layer 5b — final column projection (whitelist)
#
# The dyad frame out of build_dyad carries a src_/dst_ mirror of EVERY per-muni
# feature plus the pair-level columns. This trims it to exactly the agreed
# variable set: each per-muni var below appears mirrored (src_<v>, dst_<v>), and
# the dyad-level vars appear once. Anything else (stray mirrored columns from a
# loader, intermediate helpers) is dropped. Missing expected columns are only
# warned about — a skipped upstream loader shouldn't hard-fail the projection.
# ---------------------------------------------------------------------------

# Per-muni features kept in the final dataset (each mirrored into src_/dst_).
FINAL_MUNI_VARS = [
    "coffee_pct", "common_bean_pct", "corn_pct", "cotton_pct", "soybean_pct", "sugarcane_pct",
    "agri_gdp_mil_brl_2010", "agri_hhi_2010_2010", "agri_nat_pct_2010", "agri_value_cagr_pct_2010",
    "wind_mean", "wet_bulb_F", "precip", "temp", "uv_log_mean",
    "num_degreedays", "degreedays_streak", "z_from_baseline_temp", "z_from_baseline_precip",
    "pct_urban_w_2010", "pct_18_35_2010", "male_pct_w_2010", "informal_pct_2010", "unemp_pct_2010",
    "primary_or_less_%_2010", "secondary_%_2010", "tertiary_%_2010",
    "pct_pension_2010", "hh_mean_size_2010", "dependency_ratio_2010", "pct_bolsa_familia_2010",
    "mean_income_brl_2010", "total_pop_2010",
    "population_density", "GDP_per_capita",
    "no_electricity_%_2010", "no_piped_water_%_2010", "no_sewage_%_2010",
    "indigenous_pct", "deforestation_pct", "accessibility",
    "mining_pct", "pasture_pct", "urban_area_pct",
    "disaster_pct", "fire_pct", "semiarid_pct",
    "road_density", "railway_density",
    "ndvi",
]

# Dyad-level columns kept as-is (not mirrored). flow/log_flow retained per spec.
FINAL_DYAD_VARS = [
    "source_code", "dest_code", "flow", "log_flow",
    "distance_km", "log_distance_km", "same_state", "pop_ratio",
    "income_diff", "temp_diff", "precip_diff", "ndvi_diff",
]


def select_final(dyad: pd.DataFrame) -> pd.DataFrame:
    """Project the dyad frame onto exactly the agreed final variable set."""
    allowed = (list(FINAL_DYAD_VARS)
               + [f"src_{v}" for v in FINAL_MUNI_VARS]
               + [f"dst_{v}" for v in FINAL_MUNI_VARS])
    allowed_set = set(allowed)

    keep    = [c for c in allowed if c in dyad.columns]
    missing = [c for c in allowed if c not in dyad.columns]
    dropped = [c for c in dyad.columns if c not in allowed_set]

    if missing:
        print(f"  [select_final] {len(missing)} expected column(s) absent "
              f"(upstream loader skipped?): {missing}")
    if dropped:
        print(f"  [select_final] dropped {len(dropped)} column(s) not in the final set")
    print(f"  [select_final] kept {len(keep)} columns")
    return dyad[keep]


def save_final(dyad: pd.DataFrame, name: str = "dyad_dataset.csv") -> str:
    out = FINAL / name
    out.parent.mkdir(parents=True, exist_ok=True)
    dyad.to_csv(out, index=False)
    return str(out)
