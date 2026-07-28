"""
Moran's I on municipality-level out-migration flows, plus an OLS table with
spatial diagnostics and a flow choropleth.

Inputs are the modelling splits from `models/make_splits.py` and the harmonized
boundaries from the pipeline. Run from anywhere:

    python data-pipeline/analysis/moran.py

Outputs go to this directory (`ols_summary.txt`, `flow_map.png`).
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from libpysal.weights import KNN
from esda.moran import Moran
from spreg import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Repo root: this file is at <root>/data-pipeline/analysis/moran.py
ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROCESSED = ROOT / "data-pipeline" / "data" / "processed"

plt.rcParams.update({
    "font.size": 26,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
    "legend.fontsize": 19,
    "figure.titlesize": 24,
    "font.family": "serif",
})

# The splits used to be written as `data/processed/X_train_mx.csv`; they are now
# `data/X_train.csv` (the `_mx` suffix is redundant in a Mexico-only repository)
# and are produced by models/make_splits.py. `source_code` is read as a string so
# the zero padding on states 01-09 survives the join to `cvegeo` below.
_x_path = DATA_DIR / "X_train.csv"
if not _x_path.exists():
    raise SystemExit(
        f"missing {_x_path}"
        "\nBuild the splits first:  python models/make_splits.py"
    )

x_train = pd.read_csv(_x_path, dtype={"source_code": str, "dest_code": str})
y_train = pd.read_csv(DATA_DIR / "y_train.csv").rename(columns={'migrants': 'flow'})
df = pd.concat([x_train, y_train], axis=1)

muni_flows = df.groupby('source_code')['flow'].sum().reset_index()

coords_df = pd.read_csv(PROCESSED / "municipios_centroids.csv",
                        dtype={"cvegeo": str})[['cvegeo', 'lat', 'lon']]
coords_df['cvegeo'] = coords_df['cvegeo'].astype(str).str.zfill(5)

muni_flows['source_code'] = muni_flows['source_code'].astype(str).str.zfill(5)

x_features = x_train.groupby('source_code').mean(numeric_only=True).reset_index()
x_features['source_code'] = x_features['source_code'].astype(str).str.zfill(5)

print("muni_flows source_code sample:", muni_flows['source_code'].head().tolist())
print("coords_df cvegeo sample:", coords_df['cvegeo'].head().tolist())
print("muni_flows dtype:", muni_flows['source_code'].dtype)
print("coords_df dtype:", coords_df['cvegeo'].dtype)

muni_data = (
    muni_flows
    .merge(x_features, on='source_code')
    .merge(coords_df, left_on='source_code', right_on='cvegeo', how='inner')
)

print("muni_data rows after merge:", len(muni_data))
if len(muni_data) == 0:
    raise ValueError(
        "Merge produced 0 rows. Check that source_code and cvegeo "
        "use the same code format (width/leading zeros/state+muni order)."
    )

coords = muni_data[['lon', 'lat']].values
w = KNN.from_array(coords, k=5)
w.transform = 'R'

explicit_drop = [
    'source_code',
    'flow',
    'cvegeo',
    'lat',
    'lon',
    'dest_code'
]

derived_drops = [
    c for c in muni_data.columns
    if '_diff' in c or '_ratio' in c
]

distance_drops = (
    ['distance_km']
    if 'log_distance_km' in muni_data.columns
    else []
)

exclude = explicit_drop + derived_drops + distance_drops

X_df = (
    muni_data[
        [c for c in muni_data.columns if c not in exclude]
    ]
    .select_dtypes(include=[np.number])
)

while True:
    Xc = X_df.assign(const=1)
    vifs = [
        variance_inflation_factor(Xc.values, i)
        for i in range(Xc.shape[1])
    ]

    idx = np.argmax(vifs[:-1])
    vmax = vifs[idx]

    if vmax <= 10:
        break

    X_df = X_df.drop(columns=[X_df.columns[idx]])

feature_cols = X_df.columns.tolist()
X = X_df.values
y = np.log1p(muni_data['flow'].values.reshape(-1, 1))

ols = OLS(
    y,
    X,
    w=w,
    name_y='log_flow',
    name_x=feature_cols,
    spat_diag=True,
    moran=True
)

print(ols.summary)

lm_lag, lm_lag_p = ols.lm_lag
lm_err, lm_err_p = ols.lm_error
rlm_lag, rlm_lag_p = ols.rlm_lag
rlm_err, rlm_err_p = ols.rlm_error

alpha = 0.05

print("\n=== SPECIFICATION DECISION ===")

if lm_lag_p < alpha and lm_err_p >= alpha:
    print("SLM")
elif lm_err_p < alpha and lm_lag_p >= alpha:
    print("SEM")
elif lm_lag_p < alpha and lm_err_p < alpha:
    if rlm_lag_p < alpha and rlm_err_p >= alpha:
        print("SLM")
    elif rlm_err_p < alpha and rlm_lag_p >= alpha:
        print("SEM")
    elif rlm_lag_p < alpha and rlm_err_p < alpha:
        if rlm_lag > rlm_err:
            print("SLM")
        else:
            print("SEM")
    else:
        print("Ambiguous")
else:
    print("OLS")

resid = np.asarray(ols.u).flatten()
mi = Moran(resid, w)

with open(HERE / "ols_summary.txt", "w", encoding="utf-8") as f:
    f.write(str(ols.summary))

"""from esda.moran import Moran
import matplotlib.pyplot as plt

resid = np.asarray(ols.u).flatten()
lag_resid = w.sparse @ resid
mi = Moran(resid, w)

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(
    resid,
    lag_resid,
    s=18,
    alpha=0.7,
    color="#2c7fb8",
    edgecolor="none"
)

m, b = np.polyfit(resid, lag_resid, 1)
x = np.linspace(resid.min(), resid.max(), 300)
ax.plot(x, m * x + b, color="red", linewidth=2)

ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)

ax.set_xlabel("OLS Residuals")
ax.set_ylabel("Spatial Lag of Residuals")

ax.set_title(
    f"Residual Moran Scatterplot\n"
    f"Global Moran's I: {mi.I:.4f} (p: {mi.p_sim:.4f})"
)

plt.tight_layout()
plt.savefig(
    "data/moran_scatterplot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()"""

# --- Choropleth map using municipality polygons ---
gdf = gpd.read_file(
    PROCESSED / "municipios_harmonized_2020_shp" / "municipios_harmonized_2020.shp"
)
gdf['cvegeo'] = gdf['cvegeo'].astype(str).str.zfill(5)

map_df = gdf.merge(muni_data, on='cvegeo', how='left')

valid_map_df = map_df.dropna(subset=['flow']).reset_index(drop=True)
w_map = KNN.from_dataframe(valid_map_df, k=5)
w_map.transform = 'R'
mi_flow = Moran(valid_map_df['flow'].values, w_map)

# The headline spatial-autocorrelation figure. Printed as well as drawn on the
# map, because it is a reported number and should not have to be read off a PNG.
print(f"\n=== GLOBAL MORAN'S I, out-migration flows by source municipality ===")
print(f"  I = {mi_flow.I:.4f}   p_sim = {mi_flow.p_sim:.4f}   "
      f"n = {len(valid_map_df)} municipios")

fig, ax = plt.subplots(figsize=(12, 12))

map_df.plot(
    column='flow',
    cmap='viridis',
    scheme='quantiles',
    k=5,
    edgecolor='none',
    legend=True,
    legend_kwds={
        'title': 'Out-migration Flows',
        'loc': 'lower left',
        'fmt': "{:.2f}"
    },
    missing_kwds={
        "color": "#e0e0e0"
    },
    ax=ax
)

ax.set_axis_off()
ax.set_title(
    f"Total Out-Migration Flows by Source Municipality\n"
    f"Global Moran's I: {mi_flow.I:.4f} (p: {mi_flow.p_sim:.4f})",
    pad=15
)

plt.tight_layout()
plt.savefig(HERE / "flow_map.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()