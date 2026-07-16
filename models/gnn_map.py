import argparse
import os

import geobr
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from libpysal.weights import KNN
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv, GCNConv

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="run name of checkpoint")
parser.add_argument("--gpu", action="store_true", default=False)
args = parser.parse_args()

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

DROP_NODE_FEATURES = []
DROP_EDGE_FEATURES = []

os.makedirs("the_gnn/maps", exist_ok=True)

X_train = pl.read_csv("data/X_train.csv")
X_test = pl.read_csv("data/X_test.csv")
X_inference = pl.read_csv("data/X_inference.csv")

id_cols = ["source_code", "dest_code"]
dyadic_cols = [
    c for c in X_train.columns
    if c not in id_cols and not c.startswith("src_") and not c.startswith("dst_")
    and c not in DROP_EDGE_FEATURES
]


def build_node_frame(df, prefix):
    id_col = "source_code" if prefix == "src_" else "dest_code"
    cols = [c for c in df.columns if c.startswith(prefix)]
    renamed = {c: c[len(prefix):] for c in cols}
    return (
        df.select([id_col] + cols)
        .rename({id_col: "city_code", **renamed})
        .unique(subset="city_code", keep="first")
    )


all_nodes = (
    pl.concat([
        build_node_frame(X_train, "src_"), build_node_frame(X_train, "dst_"),
        build_node_frame(X_test, "src_"), build_node_frame(X_test, "dst_"),
        build_node_frame(X_inference, "src_"), build_node_frame(X_inference, "dst_"),
    ])
    .unique(subset="city_code", keep="first")
    .sort("city_code")
)

feat_cols = [c for c in all_nodes.columns if c != "city_code" and c not in DROP_NODE_FEATURES]
all_nodes = all_nodes.select(["city_code"] + feat_cols)

col_means = all_nodes.select(feat_cols).mean()
all_nodes = all_nodes.with_columns([pl.col(c).fill_null(col_means[c][0]) for c in feat_cols])

city_codes = all_nodes["city_code"].to_list()
city_to_idx = {code: i for i, code in enumerate(city_codes)}
num_nodes = len(city_codes)

income_keywords = ["gdp"]
income_idx = [i for i, c in enumerate(feat_cols) if any(k in c.lower() for k in income_keywords)]
climate_keywords = ["_temp", "precipitation", "ndvi", "uv_", "humid", "wind_mean", "wet_bulb", "degree_day"]
climate_idx = [i for i, c in enumerate(feat_cols) if any(k in c.lower() for k in climate_keywords)]
climate_names = [feat_cols[i] for i in climate_idx]
gravity_edge_idx = [i for i, c in enumerate(dyadic_cols) if c in ("pop_ratio", "distance_km")]

node_scaler = StandardScaler()
node_features = node_scaler.fit_transform(all_nodes.select(feat_cols).to_numpy().astype(np.float32))

edge_scaler = StandardScaler()
X_train_edge = edge_scaler.fit_transform(X_train.select(dyadic_cols).to_numpy().astype(np.float32))
X_test_edge = edge_scaler.transform(X_test.select(dyadic_cols).to_numpy().astype(np.float32))
X_infer_edge = edge_scaler.transform(X_inference.select(dyadic_cols).to_numpy().astype(np.float32))

state_from_code = (all_nodes["city_code"] // 100000).to_numpy().astype(np.int64)

centroid_cache = "data/muni_centroids.csv"
if os.path.exists(centroid_cache):
    cent = pl.read_csv(centroid_cache)
    code_to_latlon = {int(c): (float(la), float(lo))
                      for c, la, lo in zip(cent["code_muni"], cent["lat"], cent["lon"])}
    muni = geobr.read_municipality(year=2010, simplified=True, verbose=False)
else:
    muni = geobr.read_municipality(year=2010, simplified=True, verbose=False)
    code_to_latlon = {
        int(row["code_muni"]): (float(row.geometry.centroid.y), float(row.geometry.centroid.x))
        for _, row in muni.iterrows()
    }

node_lat = np.array([code_to_latlon.get(c, (np.nan, np.nan))[0] for c in city_codes])
node_lon = np.array([code_to_latlon.get(c, (np.nan, np.nan))[1] for c in city_codes])
has_coords = ~np.isnan(node_lat)
for i in np.where(~has_coords)[0]:
    peers = np.where((state_from_code == state_from_code[i]) & has_coords)[0]
    if len(peers):
        node_lat[i], node_lon[i] = node_lat[peers].mean(), node_lon[peers].mean()
    else:
        node_lat[i], node_lon[i] = -15.78, -47.93

K = 5
w = KNN.from_array(np.stack([node_lon, node_lat], axis=1), k=K)
w.transform = "R"
src_list, dst_list, wt_list = [], [], []
for i, neighbors in w.neighbors.items():
    for j, wij in zip(neighbors, w.weights[i]):
        src_list.append(j); dst_list.append(i); wt_list.append(wij)
adj_edge_index = torch.tensor(np.array([src_list, dst_list]), dtype=torch.long, device=device)
adj_weights = torch.tensor(np.array(wt_list), dtype=torch.float, device=device)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim),
        )
    def forward(self, x):
        return x + self.net(x)


class SpatialGatedDG(nn.Module):
    """Two income-conditioned climate gates: one used when a city plays the
    source role in an edge, one used when it plays the destination role.
    Trained with an L1 penalty on the effective FiLM multiplier |1+gamma|
    (and |beta|), so a closed gate (climate blocked) is the default and the
    model only opens a gate where doing so improves flow prediction enough
    to pay for it. See model.py for the training-time loss term."""

    def __init__(self, node_dim, edge_dim, income_idx, climate_idx, gravity_edge_idx,
                 hidden=4096, out=256, heads=4, dropout=0.4):
        super().__init__()
        H, O, D = hidden, out, dropout

        self.register_buffer("income_idx",      torch.tensor(income_idx, dtype=torch.long))
        self.register_buffer("climate_idx",      torch.tensor(climate_idx, dtype=torch.long))
        self.register_buffer("gravity_edge_idx", torch.tensor(gravity_edge_idx, dtype=torch.long))

        def make_gate_net():
            return nn.Sequential(
                nn.Linear(len(income_idx), 64), nn.GELU(), nn.LayerNorm(64),
                nn.Linear(64, len(climate_idx) * 2)
            )
        self.income_gate_net_src = make_gate_net()
        self.income_gate_net_dst = make_gate_net()

        self.conv1 = GCNConv(node_dim, H)
        self.res1  = nn.Linear(node_dim, H, bias=False)
        self.norm1 = nn.LayerNorm(H)

        self.conv2 = GATConv(H, H // heads, heads=heads, dropout=D, concat=True)
        self.norm2 = nn.LayerNorm(H)

        self.conv3 = GATConv(H, O, heads=1, concat=False, dropout=D)
        self.res3  = nn.Linear(H, O, bias=False)
        self.norm3 = nn.LayerNorm(O)

        self.src_proj = nn.Linear(O, O)
        self.dst_proj = nn.Linear(O, O)

        edge_proj_dim = 64
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, edge_proj_dim), nn.GELU(), nn.LayerNorm(edge_proj_dim),
            nn.Linear(edge_proj_dim, edge_proj_dim), nn.GELU()
        )

        self.gravity_skip = nn.Linear(len(gravity_edge_idx), 1, bias=True)

        dec_in = O * 3 + edge_proj_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(D),
            ResidualBlock(512, D),
            nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(D),
            ResidualBlock(256, D / 2),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(D / 2),
            nn.Linear(128, 1)
        )
        self.decoder_skip = nn.Linear(dec_in, 1)

        self.recon_self = nn.Sequential(nn.Linear(O, H), nn.GELU(), nn.Linear(H, node_dim))
        self.recon_lag  = nn.Sequential(nn.Linear(O, H), nn.GELU(), nn.Linear(H, node_dim))

    def gate_logits(self, x, role):
        net = self.income_gate_net_src if role == "src" else self.income_gate_net_dst
        return net(x[:, self.income_idx])

    def get_film_params(self, x, role):
        net = self.income_gate_net_src if role == "src" else self.income_gate_net_dst
        params = net(x[:, self.income_idx])
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return gamma, beta

    def encode(self, x, edge_index, role):
        x_in = x.clone()
        gamma, beta = self.get_film_params(x, role)
        climate_feats = x[:, self.climate_idx]
        x_in[:, self.climate_idx] = (1.0 + gamma) * climate_feats + beta
        h = self.norm1(F.elu(self.conv1(x_in, edge_index)) + self.res1(x_in))
        h = self.norm2(F.elu(self.conv2(h, edge_index)) + h)
        h = self.norm3(F.elu(self.conv3(h, edge_index)) + self.res3(h))
        return h

    def decode(self, h_src, h_dst, edge_attr):
        s, d = self.src_proj(h_src), self.dst_proj(h_dst)
        edge_proj = self.edge_proj(edge_attr)
        dec_in  = torch.cat([s, d, s * d, edge_proj], dim=1)
        gravity = self.gravity_skip(edge_attr[:, self.gravity_edge_idx]).squeeze(-1)
        out = self.decoder(dec_in).squeeze(-1) + self.decoder_skip(dec_in).squeeze(-1) + gravity
        return F.softplus(out)


model = SpatialGatedDG(
    node_dim=node_features.shape[1], edge_dim=len(dyadic_cols),
    income_idx=income_idx, climate_idx=climate_idx, gravity_edge_idx=gravity_edge_idx,
    hidden=4096, out=1024, heads=8, dropout=0.3,
).to(device)

ckpt_path = f"the_gnn/best_sgdg_{args.name}.pt"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

x = torch.tensor(node_features, dtype=torch.float, device=device)


def to_node_idx(df, col):
    return torch.tensor([city_to_idx[c] for c in df[col].to_list()], dtype=torch.long, device=device)


src_train, dst_train = to_node_idx(X_train, "source_code"), to_node_idx(X_train, "dest_code")
ea_train = torch.tensor(X_train_edge, dtype=torch.float, device=device)
src_test, dst_test = to_node_idx(X_test, "source_code"), to_node_idx(X_test, "dest_code")
ea_test = torch.tensor(X_test_edge, dtype=torch.float, device=device)
src_infer, dst_infer = to_node_idx(X_inference, "source_code"), to_node_idx(X_inference, "dest_code")
ea_infer = torch.tensor(X_infer_edge, dtype=torch.float, device=device)

with torch.no_grad():
    h_src = model.encode(x, adj_edge_index, "src")
    h_dst = model.encode(x, adj_edge_index, "dst")

    pred_train = model.decode(h_src[src_train], h_dst[dst_train], ea_train).cpu().numpy()
    pred_test = model.decode(h_src[src_test], h_dst[dst_test], ea_test).cpu().numpy()
    pred_infer = model.decode(h_src[src_infer], h_dst[dst_infer], ea_infer).cpu().numpy()

    gate_src_all, _ = model.get_film_params(x, "src")
    gate_dst_all, _ = model.get_film_params(x, "dst")
    gate_src_all = gate_src_all.cpu().numpy()
    gate_dst_all = gate_dst_all.cpu().numpy()

gate_mean_src = gate_src_all.mean(axis=1)
gate_mean_dst = gate_dst_all.mean(axis=1)

mult_src_all = np.abs(1.0 + gate_src_all)
mult_dst_all = np.abs(1.0 + gate_dst_all)
mult_mean_src = mult_src_all.mean(axis=1)
mult_mean_dst = mult_dst_all.mean(axis=1)

muni["code_muni"] = muni["code_muni"].astype(int)


def save_choropleth(values_df, value_col, title, filename, cmap="Reds",
                     center_zero=False, log_scale=False, log_floor_percentile=5):
    merged = muni.merge(values_df.to_pandas(), on="code_muni", how="left")
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_kwargs = dict(column=value_col, cmap=cmap, legend=True, ax=ax,
                        missing_kwds={"color": "lightgrey", "label": "no data"})

    if log_scale:
        vals = merged[value_col].to_numpy()
        positive = vals[np.isfinite(vals) & (vals > 0)]
        if len(positive) == 0:
            print(f"skipped {filename}: no positive values available for log scale")
            plt.close(fig)
            return
        vmin = max(np.percentile(positive, log_floor_percentile), 1e-6)
        vmax = positive.max()
        dropped = int(((vals <= 0) | ~np.isfinite(vals)).sum())
        if dropped:
            print(f"{filename}: {dropped} cities with value <= 0 shown as missing under log scale")
        merged.loc[(merged[value_col] <= 0) | (~np.isfinite(merged[value_col])), value_col] = np.nan
        plot_kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)
    elif center_zero:
        vmax = np.nanmax(np.abs(merged[value_col]))
        plot_kwargs.update(vmin=-vmax, vmax=vmax)

    merged.plot(**plot_kwargs)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(f"the_gnn/maps/{filename}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved maps/{filename}")


TRAIN_CMAP, TEST_INFER_CMAP = "PuOr", "RdBu_r"

out_train = (
    pl.DataFrame({"code_muni": X_train["source_code"].to_numpy(), "pred_flow": pred_train})
    .group_by("code_muni").agg(pl.col("pred_flow").sum())
)
out_test_infer = (
    pl.DataFrame({
        "code_muni": np.concatenate([X_test["source_code"].to_numpy(), X_inference["source_code"].to_numpy()]),
        "pred_flow": np.concatenate([pred_test, pred_infer]),
    })
    .group_by("code_muni").agg(pl.col("pred_flow").sum())
)
save_choropleth(out_train, "pred_flow", "Predicted total out-migration pressure (train set)",
                 "out_migration_pressure_train.png", cmap=TRAIN_CMAP)
save_choropleth(out_train, "pred_flow", "Predicted total out-migration pressure (train set, log scale)",
                 "out_migration_pressure_train_log.png", cmap=TRAIN_CMAP, log_scale=True)
save_choropleth(out_test_infer, "pred_flow", "Predicted total out-migration pressure (test + inference)",
                 "out_migration_pressure_test_infer.png", cmap=TEST_INFER_CMAP)
save_choropleth(out_test_infer, "pred_flow", "Predicted total out-migration pressure (test + inference, log scale)",
                 "out_migration_pressure_test_infer_log.png", cmap=TEST_INFER_CMAP, log_scale=True)


def compute_net_migration(df, pred):
    out_p = (pl.DataFrame({"code_muni": df["source_code"].to_numpy(), "pred_flow": pred})
             .group_by("code_muni").agg(pl.col("pred_flow").sum().alias("out_flow")))
    in_p = (pl.DataFrame({"code_muni": df["dest_code"].to_numpy(), "pred_flow": pred})
            .group_by("code_muni").agg(pl.col("pred_flow").sum().alias("in_flow")))
    return (out_p.join(in_p, on="code_muni", how="outer")
            .fill_null(0.0)
            .with_columns((pl.col("in_flow") - pl.col("out_flow")).alias("net_migration")))


net_train = compute_net_migration(X_train, pred_train)
combined_pairs = pl.concat([
    X_test.select(["source_code", "dest_code"]),
    X_inference.select(["source_code", "dest_code"]),
])
combined_pred = np.concatenate([pred_test, pred_infer])
net_test_infer = compute_net_migration(combined_pairs, combined_pred)
save_choropleth(net_train, "net_migration", "Predicted net migration (train set, inflow - outflow)",
                 "net_migration_train.png", cmap=TRAIN_CMAP, center_zero=True)
save_choropleth(net_test_infer, "net_migration", "Predicted net migration (test + inference, inflow - outflow)",
                 "net_migration_test_infer.png", cmap=TEST_INFER_CMAP, center_zero=True)

gate_df = pl.DataFrame({
    "code_muni": city_codes,
    "gate_src": gate_mean_src,
    "gate_dst": gate_mean_dst,
    "gate_asymmetry": gate_mean_src - gate_mean_dst,
})
save_choropleth(gate_df, "gate_src", "Climate-signal gate activation, source role (higher = climate more influential)",
                 "gate_activation_src.png", cmap="PuOr")
save_choropleth(gate_df, "gate_dst", "Climate-signal gate activation, destination role (higher = climate more influential)",
                 "gate_activation_dst.png", cmap="PuOr")
save_choropleth(gate_df, "gate_asymmetry",
                 "Gate asymmetry: climate weighted more as origin (+) vs as destination (-)",
                 "gate_activation_asymmetry.png", cmap="RdBu_r", center_zero=True)


def load_actual_flow(df, split_name):
    if "flow" in df.columns:
        return df["flow"].to_numpy()
    return pl.read_csv(f"data/y_{split_name}.csv")["flow"].to_numpy()


y_train_actual = load_actual_flow(X_train, "train")
y_test_actual = load_actual_flow(X_test, "test")
y_infer_actual = load_actual_flow(X_inference, "inference")

resid_train = pred_train - y_train_actual
resid_test = pred_test - y_test_actual
resid_infer = pred_infer - y_infer_actual

resid_train_df = (
    pl.DataFrame({"code_muni": X_train["source_code"].to_numpy(), "resid": resid_train})
    .group_by("code_muni").agg(pl.col("resid").mean())
)
resid_test_infer_df = (
    pl.DataFrame({
        "code_muni": np.concatenate([X_test["source_code"].to_numpy(), X_inference["source_code"].to_numpy()]),
        "resid": np.concatenate([resid_test, resid_infer]),
    })
    .group_by("code_muni").agg(pl.col("resid").mean())
)
save_choropleth(resid_train_df, "resid", "Mean prediction residual by origin city (train set)",
                 "residuals_train.png", cmap=TRAIN_CMAP, center_zero=True)
save_choropleth(resid_test_infer_df, "resid", "Mean prediction residual by origin city (test + inference)",
                 "residuals_test_infer.png", cmap=TEST_INFER_CMAP, center_zero=True)

income_col = next((c for c in feat_cols if "gdp_per_capita" in c.lower()), None)
if income_col:
    income_df = pl.DataFrame({"code_muni": city_codes, "income": all_nodes[income_col].to_numpy()})
    save_choropleth(income_df, "income", "GDP per capita by municipality",
                     "income_reference.png", cmap="Greens")
    save_choropleth(income_df, "income", "GDP per capita by municipality (log scale)",
                     "income_reference_log.png", cmap="Greens", log_scale=True)

os.makedirs("the_gnn/maps/climate_gates", exist_ok=True)
for ci, name in enumerate(climate_names):
    feat_df = pl.DataFrame({
        "code_muni": city_codes,
        "gate_src": gate_src_all[:, ci],
        "gate_dst": gate_dst_all[:, ci],
    })
    save_choropleth(feat_df, "gate_src", f"Gate activation for {name} (source role)",
                     f"climate_gates/gate_src_{name}.png", cmap="PuOr")
    save_choropleth(feat_df, "gate_dst", f"Gate activation for {name} (destination role)",
                     f"climate_gates/gate_dst_{name}.png", cmap="PuOr")

gravity_design_train = np.column_stack([
    np.ones(len(X_train)), X_train.select(["distance_km", "pop_ratio"]).to_numpy().astype(np.float64),
])
gravity_coef, *_ = np.linalg.lstsq(gravity_design_train, y_train_actual.astype(np.float64), rcond=None)

gravity_design_test = np.column_stack([
    np.ones(len(X_test)), X_test.select(["distance_km", "pop_ratio"]).to_numpy().astype(np.float64),
])
gravity_pred_test = gravity_design_test @ gravity_coef
gravity_resid_test = y_test_actual.astype(np.float64) - gravity_pred_test

improvement_df = (
    pl.DataFrame({
        "code_muni": X_test["source_code"].to_numpy(),
        "abs_gravity_resid": np.abs(gravity_resid_test),
        "abs_model_resid": np.abs(resid_test),
    })
    .group_by("code_muni")
    .agg([pl.col("abs_gravity_resid").mean(), pl.col("abs_model_resid").mean()])
    .with_columns((pl.col("abs_gravity_resid") - pl.col("abs_model_resid")).alias("gnn_improvement"))
)
save_choropleth(improvement_df, "gnn_improvement",
                 "Error reduction vs gravity baseline (+ = GNN more accurate)",
                 "gnn_vs_gravity_improvement.png", cmap="RdBu_r", center_zero=True)

os.makedirs("the_gnn/maps/mechanism", exist_ok=True)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "figure.dpi": 200,
})

raw_income = all_nodes[income_col].to_numpy() if income_col else None


def binned_trend(x_vals, y_vals, n_bins=20):
    """Equal-count income bins with mean +/- SEM per bin, for a clean trend
    line over noisy per-city scatter."""
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals, y_vals = x_vals[valid], y_vals[valid]
    order = np.argsort(x_vals)
    x_sorted, y_sorted = x_vals[order], y_vals[order]
    edges = np.array_split(np.arange(len(x_sorted)), n_bins)
    bin_x, bin_y, bin_se = [], [], []
    for idx in edges:
        if len(idx) == 0:
            continue
        bin_x.append(x_sorted[idx].mean())
        bin_y.append(y_sorted[idx].mean())
        bin_se.append(y_sorted[idx].std(ddof=1) / max(np.sqrt(len(idx)), 1))
    return np.array(bin_x), np.array(bin_y), np.array(bin_se)


def income_vs_gate_plot(income, mult_src, mult_dst, filename,
                         title="Climate signal passed through the income gate",
                         log_x=True):
    fig, ax = plt.subplots(figsize=(8, 6))

    valid = np.isfinite(income) & (income > 0 if log_x else np.isfinite(income))
    inc = income[valid]

    for mult, label, color in [(mult_src[valid], "Origin role", "#c0522d"),
                                (mult_dst[valid], "Destination role", "#2d6ac0")]:
        ax.scatter(inc, mult, s=10, alpha=0.15, color=color, linewidths=0)
        bx, by, bse = binned_trend(inc, mult, n_bins=20)
        ax.plot(bx, by, color=color, linewidth=2.4, label=label)
        ax.fill_between(bx, by - 1.96 * bse, by + 1.96 * bse, color=color, alpha=0.18)

    if log_x:
        ax.set_xscale("log")
    ax.axhline(1.0, color="#888888", linewidth=1, linestyle="--", zorder=0)
    ax.text(ax.get_xlim()[1], 1.02, "fully open (=1)", ha="right", va="bottom",
            color="#888888", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("City income (GDP per capita, log scale)" if log_x else "City income (GDP per capita)")
    ax.set_ylabel(r"Effective climate gate multiplier  $|1+\gamma|$")
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.15, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(f"the_gnn/maps/mechanism/{filename}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved maps/mechanism/{filename}")


if raw_income is not None:
    income_vs_gate_plot(raw_income, mult_mean_src, mult_mean_dst,
                         "income_vs_gate_openness.png")

    n_feat = len(climate_names)
    ncols = 3
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    valid = np.isfinite(raw_income) & (raw_income > 0)
    for ci, name in enumerate(climate_names):
        ax = axes[ci]
        inc = raw_income[valid]
        m_src = mult_src_all[valid, ci]
        m_dst = mult_dst_all[valid, ci]
        bx_s, by_s, _ = binned_trend(inc, m_src, n_bins=15)
        bx_d, by_d, _ = binned_trend(inc, m_dst, n_bins=15)
        ax.plot(bx_s, by_s, color="#c0522d", linewidth=1.8, label="Origin")
        ax.plot(bx_d, by_d, color="#2d6ac0", linewidth=1.8, label="Destination")
        ax.axhline(1.0, color="#aaaaaa", linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_title(name, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.15, linewidth=0.5)
    for ax in axes[n_feat:]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Gate multiplier vs. income, by climate feature", fontsize=14, y=1.01)
    fig.text(0.5, -0.01, "City income (GDP per capita, log scale)", ha="center")
    fig.text(-0.01, 0.5, r"Gate multiplier $|1+\gamma|$", va="center", rotation="vertical")
    fig.tight_layout()
    fig.savefig("the_gnn/maps/mechanism/income_vs_gate_by_feature.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved maps/mechanism/income_vs_gate_by_feature.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    valid = np.isfinite(raw_income) & (raw_income > 0)
    inc = raw_income[valid]
    asym = mult_mean_src[valid] - mult_mean_dst[valid]
    ax.scatter(inc, asym, s=10, alpha=0.15, color="#555555", linewidths=0)
    bx, by, bse = binned_trend(inc, asym, n_bins=20)
    ax.plot(bx, by, color="#5b2d8c", linewidth=2.4)
    ax.fill_between(bx, by - 1.96 * bse, by + 1.96 * bse, color="#5b2d8c", alpha=0.18)
    ax.axhline(0.0, color="#888888", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("City income (GDP per capita, log scale)")
    ax.set_ylabel("Gate asymmetry (origin multiplier - destination multiplier)")
    ax.set_title("Does the origin/destination climate-sensitivity gap vary with income?", fontsize=13, pad=12)
    ax.grid(alpha=0.15, linewidth=0.6)
    fig.tight_layout()
    fig.savefig("the_gnn/maps/mechanism/income_vs_gate_asymmetry.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved maps/mechanism/income_vs_gate_asymmetry.png")

    q1, q2 = np.nanquantile(raw_income[valid], [1 / 3, 2 / 3])
    tertile_labels = np.select(
        [raw_income <= q1, (raw_income > q1) & (raw_income <= q2), raw_income > q2],
        ["Low income", "Mid income", "High income"], default="NA"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"Low income": "#c0522d", "Mid income": "#c9a13b", "High income": "#2d6ac0"}
    for ax, mult, role in [(axes[0], mult_mean_src, "Origin role"), (axes[1], mult_mean_dst, "Destination role")]:
        for label, color in colors.items():
            mask = (tertile_labels == label) & np.isfinite(mult)
            ax.hist(mult[mask], bins=30, alpha=0.55, color=color, label=label, density=True)
        ax.axvline(1.0, color="#888888", linewidth=1, linestyle="--")
        ax.set_title(role, fontsize=12)
        ax.set_xlabel(r"Gate multiplier $|1+\gamma|$")
        ax.grid(alpha=0.15, linewidth=0.5)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Distribution of climate gate openness by income tertile", fontsize=14)
    fig.tight_layout()
    fig.savefig("the_gnn/maps/mechanism/gate_distribution_by_tertile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved maps/mechanism/gate_distribution_by_tertile.png")
else:
    print("no GDP-per-capita column found in feat_cols -- skipping income x gate figures")

def _get_state_basemap():
    states = geobr.read_state(year=2010, simplified=True, verbose=False)
    return states


def save_flow_map(
    edges_df,
    code_to_latlon,
    value_col,
    title,
    filename,
    top_n=150,
    cmap="viridis",
    curvature=0.15,
    basemap=None,
    node_values=None,
    node_col=None,
    node_cmap="Greys",
    node_size_range=(6, 60),
    figsize=(11, 11),
    min_linewidth=0.4,
    max_linewidth=4.0,
    min_alpha=0.15,
    max_alpha=0.9,
):
    if hasattr(edges_df, "to_pandas"):
        edges_df = edges_df.to_pandas()

    edges_df = edges_df.copy()
    edges_df["abs_val"] = edges_df[value_col].abs()
    edges_df = edges_df.sort_values("abs_val", ascending=False).head(top_n)

    lat0 = edges_df["source_code"].map(lambda c: code_to_latlon.get(int(c), (np.nan, np.nan))[0])
    lon0 = edges_df["source_code"].map(lambda c: code_to_latlon.get(int(c), (np.nan, np.nan))[1])
    lat1 = edges_df["dest_code"].map(lambda c: code_to_latlon.get(int(c), (np.nan, np.nan))[0])
    lon1 = edges_df["dest_code"].map(lambda c: code_to_latlon.get(int(c), (np.nan, np.nan))[1])
    keep = lat0.notna() & lat1.notna()
    edges_df = edges_df[keep]
    lat0, lon0, lat1, lon1 = lat0[keep], lon0[keep], lat1[keep], lon1[keep]

    if len(edges_df) == 0:
        print(f"skipped {filename}: no plottable edges (missing centroids)")
        return

    if basemap is None:
        basemap = _get_state_basemap()

    fig, ax = plt.subplots(figsize=figsize)
    basemap.plot(ax=ax, color="#f2f2f0", edgecolor="#bbbbbb", linewidth=0.6, zorder=0)

    vals = edges_df[value_col].to_numpy()
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    abs_vals = edges_df["abs_val"].to_numpy()
    amin, amax = abs_vals.min(), abs_vals.max()
    denom = (amax - amin) if amax > amin else 1.0
    lws = min_linewidth + (abs_vals - amin) / denom * (max_linewidth - min_linewidth)
    alphas = min_alpha + (abs_vals - amin) / denom * (max_alpha - min_alpha)

    order = np.argsort(abs_vals)
    for idx in order:
        row_lat0, row_lon0 = lat0.iloc[idx], lon0.iloc[idx]
        row_lat1, row_lon1 = lat1.iloc[idx], lon1.iloc[idx]
        color = colormap(norm(vals[idx]))
        arrow = FancyArrowPatch(
            (row_lon0, row_lat0), (row_lon1, row_lat1),
            connectionstyle=f"arc3,rad={curvature}",
            arrowstyle="-|>", mutation_scale=8,
            color=color, linewidth=lws[idx], alpha=alphas[idx], zorder=2,
        )
        ax.add_patch(arrow)

    if node_values is not None and node_col is not None:
        if hasattr(node_values, "to_pandas"):
            node_values = node_values.to_pandas()
        n_lat = node_values["code_muni"].map(lambda c: code_to_latlon.get(int(c), (np.nan, np.nan))[0])
        n_lon = node_values["code_muni"].map(lambda c: code_to_latlon.get(int(c), (np.nan, np.nan))[1])
        n_val = node_values[node_col].to_numpy()
        n_keep = n_lat.notna()
        n_lat, n_lon, n_val = n_lat[n_keep], n_lon[n_keep], n_val[n_keep]

        nabs = np.abs(n_val)
        nmin, nmax = nabs.min(), nabs.max()
        ndenom = (nmax - nmin) if nmax > nmin else 1.0
        sizes = node_size_range[0] + (nabs - nmin) / ndenom * (node_size_range[1] - node_size_range[0])

        sc = ax.scatter(
            n_lon, n_lat, s=sizes, c=n_val, cmap=node_cmap,
            edgecolor="black", linewidth=0.3, zorder=3,
        )
        fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.02, label=node_col)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.06, label=value_col)

    ax.set_title(title)
    ax.set_xlim(basemap.total_bounds[[0, 2]])
    ax.set_ylim(basemap.total_bounds[[1, 3]])
    ax.axis("off")
    fig.savefig(f"the_gnn/maps/{filename}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved maps/{filename}")


state_basemap = _get_state_basemap()

flow_edges = X_train.select(["source_code", "dest_code"]).with_columns(pl.Series("pred_flow", pred_train))
save_flow_map(flow_edges, code_to_latlon, value_col="pred_flow",
              title="Top predicted migration flows (train set)",
              filename="flow_map_top_predicted_train.png",
              top_n=150, cmap="viridis", basemap=state_basemap,
              node_values=net_train, node_col="net_migration", node_cmap="RdBu_r")

gate_edge_df = X_train.select(["source_code", "dest_code"]).with_columns(
    pl.Series("gate_strength", mult_mean_src[[city_to_idx[c] for c in X_train["source_code"]]])
)
save_flow_map(gate_edge_df, code_to_latlon, value_col="gate_strength",
              title="Predicted flows weighted by origin climate-gate openness",
              filename="flow_map_gate_weighted.png",
              top_n=150, cmap="PuOr", basemap=state_basemap)
