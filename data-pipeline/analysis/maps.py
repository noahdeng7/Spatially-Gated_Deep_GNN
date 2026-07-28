import argparse
import os
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from torch_geometric.nn import GCNConv, GATConv
from torch.nn.parallel import replicate, parallel_apply
from libpysal.weights import KNN
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

import geobr

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--income",  type=str, default="mean_income")
parser.add_argument("--n_perm", type=int, default=999)
parser.add_argument("--top_edges", type=int, default=3000)
args = parser.parse_args()

torch.manual_seed(42)
np.random.seed(42)
rng = np.random.default_rng(42)

INCOME_KEY = args.income

os.makedirs("the_gnn", exist_ok=True)
FIG_DIR = f"the_gnn/figs_{args.name}"
os.makedirs(FIG_DIR, exist_ok=True)

log_file = open(f"the_gnn/info_{args.name}_maps.txt", "w")
def log(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

plt.rcParams.update({
    "font.size": 19,
    "axes.titlesize": 19,
    "axes.labelsize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 19,
    "font.family": "serif",
})

def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def binned_line_plot(ax, xvals, yvals, n_bins=20, color="#2c7fb8", label=None):
    """Bin xvals into n_bins quantile bins, plot mean(yvals) per bin as a
    connected line whose segment thickness scales with the bin's sample size."""
    xvals = np.asarray(xvals, dtype=np.float64)
    yvals = np.asarray(yvals, dtype=np.float64)
    valid_mask = ~(np.isnan(xvals) | np.isnan(yvals))
    xv, yv = xvals[valid_mask], yvals[valid_mask]
    if len(xv) < 2:
        return

    nb = min(n_bins, max(2, len(xv) // 5))
    edges = np.quantile(xv, np.linspace(0, 1, nb + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        ax.scatter(xv, yv, s=14, alpha=0.35, color=color, edgecolor="none", label=label)
        return

    bin_idx = np.digitize(xv, edges[1:-1], right=True)

    centers, means, counts = [], [], []
    for b in range(len(edges) - 1):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        centers.append(xv[m].mean())
        means.append(yv[m].mean())
        counts.append(m.sum())

    centers, means, counts = np.array(centers), np.array(means), np.array(counts)
    if len(centers) < 2:
        ax.scatter(xv, yv, s=14, alpha=0.35, color=color, edgecolor="none", label=label)
        return

    counts_norm = counts / counts.max()
    lw_min, lw_max = 1.0, 8.0

    points = np.array([centers, means]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_widths = lw_min + (lw_max - lw_min) * (
        (counts_norm[:-1] + counts_norm[1:]) / 2
    )

    lc = LineCollection(segments, linewidths=seg_widths, colors=color, alpha=0.85, label=label)
    ax.add_collection(lc)
    ax.scatter(centers, means, s=(10 + 40 * counts_norm), color=color,
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlim(centers.min() - 0.02 * np.ptp(centers), centers.max() + 0.02 * np.ptp(centers))
    y_ptp = np.ptp(means) if np.ptp(means) > 0 else 1.0
    ax.set_ylim(means.min() - 0.1 * y_ptp, means.max() + 0.1 * y_ptp)


DROP_NODE_FEATURES = []
DROP_EDGE_FEATURES = []

X_train     = pl.read_csv("data/X_train.csv")
X_test      = pl.read_csv("data/X_test.csv")
X_inference = pl.read_csv("data/X_inference.csv")
Y_train     = pl.read_csv("data/y_train.csv")
Y_test      = pl.read_csv("data/y_test.csv")
Y_inference = pl.read_csv("data/y_inference.csv")

y_col = "flow" if "flow" in Y_train.columns else Y_train.columns[0]

id_cols = ["source_code", "dest_code"]
dyadic_cols = [c for c in X_train.columns
               if c not in id_cols and not c.startswith("src_") and not c.startswith("dst_")
               and c not in DROP_EDGE_FEATURES]

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

city_codes  = all_nodes["city_code"].to_list()
num_nodes   = len(city_codes)
city_map_df = pl.DataFrame({"city_code": city_codes, "node_idx": list(range(num_nodes))})

income_keywords  = [INCOME_KEY]   # HEREEEEE
climate_keywords = ["_temp", "temp", "_precip", "precip", "ndvi", "uv", "wind_mean", "wet_bulb", "degree_day"]
income_idx  = [i for i, c in enumerate(feat_cols) if any(k in c.lower() for k in income_keywords)]
climate_idx = [i for i, c in enumerate(feat_cols) if any(k in c.lower() for k in climate_keywords)]

node_scaler   = StandardScaler()
node_features = node_scaler.fit_transform(all_nodes.select(feat_cols).to_numpy().astype(np.float32))

for split_name, df in [("train", X_train), ("test", X_test), ("inference", X_inference)]:
    nulls = df.select(dyadic_cols).null_count()
    bad = {c: nulls[c][0] for c in dyadic_cols if nulls[c][0] > 0}
    if bad:
        raise ValueError(f"NaN in {split_name} dyadic columns {bad}; fix upstream in data.py")

edge_scaler  = StandardScaler()
X_train_edge = edge_scaler.fit_transform(X_train.select(dyadic_cols).to_numpy().astype(np.float32))
X_test_edge  = edge_scaler.transform(X_test.select(dyadic_cols).to_numpy().astype(np.float32))
X_infer_edge = edge_scaler.transform(X_inference.select(dyadic_cols).to_numpy().astype(np.float32))

state_from_code = (all_nodes["city_code"] // 100000).to_numpy().astype(np.int64)

centroid_cache = "data/muni_centroids.csv"
if os.path.exists(centroid_cache):
    cent = pl.read_csv(centroid_cache)
    code_to_latlon = {int(c): (float(la), float(lo))
                      for c, la, lo in zip(cent["code_muni"], cent["lat"], cent["lon"])}
else:
    muni_src = geobr.read_municipality(year=2010, simplified=True, verbose=False)
    code_to_latlon = {
        int(row["code_muni"]): (float(row.geometry.centroid.y), float(row.geometry.centroid.x))
        for _, row in muni_src.iterrows()
    }
    pl.DataFrame({"code_muni": list(code_to_latlon),
                  "lat": [v[0] for v in code_to_latlon.values()],
                  "lon": [v[1] for v in code_to_latlon.values()]}).write_csv(centroid_cache)

node_lat = np.array([code_to_latlon.get(c, (np.nan, np.nan))[0] for c in city_codes])
node_lon = np.array([code_to_latlon.get(c, (np.nan, np.nan))[1] for c in city_codes])
has_coords = ~np.isnan(node_lat)

for i in np.where(~has_coords)[0]:
    peers = np.where((state_from_code == state_from_code[i]) & has_coords)[0]
    if len(peers):
        node_lat[i], node_lon[i] = node_lat[peers].mean(), node_lon[peers].mean()
    else:
        node_lat[i], node_lon[i] = -15.78, -47.93
    has_coords[i] = True

K = 5
w = KNN.from_array(np.stack([node_lon, node_lat], axis=1), k=K)
w.transform = "R"
src_list, dst_list, wt_list = [], [], []
for i, neighbors in w.neighbors.items():
    for j, wij in zip(neighbors, w.weights[i]):
        src_list.append(j); dst_list.append(i); wt_list.append(wij)
adj_edge_index = torch.tensor(np.array([src_list, dst_list]), dtype=torch.long)
adj_weights    = torch.tensor(np.array(wt_list), dtype=torch.float)

gravity_edge_idx = [i for i, c in enumerate(dyadic_cols) if c in ("pop_ratio", "distance_km")]

x = torch.tensor(node_features, dtype=torch.float)
deg = torch.zeros(x.shape[0])
deg.index_add_(0, adj_edge_index[1], torch.ones(adj_edge_index.shape[1]))
Wx = torch.zeros_like(x)
Wx.index_add_(0, adj_edge_index[1], x[adj_edge_index[0]])
Wx = Wx / deg.clamp(min=1).unsqueeze(1)

def map_to_node_idx(df, col):
    return (df.select(col).rename({col: "city_code"})
              .join(city_map_df, on="city_code", how="left")["node_idx"].to_numpy())

def make_tensors(X, Y, X_edge):
    src = torch.tensor(map_to_node_idx(X, "source_code"), dtype=torch.long)
    dst = torch.tensor(map_to_node_idx(X, "dest_code"),   dtype=torch.long)
    ea  = torch.tensor(X_edge, dtype=torch.float)
    y   = torch.tensor(Y[y_col].to_numpy().astype(np.float32), dtype=torch.float)
    return src, dst, ea, y

src_train, dst_train, ea_train, y_train = make_tensors(X_train,     Y_train,     X_train_edge)
src_test,  dst_test,  ea_test,  y_test  = make_tensors(X_test,      Y_test,      X_test_edge)
src_infer, dst_infer, ea_infer, y_infer = make_tensors(X_inference, Y_inference, X_infer_edge)

log(f"nodes {x.shape[0]} | node_dim {x.shape[1]} | edge_dim {ea_train.shape[1]} | "
    f"train {src_train.shape[0]} | test {src_test.shape[0]} | infer {src_infer.shape[0]}")


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
    def __init__(self, node_dim, edge_dim, income_idx, climate_idx, gravity_edge_idx,
                 hidden=4096, out=256, heads=4, dropout=0.4):
        super().__init__()
        H, O, D = hidden, out, dropout

        self.register_buffer("income_idx", torch.tensor(income_idx, dtype=torch.long))
        self.register_buffer("climate_idx", torch.tensor(climate_idx, dtype=torch.long))
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

        assert H % heads == 0
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
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.norm2(F.elu(self.conv2(h, edge_index)) + h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.norm3(F.elu(self.conv3(h, edge_index)) + self.res3(h))
        return h

    def decode(self, h_src, h_dst, edge_attr):
        s, d = self.src_proj(h_src), self.dst_proj(h_dst)
        edge_proj = self.edge_proj(edge_attr)
        dec_in = torch.cat([s, d, s * d, edge_proj], dim=1)
        gravity = self.gravity_skip(edge_attr[:, self.gravity_edge_idx]).squeeze(-1)
        out = self.decoder(dec_in).squeeze(-1) + self.decoder_skip(dec_in).squeeze(-1) + gravity
        return F.softplus(out)

    def forward(self, x, edge_index, edge_attr, src_idx, dst_idx):
        h_src = self.encode(x, edge_index, "src")
        h_dst = self.encode(x, edge_index, "dst")
        return self.decode(h_src[src_idx], h_dst[dst_idx], edge_attr)


def cpc(pred, target):
    p, t = pred.detach().cpu().numpy(), target.detach().cpu().numpy()
    return 2 * np.sum(np.minimum(p, t)) / (np.sum(p) + np.sum(t) + 1e-8)

def rmse(pred, target):
    return torch.sqrt(((pred - target) ** 2).mean()).item()

def morans_i(values, edge_index, edge_weights):
    values = values.to(torch.float)
    v = values - values.mean()
    src, dst = edge_index[0], edge_index[1]
    num   = (edge_weights * v[src] * v[dst]).sum()
    S0    = edge_weights.sum() + 1e-8
    denom = (v ** 2).sum() + 1e-8
    return (values.shape[0] / S0 * num / denom).item()

def morans_i_test(values, edge_index, edge_weights, n_perm=999, seed=42):
    obs = morans_i(values, edge_index, edge_weights)
    rng_local = np.random.default_rng(seed)
    v_np = values.cpu().numpy()
    n = v_np.shape[0]
    edge_index_cpu, edge_weights_cpu = edge_index.cpu(), edge_weights.cpu()
    perm_stats = np.empty(n_perm)
    for p in range(n_perm):
        shuffled = torch.tensor(v_np[rng_local.permutation(n)], dtype=torch.float)
        perm_stats[p] = morans_i(shuffled, edge_index_cpu, edge_weights_cpu)
    z = (obs - perm_stats.mean()) / (perm_stats.std() + 1e-8)
    p_value = (np.sum(np.abs(perm_stats) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, z, p_value

devices   = ([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
             if args.gpu and torch.cuda.is_available() else [torch.device("cpu")])
dev       = devices[0]
log(f"devices: {[str(d) for d in devices]}")

model = SpatialGatedDG(
    node_dim=x.shape[1], edge_dim=ea_train.shape[1],
    income_idx=income_idx, climate_idx=climate_idx, gravity_edge_idx=gravity_edge_idx,
    hidden=4096, out=1024, heads=8, dropout=0.3
).to(dev)
log(f"parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

def to_dev(*tensors): return [t.to(dev) for t in tensors]
x, Wx, adj_edge_index, adj_weights = to_dev(x, Wx, adj_edge_index, adj_weights)
src_train, dst_train, ea_train, y_train = to_dev(src_train, dst_train, ea_train, y_train)
src_test,  dst_test,  ea_test,  y_test  = to_dev(src_test,  dst_test,  ea_test,  y_test)
src_infer, dst_infer, ea_infer, y_infer = to_dev(src_infer, dst_infer, ea_infer, y_infer)

ckpt_path = f"the_gnn/best_sgdg_{args.name}.pt"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"no checkpoint found at {ckpt_path}; this script does not train a model")

log(f"loading checkpoint from {ckpt_path}")
model.load_state_dict(torch.load(ckpt_path, map_location=dev))
model.eval()

with torch.no_grad():
    h_src_final = model.encode(x, adj_edge_index, "src")
    h_dst_final = model.encode(x, adj_edge_index, "dst")
    pred_test  = model.decode(h_src_final[src_test],  h_dst_final[dst_test],  ea_test)
    pred_infer = model.decode(h_src_final[src_infer], h_dst_final[dst_infer], ea_infer)

    test_cpc, test_rmse   = cpc(pred_test, y_test),   rmse(pred_test, y_test)
    infer_cpc, infer_rmse = cpc(pred_infer, y_infer), rmse(pred_infer, y_infer)

    lag_r2  = 1 - (0.5 * (F.mse_loss(model.recon_lag(h_src_final),  Wx).item()
                          + F.mse_loss(model.recon_lag(h_dst_final), Wx).item())) / Wx.var().item()
    self_r2 = 1 - (0.5 * (F.mse_loss(model.recon_self(h_src_final), x).item()
                          + F.mse_loss(model.recon_self(h_dst_final), x).item())) / x.var().item()

    gamma_src, beta_src = model.get_film_params(x, "src")
    gamma_dst, beta_dst = model.get_film_params(x, "dst")

    mean_gamma_src = gamma_src.cpu().numpy()
    mean_beta_src  = beta_src.cpu().numpy()
    mean_gamma_dst = gamma_dst.cpu().numpy()
    mean_beta_dst  = beta_dst.cpu().numpy()

log("=" * 60)
log(f"test  CPC {test_cpc:.4f}  RMSE {test_rmse:.2f}")
log(f"infer CPC {infer_cpc:.4f}  RMSE {infer_rmse:.2f}")
log(f"lag reconstruction R2 {lag_r2:.4f}  |  self reconstruction R2 {self_r2:.4f}")
log("=" * 60)

income_col_idx = next((i for i, c in enumerate(feat_cols) if "top_pop" in c), None)
raw_income = None
valid = None
q1 = q2 = None
if income_col_idx is not None:
    raw_income = all_nodes[feat_cols[income_col_idx]].to_numpy()
    valid = ~np.isnan(raw_income)
    q1, q2 = np.nanquantile(raw_income, [1/3, 2/3])

for role, (gamma_val, beta_val) in [("src", (mean_gamma_src, mean_beta_src)),
                                    ("dst", (mean_gamma_dst, mean_beta_dst))]:
    gmean, gstd = float(gamma_val.mean()), float(gamma_val.std())
    bmean, bstd = float(beta_val.mean()), float(beta_val.std())
    log(f"FiLM [{role}] -> Gamma mean {gmean:.4f} (std {gstd:.4f}) | Beta mean {bmean:.4f} (std {bstd:.4f})")
    if raw_income is not None:
        corr, pval = spearmanr(raw_income[valid], gamma_val.mean(axis=1)[valid])
        log(f"  gamma[{role}] ~ income spearman r={corr:.4f} p={pval:.3e}")
        for label, mask in [("low", raw_income <= q1),
                            ("mid", (raw_income > q1) & (raw_income <= q2)),
                            ("high", raw_income > q2)]:
            log(f"    gamma[{role}] by income tertile [{label:4s}]: {gamma_val[mask].mean():.4f}  n={mask.sum()}")

log("=" * 60)

for role, gamma_val in [("src", mean_gamma_src), ("dst", mean_gamma_dst)]:
    node_gamma_summary = gamma_val.mean(axis=1)
    i_gate, z_gate, p_gate = morans_i_test(
        torch.tensor(node_gamma_summary, dtype=torch.float, device=dev), adj_edge_index, adj_weights, args.n_perm, 42)
    log(f"  gamma_mean[{role}] activations   I={i_gate:+.4f}  z={z_gate:+.2f}  p={p_gate:.4f}")

if raw_income is not None:
    i_inc, z_inc, p_inc = morans_i_test(
        torch.tensor(raw_income, dtype=torch.float, device=dev), adj_edge_index, adj_weights, args.n_perm, 42)
    log(f"  raw income                I={i_inc:+.4f}  z={z_inc:+.2f}  p={p_inc:.4f}")

def node_mean_residual(pred, y, src_idx):
    resid = (pred - y).detach().cpu().numpy()
    df = pl.DataFrame({"node": src_idx.detach().cpu().numpy(), "resid": resid}).group_by("node").agg(pl.col("resid").mean())
    arr  = np.zeros(num_nodes, dtype=np.float32)
    mask = np.zeros(num_nodes, dtype=bool)
    nodes, resids = df["node"].to_numpy(), df["resid"].to_numpy().astype(np.float32)
    arr[nodes], mask[nodes] = resids, True
    return arr, mask

def moran_on_subgraph(values, mask, n_perm, seed):
    edge_np  = adj_edge_index.cpu().numpy()
    sub_mask = mask[edge_np[0]] & mask[edge_np[1]]
    if mask.sum() < 10 or sub_mask.sum() < 10:
        return None
    sub_idx = torch.tensor(sub_mask, device=dev)
    return morans_i_test(torch.tensor(values, dtype=torch.float, device=dev),
                         adj_edge_index[:, sub_idx], adj_weights[sub_idx], n_perm, seed)

model_resid, model_mask = node_mean_residual(pred_test, y_test, src_test)
result = moran_on_subgraph(model_resid, model_mask, args.n_perm, 42)
if result:
    i_m, z_m, p_m = result
    log(f"  model residual / node     I={i_m:+.4f}  z={z_m:+.2f}  p={p_m:.4f}  n={int(model_mask.sum())}")

log("=" * 60)

if raw_income is not None:
    src_test_income = raw_income[src_test.cpu().numpy()]
    groups = {
        "low":  src_test_income <= q1,
        "mid":  (src_test_income > q1) & (src_test_income <= q2),
        "high": src_test_income > q2,
    }
    group_masks = {k: torch.tensor(v, device=dev) for k, v in groups.items()}
    for feat_name in ["wet_bulb_F", "temp", "precip", "ndvi", "wind_mean", "uv_log_mean_annual", INCOME_KEY]: # HEREEE
        if feat_name not in feat_cols:
            continue
        fi = feat_cols.index(feat_name)
        x_perm = x.clone()
        x_perm[:, fi] = x_perm[torch.randperm(x_perm.shape[0]), fi]
        with torch.no_grad():
            h_p_src = model.encode(x_perm, adj_edge_index, "src")
            h_p_dst = model.encode(x_perm, adj_edge_index, "dst")
            perm_pred = model.decode(h_p_src[src_test], h_p_dst[dst_test], ea_test)
        row = []
        for label, mask in group_masks.items():
            base_g = cpc(pred_test[mask], y_test[mask])
            perm_g = cpc(perm_pred[mask], y_test[mask])
            row.append(f"{label}={base_g - perm_g:+.5f}")
        log(f"  {feat_name:25s}  " + "  ".join(row))
log("=" * 60)

importance_targets = sorted(set(income_idx) | set(climate_idx))
log(f"node feature importance, restricted set (n={len(importance_targets)})")
node_importances = []
for i in importance_targets:
    name = feat_cols[i]
    x_perm = x.clone()
    x_perm[:, i] = x_perm[torch.randperm(x_perm.shape[0]), i]
    with torch.no_grad():
        h_p_src = model.encode(x_perm, adj_edge_index, "src")
        h_p_dst = model.encode(x_perm, adj_edge_index, "dst")
        sc = cpc(model.decode(h_p_src[src_test], h_p_dst[dst_test], ea_test), y_test)
    node_importances.append((name, test_cpc - sc))
node_importances.sort(key=lambda t: t[1], reverse=True)
for name, imp in node_importances:
    log(f"  {name:45s}  {imp:+.5f}")

log("edge feature importance (drop in test CPC)")
edge_importances = []
for i, name in enumerate(dyadic_cols):
    ea_perm = ea_test.clone()
    ea_perm[:, i] = ea_perm[torch.randperm(ea_perm.shape[0]), i]
    with torch.no_grad():
        sc = cpc(model.decode(h_src_final[src_test], h_dst_final[dst_test], ea_perm), y_test)
    edge_importances.append((name, test_cpc - sc))
edge_importances.sort(key=lambda t: t[1], reverse=True)
for name, imp in edge_importances:
    log(f"  {name:30s}  {imp:+.5f}")

np.savez(f"the_gnn/climate_embeddings_{args.name}.npz",
         climate_names=np.array([feat_cols[i] for i in climate_idx]),
         gamma_src=mean_gamma_src, beta_src=mean_beta_src,
         gamma_dst=mean_gamma_dst, beta_dst=mean_beta_dst,
         h_src=h_src_final.cpu().numpy(), h_dst=h_dst_final.cpu().numpy())
log(f"saved gates + embeddings -> the_gnn/climate_embeddings_{args.name}.npz")

# =================================================================
# PLOTS
# =================================================================

if raw_income is not None:
    for role, gamma_val in [("src", mean_gamma_src), ("dst", mean_gamma_dst)]:
        gate_mean = gamma_val.mean(axis=1)
        corr, pval = spearmanr(raw_income[valid], gate_mean[valid])

        fig, ax = plt.subplots(figsize=(8, 8))
        binned_line_plot(ax, raw_income[valid], gate_mean[valid], n_bins=25, color="#2c7fb8")
        z = np.polyfit(raw_income[valid], gate_mean[valid], 1)
        xs = np.linspace(raw_income[valid].min(), raw_income[valid].max(), 200)
        ax.plot(xs, z[0] * xs + z[1], color="#d7301f", linewidth=2.2, linestyle="--")
        ax.set_xlabel(INCOME_KEY)
        ax.set_ylabel(r"Mean Climate Gate Activation ($\gamma$)")
        ax.set_title(f"Income vs. Climate Gate ({role})\nSpearman $r$={corr:.3f}, $p$={pval:.1e}")
        savefig(fig, f"income_vs_gate_{role}")

    climate_names_all = [feat_cols[i] for i in climate_idx]
    n_feats = len(climate_names_all)
    ncols = 3
    nrows = int(np.ceil(n_feats / ncols))
    for role, gamma_val in [("src", mean_gamma_src), ("dst", mean_gamma_dst)]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
        axes = np.array(axes).reshape(-1)
        for ci, name in enumerate(climate_names_all):
            ax = axes[ci]
            g = gamma_val[:, ci]
            binned_line_plot(ax, raw_income[valid], g[valid], n_bins=15, color="#41ab5d")
            corr, pval = spearmanr(raw_income[valid], g[valid])
            ax.set_title(f"{name}\n$r$={corr:.2f}", fontsize=15)
            ax.tick_params(labelsize=12)
        for j in range(n_feats, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"Income vs. Per-Feature Climate Gate ({role})")
        savefig(fig, f"income_vs_gate_perfeature_{role}")

pred_test_np = pred_test.detach().cpu().numpy()
y_test_np = y_test.detach().cpu().numpy()

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(np.log1p(y_test_np), np.log1p(pred_test_np), s=10, alpha=0.3,
           color="#6a51a3", edgecolor="none")
lims = [0, max(np.log1p(y_test_np).max(), np.log1p(pred_test_np).max())]
ax.plot(lims, lims, color="black", linewidth=1.5, linestyle="--")
ax.set_xlabel("log(1 + True Flow)")
ax.set_ylabel("log(1 + Predicted Flow)")
ax.set_title(f"Predicted vs. True Flow (Test)\nCPC={test_cpc:.3f}  RMSE={test_rmse:.2f}")
ax.set_xlim(lims); ax.set_ylim(lims)
savefig(fig, "pred_vs_true_flow")

muni = geobr.read_municipality(code_muni="all", year=2020)

def flow_graph_plot(src_idx_np, dst_idx_np, weights, title, fname, top_n, cmap_name="viridis"):
    order = np.argsort(weights)[::-1][:top_n]
    s_lat = np.array([code_to_latlon.get(city_codes[i], (np.nan, np.nan))[0] for i in src_idx_np[order]])
    s_lon = np.array([code_to_latlon.get(city_codes[i], (np.nan, np.nan))[1] for i in src_idx_np[order]])
    d_lat = np.array([code_to_latlon.get(city_codes[i], (np.nan, np.nan))[0] for i in dst_idx_np[order]])
    d_lon = np.array([code_to_latlon.get(city_codes[i], (np.nan, np.nan))[1] for i in dst_idx_np[order]])
    wgt = weights[order]
    w_norm = (wgt - wgt.min()) / (wgt.max() - wgt.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(14, 14))
    muni.boundary.plot(ax=ax, linewidth=0.15, color="#cccccc")

    cmap = plt.get_cmap(cmap_name)
    valid_edges = ~(np.isnan(s_lat) | np.isnan(d_lat))
    for i in np.where(valid_edges)[0]:
        ax.plot([s_lon[i], d_lon[i]], [s_lat[i], d_lat[i]],
                color=cmap(w_norm[i]), linewidth=0.4 + 1.6 * w_norm[i],
                alpha=0.15 + 0.5 * w_norm[i])

    sm_ = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=wgt.min(), vmax=wgt.max()))
    cbar = fig.colorbar(sm_, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Flow")
    ax.set_axis_off()
    ax.set_title(title, fontsize=19, pad=15)
    savefig(fig, fname)

src_test_np = src_test.detach().cpu().numpy()
dst_test_np = dst_test.detach().cpu().numpy()

flow_graph_plot(src_test_np, dst_test_np, y_test_np,
                title=f"True Migration Flow Network (Test, Top {args.top_edges} Edges)",
                fname="true_flow_graph", top_n=args.top_edges)

flow_graph_plot(src_test_np, dst_test_np, pred_test_np,
                title=f"Predicted Migration Flow Network (Test, Top {args.top_edges} Edges)",
                fname="pred_flow_graph", top_n=args.top_edges)

resid_arr, resid_mask = node_mean_residual(pred_test, y_test, src_test)
resid_df = pl.DataFrame({
    "city_code": [city_codes[i] for i in range(num_nodes) if resid_mask[i]],
    "residual": resid_arr[resid_mask],
})
muni["code_muni"] = pd.to_numeric(muni["code_muni"], errors="coerce").fillna(0).astype(int)
resid_map = muni.merge(resid_df.to_pandas(), left_on="code_muni", right_on="city_code", how="left")

fig, ax = plt.subplots(figsize=(12, 12))
resid_map.plot(
    column="residual", cmap="RdBu_r", scheme="quantiles", k=7,
    edgecolor="none", legend=True,
    legend_kwds={"title": "Mean Residual\n(True - Predicted)", "loc": "lower left", "fmt": "{:.1f}"},
    missing_kwds={"color": "#e0e0e0"},
    ax=ax,
)
ax.set_axis_off()
ax.set_title("Model Residuals by Source Municipality (Test)", fontsize=19, pad=15)
savefig(fig, "residual_map")

log(f"saved production plots -> {FIG_DIR}/")
log_file.close()
