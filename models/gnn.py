import argparse
import os

import geobr
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from libpysal.weights import KNN
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from torch.nn.parallel import parallel_apply, replicate
from torch_geometric.nn import GATConv, GCNConv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--gpu",  action="store_true", default=False)
args = parser.parse_args()

torch.manual_seed(42)
np.random.seed(42)
rng = np.random.default_rng(42)

os.makedirs("data",    exist_ok=True)
os.makedirs("the_gnn", exist_ok=True)

log_file = open(f"the_gnn/info_{args.name}.txt", "w")
def log(msg):
    print(msg); log_file.write(msg + "\n"); log_file.flush()

DROP_NODE_FEATURES = []
DROP_EDGE_FEATURES = []

X_train     = pl.read_csv("data/X_train.csv")
X_test      = pl.read_csv("data/X_test.csv")
X_inference = pl.read_csv("data/X_inference.csv")
Y_train     = pl.read_csv("data/y_train.csv")
Y_test      = pl.read_csv("data/y_test.csv")
Y_inference = pl.read_csv("data/y_inference.csv")

y_col = "flow" if "flow" in Y_train.columns else Y_train.columns[0]

id_cols  = ["source_code", "dest_code"]
src_cols = [c for c in X_train.columns if c.startswith("src_")]
dst_cols = [c for c in X_train.columns if c.startswith("dst_")]
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
city_to_idx = {code: i for i, code in enumerate(city_codes)}
num_nodes   = len(city_codes)
city_map_df = pl.DataFrame({"city_code": city_codes, "node_idx": list(range(num_nodes))})

income_keywords  = ["gdp"]
climate_keywords = ["_temp", "precipitation", "ndvi", "uv_", "humid",
                    "wind_mean", "wet_bulb", "degree_day"]
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

# compute nodes have no internet: prefer the centroid cache written by
# prep_centroids.py (run once on a login node), fall back to downloading.
centroid_cache = "data/muni_centroids.csv"
if os.path.exists(centroid_cache):
    cent = pl.read_csv(centroid_cache)
    code_to_latlon = {int(c): (float(la), float(lo))
                      for c, la, lo in zip(cent["code_muni"], cent["lat"], cent["lon"])}
else:
    muni = geobr.read_municipality(year=2010, simplified=True, verbose=False)
    code_to_latlon = {
        int(row["code_muni"]): (float(row.geometry.centroid.y), float(row.geometry.centroid.x))
        for _, row in muni.iterrows()
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

known_codes = np.unique(
    np.concatenate([src_train.numpy(), src_test.numpy(), src_infer.numpy()]).astype(np.int64) * num_nodes
    + np.concatenate([dst_train.numpy(), dst_test.numpy(), dst_infer.numpy()]).astype(np.int64)
)

def sample_negatives(n, rng, device):
    s_chunks, d_chunks = [], []
    remaining = n
    while remaining > 0:
        cs = rng.integers(0, num_nodes, size=int(remaining * 1.4) + 64)
        cd = rng.integers(0, num_nodes, size=int(remaining * 1.4) + 64)
        cs, cd = cs[cs != cd], cd[cs != cd]
        codes = cs.astype(np.int64) * num_nodes + cd.astype(np.int64)
        pos  = np.clip(np.searchsorted(known_codes, codes), 0, len(known_codes) - 1)
        mask = known_codes[pos] != codes
        s_chunks.append(cs[mask][:remaining]); d_chunks.append(cd[mask][:remaining])
        remaining -= len(s_chunks[-1])
    src_neg, dst_neg = np.concatenate(s_chunks), np.concatenate(d_chunks)
    ea_neg = torch.zeros(len(src_neg), ea_train.shape[1], dtype=torch.float, device=device)
    return (torch.tensor(src_neg, dtype=torch.long, device=device),
            torch.tensor(dst_neg, dtype=torch.long, device=device),
            ea_neg, torch.zeros(len(src_neg), dtype=torch.float, device=device))


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
    source role in an edge, one used when it plays the destination role."""

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

        self._init_weights()

    def _init_weights(self):
        w = self.conv1.lin.weight
        nn.init.eye_(w[:min(w.shape), :min(w.shape)])
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.res1.weight)
        nn.init.ones_(self.gravity_skip.weight)
        nn.init.zeros_(self.gravity_skip.bias)
        for net in (self.income_gate_net_src, self.income_gate_net_dst):
            nn.init.normal_(net[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(net[-1].bias)
        nn.init.eye_(self.src_proj.weight); nn.init.zeros_(self.src_proj.bias)
        nn.init.eye_(self.dst_proj.weight); nn.init.zeros_(self.dst_proj.bias)

    def gate_logits(self, x, role):
        net = self.income_gate_net_src if role == "src" else self.income_gate_net_dst
        return net(x[:, self.income_idx])

    def gate_activations(self, x, role):
        return torch.sigmoid(self.gate_logits(x, role))

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
        dec_in  = torch.cat([s, d, s * d, edge_proj], dim=1)
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
multi_gpu = len(devices) > 1
dev       = devices[0]
log(f"devices: {[str(d) for d in devices]}")

model = SpatialGatedDG(
    node_dim=x.shape[1], edge_dim=ea_train.shape[1],
    income_idx=income_idx, climate_idx=climate_idx, gravity_edge_idx=gravity_edge_idx,
    hidden=4096, out=1024, heads=8, dropout=0.3
).to(dev)
log(f"parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800, eta_min=1e-5)

def to_dev(*tensors): return [t.to(dev) for t in tensors]
x, Wx, adj_edge_index, adj_weights = to_dev(x, Wx, adj_edge_index, adj_weights)
src_train, dst_train, ea_train, y_train = to_dev(src_train, dst_train, ea_train, y_train)
src_test,  dst_test,  ea_test,  y_test  = to_dev(src_test,  dst_test,  ea_test,  y_test)
src_infer, dst_infer, ea_infer, y_infer = to_dev(src_infer, dst_infer, ea_infer, y_infer)

log("target: raw flow + Poisson NLL")

neg_test  = sample_negatives(int(len(src_test)  * 0.3), rng, dev)
neg_infer = sample_negatives(int(len(src_infer) * 0.3), rng, dev)

def flow_loss(pred, target):
    return (pred - target * torch.log(pred + 1e-8)).mean()

def run_decode(model, h_src, h_dst, ea, src, dst):
    if multi_gpu:
        replicas = replicate(model, devices)
        n = len(devices)
        outs = parallel_apply(
            [r.decode for r in replicas],
            [(h_src[s].to(d), h_dst[dt].to(d), e.to(d))
             for s, dt, e, d in zip(torch.chunk(src, n), torch.chunk(dst, n), torch.chunk(ea, n), devices)]
        )
        return torch.cat([o.to(dev) for o in outs])
    return model.decode(h_src[src], h_dst[dst], ea)

def eval_split(model, h_src, h_dst, src, dst, ea, y_raw, neg):
    sn, dn, ean, yn_raw = neg
    pred = run_decode(model, h_src, h_dst, torch.cat([ea, ean]), torch.cat([src, sn]), torch.cat([dst, dn]))
    loss = flow_loss(pred, torch.cat([y_raw, yn_raw])).item()
    return loss, rmse(pred, torch.cat([y_raw, yn_raw])), cpc(pred, torch.cat([y_raw, yn_raw]))

ckpt_path      = f"the_gnn/best_sgdg_{args.name}.pt"
best_infer_cpc = -1
n_neg_train    = int(len(src_train) * 0.3)
pbar = tqdm(range(1, 800 + 1))

for epoch in pbar:
    model.train(); optimizer.zero_grad()
    h_src = model.encode(x, adj_edge_index, "src")
    h_dst = model.encode(x, adj_edge_index, "dst")

    self_loss = 0.5 * (F.mse_loss(model.recon_self(h_src), x) + F.mse_loss(model.recon_self(h_dst), x))
    lag_loss  = 0.5 * (F.mse_loss(model.recon_lag(h_src),  Wx) + F.mse_loss(model.recon_lag(h_dst),  Wx))
    # L1 on the effective FiLM multiplier |1+gamma| (and |beta| so signal can't
    # leak back through the shift): open gates cost, so climate only passes
    # where it buys prediction. gamma=0 is a fully open gate, gamma=-1 closed.
    gamma_s, beta_s = model.get_film_params(x, "src")
    gamma_d, beta_d = model.get_film_params(x, "dst")
    gate_open = 0.5 * ((1.0 + gamma_s).abs().mean() + (1.0 + gamma_d).abs().mean())
    beta_reg  = 0.5 * (beta_s.abs().mean() + beta_d.abs().mean())

    sn, dn, ean, yn_raw = sample_negatives(n_neg_train, rng, dev)
    pred = run_decode(model, h_src, h_dst, torch.cat([ea_train, ean]), torch.cat([src_train, sn]), torch.cat([dst_train, dn]))
    f_loss = flow_loss(pred, torch.cat([y_train, yn_raw]))
    loss   = f_loss + 0.1 * lag_loss + 0.1 * self_loss + 0.01 * (gate_open + beta_reg)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step(); scheduler.step()

    pbar.set_postfix(flow=f"{f_loss.item():.4f}", lag=f"{lag_loss.item():.4f}",
                     gate=f"{gate_open.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            h_src_eval = model.encode(x, adj_edge_index, "src")
            h_dst_eval = model.encode(x, adj_edge_index, "dst")
            _, _, tc = eval_split(model, h_src_eval, h_dst_eval, src_test,  dst_test,  ea_test,  y_test,  neg_test)
            _, _, ic = eval_split(model, h_src_eval, h_dst_eval, src_infer, dst_infer, ea_infer, y_infer, neg_infer)
        log(f"epoch {epoch:4d}  test_CPC {tc:.4f}  infer_CPC {ic:.4f}")
        if ic > best_infer_cpc:
            best_infer_cpc = ic
            torch.save(model.state_dict(), ckpt_path)

log(f"best inference CPC: {best_infer_cpc:.4f} -> {ckpt_path}")

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

income_col_idx = next((i for i, c in enumerate(feat_cols) if "mean_income" in c), None)
raw_income = None
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
        torch.tensor(node_gamma_summary, dtype=torch.float, device=dev), adj_edge_index, adj_weights, 999, 42)
    log(f"  gamma_mean[{role}] activations   I={i_gate:+.4f}  z={z_gate:+.2f}  p={p_gate:.4f}")

if raw_income is not None:
    i_inc, z_inc, p_inc = morans_i_test(
        torch.tensor(raw_income, dtype=torch.float, device=dev), adj_edge_index, adj_weights, 999, 42)
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
result = moran_on_subgraph(model_resid, model_mask, 999, 42)
if result:
    i_m, z_m, p_m = result
    log(f"  model residual / node     I={i_m:+.4f}  z={z_m:+.2f}  p={p_m:.4f}  n={int(model_mask.sum())}")

gravity_feats  = X_train.select(["distance_km", "pop_ratio"]).to_numpy().astype(np.float64)
gravity_design = np.column_stack([np.ones(len(gravity_feats)), gravity_feats])
gravity_coef, *_ = np.linalg.lstsq(gravity_design, y_train.cpu().numpy().astype(np.float64), rcond=None)

test_gravity_feats  = X_test.select(["distance_km", "pop_ratio"]).to_numpy().astype(np.float64)
test_gravity_design = np.column_stack([np.ones(len(test_gravity_feats)), test_gravity_feats])
gravity_pred_test    = test_gravity_design @ gravity_coef
gravity_resid        = y_test.cpu().numpy().astype(np.float64) - gravity_pred_test

gravity_resid_t, gravity_mask = node_mean_residual(
    torch.tensor(gravity_resid, dtype=torch.float, device=dev), torch.zeros_like(y_test), src_test)
result = moran_on_subgraph(gravity_resid_t, gravity_mask, 999, 42)
if result:
    i_g, z_g, p_g = result
    log(f"  gravity-only residual/node I={i_g:+.4f}  z={z_g:+.2f}  p={p_g:.4f}  n={int(gravity_mask.sum())}")
log("=" * 60)

if raw_income is not None:
    src_test_income = raw_income[src_test.cpu().numpy()]
    groups = {
        "low":  src_test_income <= q1,
        "mid":  (src_test_income > q1) & (src_test_income <= q2),
        "high": src_test_income > q2,
    }
    group_masks = {k: torch.tensor(v, device=dev) for k, v in groups.items()}
    for feat_name in ["wet_bulb_F", "UV_slope", "Precipitation", "mean_income_brl_2010"]:
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

log("node feature importance (drop in test CPC)")
node_importances = []
for i, name in enumerate(feat_cols):
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

log("=" * 60)
log("embedding summary per climate feature (src-gate vs dst-gate, gate-weighted mean embedding)")
climate_names = [feat_cols[i] for i in climate_idx]
emb_src_np = h_src_final.cpu().numpy()
emb_dst_np = h_dst_final.cpu().numpy()

for ci, name in enumerate(climate_names):
    g_src = mean_gamma_src[:, ci]
    g_dst = mean_gamma_dst[:, ci]
    w_src = g_src / (g_src.sum() + 1e-8)
    w_dst = g_dst / (g_dst.sum() + 1e-8)
    emb_src_w = (emb_src_np * w_src[:, None]).sum(axis=0)
    emb_dst_w = (emb_dst_np * w_dst[:, None]).sum(axis=0)
    log(f"  [{name:25s}] src  gamma_mean={g_src.mean():.4f}  ||emb||={np.linalg.norm(emb_src_w):.4f}  "
        f"dims[:8]={np.array2string(emb_src_w[:8], precision=3, suppress_small=True)}")
    log(f"  [{name:25s}] dst  gamma_mean={g_dst.mean():.4f}  ||emb||={np.linalg.norm(emb_dst_w):.4f}  "
        f"dims[:8]={np.array2string(emb_dst_w[:8], precision=3, suppress_small=True)}")

np.savez(f"the_gnn/climate_embeddings_{args.name}.npz",
         climate_names=np.array(climate_names),
         gamma_src=mean_gamma_src, beta_src=mean_beta_src,
         gamma_dst=mean_gamma_dst, beta_dst=mean_beta_dst,
         h_src=emb_src_np, h_dst=emb_dst_np)
log(f"saved per-climate-feature gates + embeddings -> the_gnn/climate_embeddings_{args.name}.npz")

log_file.close()
