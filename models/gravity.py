"""
PPML gravity baseline for the Mexican municipality-to-municipality migration
panel.

A Poisson GLM of migrant counts on log origin population, log destination
population and log distance -- the standard gravity specification, estimated by
pseudo-maximum-likelihood so that zeros are admissible.

Inputs are the splits written by `models/make_splits.py`.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Repo root = parent of models/
ROOT = Path(__file__).resolve().parent.parent

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--data-dir", type=Path, default=ROOT / "data",
                 help="directory holding the splits from make_splits.py")
_ap.add_argument("--outdir", type=Path, default=ROOT / "models" / "runs" / "gravity",
                 help="where the fitted model and metrics are written")
_ap.add_argument("--stdout", action="store_true",
                 help="print to the terminal instead of redirecting to a file")
args = _ap.parse_args()

# The output directory is created BEFORE stdout is redirected into it. The
# previous version redirected stdout to `simple_gravity/baseline_gravity.txt`
# at import time without creating the directory, so the script died on its
# first statement -- and because stderr had been pointed at the same unopened
# file, it died silently.
args.outdir.mkdir(parents=True, exist_ok=True)

if not args.stdout:
    sys.stdout = open(args.outdir / "baseline_gravity.txt", "w",
                      buffering=1, encoding="utf-8")
    sys.stderr = sys.stdout


def common_part_of_commuters(y_true, y_pred):
    y_true = np.clip(y_true, a_min=0, a_max=None)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    return numerator / denominator if denominator > 0 else np.nan


def eval_metrics(label, y_true, y_pred):
    """Compute + print flow/log metrics for a given split, tagged with `label`."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    pred_log = np.log1p(np.clip(y_pred, a_min=0, a_max=None))
    r2_log = r2_score(np.log1p(y_true), pred_log)
    rmse_log = np.sqrt(mean_squared_error(np.log1p(y_true), pred_log))
    r2_flow = r2_score(y_true, y_pred)
    rmse_flow = np.sqrt(mean_squared_error(y_true, y_pred))
    cpc_flow = common_part_of_commuters(y_true, y_pred)
    print(f"[{label}] log  R2={r2_log:.6f}  RMSE={rmse_log:.6f}")
    print(f"[{label}] flow R2={r2_flow:.6f}  RMSE={rmse_flow:.2f}  CPC={cpc_flow:.6f}")
    return {
        "r2_log": r2_log, "rmse_log": rmse_log,
        "r2_flow": r2_flow, "rmse_flow": rmse_flow,
        "cpc_flow": cpc_flow,
    }


t0 = time.time()
rng = np.random.default_rng(42)
NEG_RATE = 0.3

def _read(name):
    path = args.data_dir / name
    if not path.exists():
        raise SystemExit(
            f"missing {path}"
            "\nBuild the splits first:  python models/make_splits.py"
        )
    return pl.read_csv(path)

X_train_df = _read("X_train.csv")
X_test_df = _read("X_test.csv")
y_train_df = _read("y_train.csv")
y_test_df = _read("y_test.csv")

# The inference split is reported separately so that no number in the paper is
# read off a split that was used for model selection.
X_inference_df = _read("X_inference.csv")
y_inference_df = _read("y_inference.csv")

# Column names as the pipeline writes them. The previous list
# ("src_total_pop", "dst_total_pop", "distance_km") came from an earlier panel
# and matches nothing here, so every run raised a polars ColumnNotFound.
FEATS = ["src_pop", "dst_pop", "dist_geodesic_km"]
_missing = [c for c in FEATS if c not in X_train_df.columns]
if _missing:
    raise SystemExit(f"missing feature columns {_missing}; have {X_train_df.columns}")
pop_train = X_train_df.select(FEATS).to_numpy().astype(np.float64)
pop_test = X_test_df.select(FEATS).to_numpy().astype(np.float64)
pop_inference = X_inference_df.select(FEATS).to_numpy().astype(np.float64)

y_train_pos = y_train_df.to_numpy().astype(np.float64).squeeze()
y_test_pos = y_test_df.to_numpy().astype(np.float64).squeeze()
y_inference_pos = y_inference_df.to_numpy().astype(np.float64).squeeze()

n_neg_train = int(len(y_train_pos) * NEG_RATE)
n_neg_test = int(len(y_test_pos) * NEG_RATE)

neg_idx_train = rng.choice(len(pop_train), size=n_neg_train, replace=True)
neg_idx_test = rng.choice(len(pop_test), size=n_neg_test, replace=True)

pop_train_full = np.vstack([pop_train, pop_train[neg_idx_train]])
pop_test_full = np.vstack([pop_test, pop_test[neg_idx_test]])

y_train = np.concatenate([y_train_pos, np.zeros(n_neg_train)])
y_test = np.concatenate([y_test_pos, np.zeros(n_neg_test)])

pop_inference_full = pop_inference
y_inference = y_inference_pos


def build_design(pop):
    log_pop_o = np.log(pop[:, 0])
    log_pop_d = np.log(pop[:, 1])
    log_dist = np.log(pop[:, 2])
    X = np.column_stack([log_pop_o, log_pop_d, log_dist])
    return sm.add_constant(X, has_constant="add")


X_train = build_design(pop_train_full)
X_test = build_design(pop_test_full)
X_inference = build_design(pop_inference_full)

print(f"train {X_train.shape}  test {X_test.shape}  inference {X_inference.shape}  "
      f"neg_rate={NEG_RATE}  ({time.time()-t0:.1f}s)")

model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
result = model.fit()
print(result.summary())

pred_test = result.predict(X_test)
pred_inference = result.predict(X_inference)

test_metrics = eval_metrics("test", y_test, pred_test)
inference_metrics = eval_metrics("inference", y_inference, pred_inference)

result.save(str(args.outdir / "gravity_ppml_model.pkl"))

results = {
    "params": result.params.tolist(),
    "test": test_metrics,
    "inference": inference_metrics,
    "neg_sample_rate": NEG_RATE,
    "seed": 42,
}
joblib.dump(results, args.outdir / "results_baseline_gravity.pkl")
print(f"\n{time.time()-t0:.1f}s")
print(f"wrote {args.outdir}")
if not args.stdout:
    sys.stdout.close()
