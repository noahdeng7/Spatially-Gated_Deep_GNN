"""
Random-forest baseline for the Mexican municipality-to-municipality migration
panel.

Regresses log1p(migrants) on the dyad's covariates, tuning `min_samples_leaf`
and `n_estimators` by out-of-bag MSE. Inputs are the splits written by
`models/make_splits.py`.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Repo root = parent of models/
ROOT = Path(__file__).resolve().parent.parent

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--data-dir", type=Path, default=ROOT / "data",
                 help="directory holding the splits from make_splits.py")
_ap.add_argument("--outdir", type=Path,
                 default=ROOT / "models" / "runs" / "random_forest",
                 help="where the fitted model and metrics are written")
_ap.add_argument("--stdout", action="store_true",
                 help="print to the terminal instead of redirecting to a file")
args = _ap.parse_args()

# Created BEFORE stdout is redirected into it. The previous version redirected
# both stdout and stderr into `random_forest/final_run.txt` at import time
# without creating the directory, so the script died on its first statement with
# nowhere to report the error.
args.outdir.mkdir(parents=True, exist_ok=True)

if not args.stdout:
    sys.stdout = open(args.outdir / "final_run.txt", "w",
                      buffering=1, encoding="utf-8")
    sys.stderr = sys.stdout

NEG_SAMPLE_RATE = 0.3
MTRY_FRACTION = 1.0 / 3.0
rng = np.random.default_rng(42)

# The two geographic keys are identifiers, not covariates. They must be dropped
# explicitly: read from CSV, `01001` is inferred as the integer 1001, so the
# automatic "drop the string columns" rule below does not catch them and the
# municipality code would enter the forest as an ordinal predictor -- adding
# noise and making the feature-importance table unreadable.
ID_COLS = ["source_code", "dest_code"]


def common_part_of_commuters(y_true, y_pred):
    y_true = np.clip(y_true, a_min=0, a_max=None)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    numerator = 2 * np.sum(np.minimum(y_true, y_pred))
    denominator = np.sum(y_true) + np.sum(y_pred)
    return numerator / denominator if denominator > 0 else np.nan


def eval_metrics(label, y_test_log, preds_log):
    preds_flow = np.expm1(preds_log)
    y_test_flow = np.expm1(y_test_log)
    rmse_log = np.sqrt(mean_squared_error(y_test_log, preds_log))
    r2_log = r2_score(y_test_log, preds_log)
    rmse_flow = np.sqrt(mean_squared_error(y_test_flow, preds_flow))
    r2_flow = r2_score(y_test_flow, preds_flow)
    cpc_flow = common_part_of_commuters(y_test_flow, preds_flow)
    print(f"\n[{label}] RMSE (log1p space): {rmse_log:.6f}")
    print(f"[{label}] R2   (log1p space): {r2_log:.6f}")
    print(f"[{label}] RMSE (flow space):  {rmse_flow:.2f}")
    print(f"[{label}] R2   (flow space):  {r2_flow:.6f}")
    print(f"[{label}] CPC  (flow space):  {cpc_flow:.6f}")
    return {
        "rmse_log": rmse_log, "r2_log": r2_log,
        "rmse_flow": rmse_flow, "r2_flow": r2_flow,
        "cpc_flow": cpc_flow,
    }


def load_xy(x_path, y_path, non_numeric_cols=None):
    for p in (x_path, y_path):
        if not p.exists():
            raise SystemExit(
                f"missing {p}"
                "\nBuild the splits first:  python models/make_splits.py"
            )
    x_df = pl.read_csv(x_path)
    y_df = pl.read_csv(y_path)
    if non_numeric_cols is None:
        non_numeric_cols = [c for c, t in zip(x_df.columns, x_df.dtypes) if t == pl.String]
        non_numeric_cols += [c for c in ID_COLS
                             if c in x_df.columns and c not in non_numeric_cols]
    x_df = x_df.drop(non_numeric_cols)
    FILL_VALUE = np.finfo(np.float32).min
    x_df = x_df.with_columns([pl.col(c).fill_null(FILL_VALUE) for c in x_df.columns])
    X = np.ascontiguousarray(x_df.to_numpy(), dtype=np.float32)
    y = np.ascontiguousarray(y_df.to_numpy(), dtype=np.float32).squeeze()
    return X, y, x_df.columns, non_numeric_cols


def add_negatives(X, y, rate, rng):
    n_neg = int(len(y) * rate)
    idx = rng.choice(len(X), size=n_neg, replace=True)
    X_full = np.vstack([X, X[idx]])
    y_full = np.concatenate([y, np.zeros(n_neg, dtype=np.float32)])
    return X_full, y_full, n_neg


def oob_mse(model, y_true):
    return mean_squared_error(y_true, model.oob_prediction_)


X_train, y_train, train_cols, non_numeric_cols = load_xy(
    args.data_dir / "X_train.csv", args.data_dir / "y_train_log1p.csv"
)
X_test, y_test, _, _ = load_xy(
    args.data_dir / "X_test.csv", args.data_dir / "y_test_log1p.csv", non_numeric_cols
)
X_inference, y_inference, _, _ = load_xy(
    args.data_dir / "X_inference.csv", args.data_dir / "y_inference_log1p.csv",
    non_numeric_cols
)
print(f"dropped as non-predictive: {non_numeric_cols}")
print(f"predictors ({len(train_cols)}): {train_cols}")

X_train, y_train, n_neg_train = add_negatives(X_train, y_train, NEG_SAMPLE_RATE, rng)
X_test, y_test, n_neg_test = add_negatives(X_test, y_test, NEG_SAMPLE_RATE, rng)

print(f"neg_sample_rate={NEG_SAMPLE_RATE}  n_neg_train={n_neg_train}  n_neg_test={n_neg_test}")
print(f"train shape={X_train.shape}  test shape={X_test.shape}  inference shape={X_inference.shape}")

leaf_candidates = [5, 10, 20, 50, 100]
leaf_scores = {}
for leaf in leaf_candidates:
    rf = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=leaf,
        max_features=MTRY_FRACTION,
        criterion="squared_error",
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    leaf_scores[leaf] = oob_mse(rf, y_train)
    print(f"leaf_size={leaf:4d}  OOB MSE={leaf_scores[leaf]:.6f}")

best_leaf = min(leaf_scores, key=leaf_scores.get)
print(f"Best leaf size: {best_leaf} (OOB MSE {leaf_scores[best_leaf]:.6f})")

tree_candidates = [20, 50, 100, 200]
tree_scores = {}
for n_trees in tree_candidates:
    rf = RandomForestRegressor(
        n_estimators=n_trees,
        min_samples_leaf=best_leaf,
        max_features=MTRY_FRACTION,
        criterion="squared_error",
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    tree_scores[n_trees] = oob_mse(rf, y_train)
    print(f"n_trees={n_trees:4d}  OOB MSE={tree_scores[n_trees]:.6f}")

best_n_trees = min(tree_scores, key=tree_scores.get)
print(f"Best n_trees: {best_n_trees} (OOB MSE {tree_scores[best_n_trees]:.6f})")

best_params = {
    "min_samples_leaf": best_leaf,
    "n_estimators": best_n_trees,
    "max_features": MTRY_FRACTION,
}
print(f"Best params: {best_params}")

best_rf = RandomForestRegressor(
    n_estimators=best_n_trees,
    min_samples_leaf=best_leaf,
    max_features=MTRY_FRACTION,
    criterion="squared_error",
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
best_rf.fit(X_train, y_train)
best_oob_mse = oob_mse(best_rf, y_train)
print(f"Best model OOB MSE: {best_oob_mse:.6f}")

preds_test_log = best_rf.predict(X_test)
preds_inference_log = best_rf.predict(X_inference)

test_metrics = eval_metrics("test", y_test, preds_test_log)
inference_metrics = eval_metrics("inference", y_inference, preds_inference_log)

importances = best_rf.feature_importances_
ranked = sorted(zip(train_cols, importances), key=lambda x: x[1], reverse=True)
for name, imp in ranked:
    print(f"{name:50s} {imp:.6f}")

joblib.dump(
    {
        "model": best_rf,
        "best_params": best_params,
        "best_oob_mse": best_oob_mse,
        "leaf_scores": leaf_scores,
        "tree_scores": tree_scores,
        "neg_sample_rate": NEG_SAMPLE_RATE,
        "test_metrics": test_metrics,
        "inference_metrics": inference_metrics,
    },
    args.outdir / "best_random_forest_model.pkl",
)
print(f"\nwrote {args.outdir}")
if not args.stdout:
    sys.stdout.close()
