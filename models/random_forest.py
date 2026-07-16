import sys
import numpy as np
import polars as pl
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
import joblib

sys.stdout = open("random_forest/final_run.txt", "w", buffering=1)
sys.stderr = sys.stdout

NEG_SAMPLE_RATE = 0.3
rng = np.random.default_rng(42)


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
    x_df = pl.read_csv(x_path)
    y_df = pl.read_csv(y_path)
    if non_numeric_cols is None:
        non_numeric_cols = [c for c, t in zip(x_df.columns, x_df.dtypes) if t == pl.String]
    x_df = x_df.drop(non_numeric_cols)
    FILL_VALUE = np.finfo(np.float32).min
    x_df = x_df.with_columns([pl.col(c).fill_null(FILL_VALUE) for c in x_df.columns])
    X = np.ascontiguousarray(x_df.to_numpy(), dtype=np.float32)
    y = np.ascontiguousarray(y_df.to_numpy(), dtype=np.float32).squeeze()
    return X, y, x_df.columns, non_numeric_cols


X_train, y_train, train_cols, non_numeric_cols = load_xy(
    "data/X_train.csv", "data/y_train_log1p.csv"
)
X_test, y_test, _, _ = load_xy(
    "data/X_test.csv", "data/y_test_log1p.csv", non_numeric_cols
)
X_inference, y_inference, _, _ = load_xy(
    "data/X_inference.csv", "data/y_inference_log1p.csv", non_numeric_cols
)

def add_negatives(X, y, rate, rng):
    n_neg = int(len(y) * rate)
    idx = rng.choice(len(X), size=n_neg, replace=True)
    X_full = np.vstack([X, X[idx]])
    y_full = np.concatenate([y, np.zeros(n_neg, dtype=np.float32)])
    return X_full, y_full, n_neg

X_train, y_train, n_neg_train = add_negatives(X_train, y_train, NEG_SAMPLE_RATE, rng)
X_test, y_test, n_neg_test = add_negatives(X_test, y_test, NEG_SAMPLE_RATE, rng)

print(f"neg_sample_rate={NEG_SAMPLE_RATE}  n_neg_train={n_neg_train}  n_neg_test={n_neg_test}")
print(f"train shape={X_train.shape}  test shape={X_test.shape}  inference shape={X_inference.shape}")

param_distributions = {
    "n_estimators":      randint(100, 800),
    "max_depth":         [10, 20, 40, 80, None],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf":  randint(1, 6),
    "max_features":      ["sqrt", "log2", 0.3, 0.5, None],
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestRegressor(
    criterion="squared_error",
    random_state=42,
    n_jobs=-1,
)
search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=20,
    cv=kf,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1,
    verbose=2,
)
search.fit(X_train, y_train)
best_rf     = search.best_estimator_
best_params = search.best_params_
best_mae    = -search.best_score_
print(f"Best params: {best_params}")
print(f"Best CV MAE (log1p): {best_mae:.6f}")

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
        "best_cv_mae_log1p": best_mae,
        "neg_sample_rate": NEG_SAMPLE_RATE,
        "test_metrics": test_metrics,
        "inference_metrics": inference_metrics,
    },
    "random_forest/best_random_forest_model_new.pkl",
)
sys.stdout.close()
