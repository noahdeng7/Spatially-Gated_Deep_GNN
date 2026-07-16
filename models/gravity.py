import os
import sys
import time

import joblib
import numpy as np
import polars as pl
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.stdout = open("simple_gravity/baseline_gravity.txt", "w", buffering=1)
sys.stderr = sys.stdout

t0 = time.time()

rng = np.random.default_rng(42)

X_train_df = pl.read_csv("data/X_train.csv")
X_test_df = pl.read_csv("data/X_test.csv")
y_train_df = pl.read_csv("data/y_train.csv")
y_test_df = pl.read_csv("data/y_test.csv")

pop_train = X_train_df.select(["src_total_pop", "dst_total_pop", "distance_km"]).to_numpy().astype(np.float64)
pop_test = X_test_df.select(["src_total_pop", "dst_total_pop", "distance_km"]).to_numpy().astype(np.float64)

y_train_pos = y_train_df.to_numpy().astype(np.float64).squeeze()
y_test_pos = y_test_df.to_numpy().astype(np.float64).squeeze()

n_neg_train = int(len(y_train_pos) * 0.3)
n_neg_test = int(len(y_test_pos) * 0.3)

neg_idx_train = rng.choice(len(pop_train), size=n_neg_train, replace=True)
neg_idx_test = rng.choice(len(pop_test), size=n_neg_test, replace=True)

pop_train_full = np.vstack([pop_train, pop_train[neg_idx_train]])
pop_test_full = np.vstack([pop_test, pop_test[neg_idx_test]])

y_train = np.concatenate([y_train_pos, np.zeros(n_neg_train)])
y_test = np.concatenate([y_test_pos, np.zeros(n_neg_test)])

log_pop_o_train = np.log(pop_train_full[:, 0])
log_pop_d_train = np.log(pop_train_full[:, 1])
log_dist_train = np.log(pop_train_full[:, 2])

log_pop_o_test = np.log(pop_test_full[:, 0])
log_pop_d_test = np.log(pop_test_full[:, 1])
log_dist_test = np.log(pop_test_full[:, 2])

X_train = np.column_stack([log_pop_o_train, log_pop_d_train, log_dist_train])
X_test = np.column_stack([log_pop_o_test, log_pop_d_test, log_dist_test])

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

print(f"train {X_train.shape}  test {X_test.shape}  neg_rate=0.3  ({time.time()-t0:.1f}s)")

model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
result = model.fit()

print(result.summary())

pred_flow = result.predict(X_test)
pred_log = np.log1p(pred_flow)

print(f"log  MSE={mean_squared_error(np.log1p(y_test), pred_log):.6f}  MAE={mean_absolute_error(np.log1p(y_test), pred_log):.6f}  R2={r2_score(np.log1p(y_test), pred_log):.6f}")
print(f"flow MSE={mean_squared_error(y_test, pred_flow):.2f}  MAE={mean_absolute_error(y_test, pred_flow):.2f}  R2={r2_score(y_test, pred_flow):.6f}")

result.save("simple_gravity/gravity_ppml_model.pkl")

results = {
    "params": result.params.tolist(),
    "r2_log": r2_score(np.log1p(y_test), pred_log),
    "r2_flow": r2_score(y_test, pred_flow),
    "mse_flow": mean_squared_error(y_test, pred_flow),
    "mae_flow": mean_absolute_error(y_test, pred_flow),
    "neg_sample_rate": 0.3,
    "seed": 42,
}

joblib.dump(results, "simple_gravity/results_baseline_gravity.pkl")

print(f"\n{time.time()-t0:.1f}s")
sys.stdout.close()
