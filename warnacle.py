"""
descriptive_accuracy.py
=============================================================================
Implements the Descriptive Accuracy criterion from Warnecke et al. (2020),
"Evaluating Explanation Methods for Deep Learning in Security" (EuroS&P).

Reuses the SAME 150 Tier-3-eligible alerts and cached SHAP values from the
cache-fidelity validation (Section IV-E / Table tab:cachefidelity), so no
new alert sampling is needed -- just re-scoring under feature masking.

Method:
  For each alert, for k = 1..6:
    - mask the top-k |SHAP|-ranked features (replace with per-feature
      median from the test set) -> re-run model -> record prob drop
    - mask k RANDOMLY chosen features (control) -> re-run model -> record
      prob drop
  AOPC (Area Over the Perturbation Curve) summarises each curve as a
  single number: mean probability drop across k=1..6. A faithful
  explanation should show AOPC_shap > AOPC_random by a clear margin.

Fill in MODEL_PATH / SCALER_PATH / the loader calls to match your existing
pipeline (same pattern as aria_real_mttr_controller.py).
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from data_loader import DataLoader  # noqa: E402

MODEL_PATH = "models/mlp_alert_train120k_4attack.pkl"
SCALER_PATH = "models/scaler_train120k_4attack.pkl"
TEST_SIZE = 40000
TEST_ATTACK_PRIOR = 0.05
N_ALERTS = 150          # match the cache-fidelity validation set size
MAX_K = 6                # match the drill-down report's "six strongest features"
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

print("Loading model, scaler, test data...")
mlp = joblib.load(MODEL_PATH)
loader = DataLoader()
loader.scaler = joblib.load(SCALER_PATH)
loader.scaler_fitted = True

df_test = loader.load_test()
X_test, y_test, feature_names, df_test_bal = loader.preprocess(
    df_test, target_attack_ratio=TEST_ATTACK_PRIOR, target_total_samples=TEST_SIZE
)
X_test = np.nan_to_num(X_test, nan=0.0)

# per-feature median across the test set -- the masking/baseline value
feature_medians = np.median(X_test, axis=0)

# --------------------------------------------------------------------------
# LOAD THE SAME 150 TIER-3 ALERTS + SHAP VALUES USED IN CACHE FIDELITY
# Replace this block with however cache_fidelity_validation.py selected its
# 150 alerts and stored their SHAP vectors -- reuse that exact set so the
# descriptive-accuracy numbers refer to the same alerts as Table
# tab:cachefidelity.
# --------------------------------------------------------------------------
# Example placeholder (adapt to your actual saved artifact):
# validation_df = pd.read_csv("experiments/results/cache_fidelity_alerts.csv")
# alert_indices = validation_df["global_idx"].values
# shap_matrix   = np.load("experiments/results/cache_fidelity_direct_shap.npy")  # shape (150, n_features)
#
# For a self-contained script, recompute directly on a fresh sample of 150
# Tier-3-eligible alerts if you don't have the saved SHAP matrix on hand:
import shap  # noqa: E402

y_prob_raw = mlp.predict_proba(X_test)[:, 1]

def prior_correct(p, train_prior=0.40, test_prior=TEST_ATTACK_PRIOR, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    odds = p / (1 - p)
    ratio = (test_prior / (1 - test_prior)) / (train_prior / (1 - train_prior))
    adj = odds * ratio
    return adj / (1 + adj)

y_prob = prior_correct(y_prob_raw)

# crude Tier-3-eligible proxy: top-confidence alerts (swap in your real QTS
# high_cutoff logic if you want an exact match to Section III-G's Tier 3)
tier3_candidates = np.argsort(y_prob)[::-1][:2000]
alert_indices = np.random.choice(tier3_candidates, N_ALERTS, replace=False)

background = shap.sample(X_test, 100, random_state=RANDOM_SEED)
explainer = shap.Explainer(mlp.predict_proba, background)

print(f"Computing SHAP for {N_ALERTS} alerts (for descriptive-accuracy masking)...")
shap_matrix = np.zeros((N_ALERTS, X_test.shape[1]))
for i, idx in enumerate(alert_indices):
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{N_ALERTS}...")
    shap_matrix[i] = explainer(X_test[idx:idx + 1]).values[0][:, 1]

# --------------------------------------------------------------------------
# DESCRIPTIVE ACCURACY: mask top-k SHAP features vs. k random features
# --------------------------------------------------------------------------
def masked_prob(x_row, feature_idx_to_mask):
    x_masked = x_row.copy()
    x_masked[feature_idx_to_mask] = feature_medians[feature_idx_to_mask]
    return mlp.predict_proba(x_masked.reshape(1, -1))[0, 1]

rows = []
for i, idx in enumerate(alert_indices):
    x_row = X_test[idx]
    p0 = y_prob_raw[i] if False else mlp.predict_proba(x_row.reshape(1, -1))[0, 1]

    shap_order = np.argsort(np.abs(shap_matrix[i]))[::-1]  # descending |SHAP|
    rand_order = np.random.permutation(len(feature_names))

    for k in range(1, MAX_K + 1):
        p_shap = masked_prob(x_row, shap_order[:k])
        p_rand = masked_prob(x_row, rand_order[:k])
        rows.append({
            "global_idx": idx, "k": k,
            "p0": p0,
            "drop_shap": p0 - p_shap,
            "drop_random": p0 - p_rand,
        })

df_da = pd.DataFrame(rows)

# AOPC per alert: mean drop across k=1..MAX_K
aopc_shap = df_da.groupby("global_idx")["drop_shap"].mean()
aopc_random = df_da.groupby("global_idx")["drop_random"].mean()

print("\n=== DESCRIPTIVE ACCURACY (Warnecke et al. 2020) ===")
print(f"Mean AOPC, SHAP-ranked masking : {aopc_shap.mean():.4f}")
print(f"Mean AOPC, random masking      : {aopc_random.mean():.4f}")
print(f"Descriptive-accuracy margin    : {aopc_shap.mean() - aopc_random.mean():+.4f}")

os.makedirs("experiments/results", exist_ok=True)
out_path = "experiments/results/descriptive_accuracy.csv"
df_da.to_csv(out_path, index=False)
print(f"\n✅ Saved per-alert, per-k results to: {out_path}")