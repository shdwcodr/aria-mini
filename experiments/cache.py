"""
ARIA Cache Fidelity Check
==========================
Validates the assumption underlying the explanation cache (Section III-G):
that the SHAP/LIME explanation of the nearest cached prototype is a
reasonable stand-in for the true, directly-computed explanation of a new
alert.

The paper currently reports a ~1871x speedup for using the cache but never
checks whether the retrieved explanation is actually *correct* for the new
instance. This script closes that gap.

For a held-out sample of Tier-3-eligible alerts, this script computes BOTH:
  (a) the cached explanation (nearest-neighbour lookup, as ARIA does at runtime)
  (b) the true direct explanation (fresh SHAP + LIME computation)
and reports several agreement metrics between them.

Run this after the explanation cache in soc_simulation.py has been built
the same way (same CACHE_SIZE, same prototype selection seed) so the
comparison reflects the actual deployed cache, not a different one.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from data_loader import DataLoader

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONFIG — must match soc_simulation.py exactly, so the cache being tested
# is the SAME cache ARIA actually uses at runtime
# =============================================================================
MODEL_PATH  = "models/mlp_alert_train120k_4attack.pkl"
SCALER_PATH = "models/scaler_train120k_4attack.pkl"
TEST_SIZE   = 40000
CACHE_SIZE  = 500
TEST_ATTACK_PRIOR  = 0.05
TRAIN_ATTACK_PRIOR = 0.40   # matches model_trainer.py's ATTACK_RATIO_TRAIN

# How many alerts to validate the cache against (kept modest — this runs
# direct SHAP+LIME for every one of these, which is the expensive path)
N_VALIDATION_ALERTS = 150

# Same QTS cutoff used in soc_simulation.py, so we only validate cache
# entries actually used for alerts that would reach Tier 3 in production
BASE_HIGH_CUTOFF = 0.15
LOAD_ADJUST      = 0.01

def prior_correct(p, train_prior=TRAIN_ATTACK_PRIOR,
                   test_prior=TEST_ATTACK_PRIOR, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    odds = p / (1 - p)
    ratio = (test_prior / (1 - test_prior)) / (train_prior / (1 - train_prior))
    adj = odds * ratio
    return adj / (1 + adj)

def quick_triage_score(conf, eps=1e-9):
    conf = np.clip(conf, eps, 1 - eps)
    entropy = -(conf * np.log(conf) + (1 - conf) * np.log(1 - conf))
    entropy_norm = entropy / np.log(2)
    return 0.65 * conf - 0.35 * entropy_norm

# =============================================================================
# LOAD MODEL + DATA
# =============================================================================
print("Loading model and test data...")
mlp = joblib.load(MODEL_PATH)
loader = DataLoader()
loader.scaler = joblib.load(SCALER_PATH)
loader.scaler_fitted = True
print(f"Scaler loaded from {SCALER_PATH}")

df_test = loader.load_test()
X_test, y_test, feature_names, df_test_bal = loader.preprocess(
    df_test,
    target_attack_ratio=TEST_ATTACK_PRIOR,
    target_total_samples=TEST_SIZE
)
X_test = np.nan_to_num(X_test, nan=0.0)

y_prob_raw = mlp.predict_proba(X_test)[:, 1]
y_prob = prior_correct(y_prob_raw)
qts_all = quick_triage_score(y_prob)

# Identify Tier-3-eligible alerts (same cutoff logic as soc_simulation.py,
# using load_factor=0 i.e. the base cutoff, since queue state doesn't
# meaningfully change which alerts are even eligible for the cache)
tier3_mask = qts_all > BASE_HIGH_CUTOFF
tier3_idx = np.where(tier3_mask)[0]
print(f"\nTier-3-eligible alerts in test set: {len(tier3_idx):,} "
      f"({len(tier3_idx)/len(X_test)*100:.2f}% of traffic)")

# =============================================================================
# BUILD THE SAME CACHE soc_simulation.py BUILDS
# =============================================================================
print(f"\nBuilding explanation cache ({CACHE_SIZE} prototypes)...")
np.random.seed(42)
cache_indices = np.random.choice(len(X_test), CACHE_SIZE, replace=False)
cache_X = X_test[cache_indices]
background = shap.sample(X_test, 100, random_state=42)

shap_explainer = shap.Explainer(mlp.predict_proba, background)
lime_explainer = LimeTabularExplainer(
    training_data=X_test[:CACHE_SIZE],
    feature_names=feature_names,
    class_names=["Normal", "Attack"],
    mode="classification",
    discretize_continuous=False
)

def compute_shap_lime(row_vec):
    """Direct, fresh SHAP+LIME computation for a single instance."""
    sv = shap_explainer(row_vec.reshape(1, -1)).values[0][:, 1]
    exp = lime_explainer.explain_instance(
        row_vec, mlp.predict_proba,
        num_features=10, num_samples=300, labels=(1,)
    )
    lv = np.zeros(len(feature_names))
    for f, v in exp.as_list(label=1):
        if f in feature_names:
            lv[feature_names.index(f)] = v
        else:
            f_clean = f.strip()
            for j, fname in enumerate(feature_names):
                if fname.strip().lower() == f_clean.lower():
                    lv[j] = v
                    break
    return sv, lv

shap_cache_map, lime_cache_map = {}, {}
for idx, cidx in enumerate(cache_indices):
    if (idx + 1) % 100 == 0 or idx == 0:
        print(f"  cache {idx+1}/{CACHE_SIZE}...")
    sv, lv = compute_shap_lime(X_test[cidx])
    shap_cache_map[cidx] = sv
    lime_cache_map[cidx] = lv

nn_cache = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(cache_X)
print("Cache ready.\n")

def get_cached_explanation(row_vec):
    _, ni = nn_cache.kneighbors(row_vec.reshape(1, -1))
    cidx = cache_indices[ni[0][0]]
    dist = np.linalg.norm(row_vec - cache_X[ni[0][0]])
    return shap_cache_map[cidx], lime_cache_map[cidx], dist

# =============================================================================
# VALIDATION SAMPLE — draw from Tier-3-eligible alerts NOT already
# in the cache itself (testing on cache members would be circular)
# =============================================================================
np.random.seed(7)
candidate_pool = np.setdiff1d(tier3_idx, cache_indices)
if len(candidate_pool) < N_VALIDATION_ALERTS:
    print(f"WARNING: only {len(candidate_pool)} eligible non-cache alerts "
          f"available; using all of them.")
    val_idx = candidate_pool
else:
    val_idx = np.random.choice(candidate_pool, N_VALIDATION_ALERTS, replace=False)

print(f"Validating cache fidelity on {len(val_idx)} held-out Tier-3 alerts...\n")

# =============================================================================
# COMPARE CACHED vs DIRECT EXPLANATIONS
# =============================================================================
def cosine_sim(a, b, eps=1e-9):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

def top_k_overlap(a, b, k=6):
    """Fraction of top-k |SHAP|-ranked features shared between two vectors."""
    top_a = set(np.argsort(np.abs(a))[-k:])
    top_b = set(np.argsort(np.abs(b))[-k:])
    return len(top_a & top_b) / k

def sign_agreement(a, b, thresh=1e-4):
    """Fraction of jointly-significant features where sign matches."""
    sig = (np.abs(a) > thresh) | (np.abs(b) > thresh)
    if not np.any(sig):
        return np.nan
    return float(np.mean(np.sign(a[sig]) == np.sign(b[sig])))

rows = []
for i, idx in enumerate(val_idx):
    if (i + 1) % 25 == 0 or i == 0:
        print(f"  validating {i+1}/{len(val_idx)}...")

    row = X_test[idx]
    shap_direct, lime_direct = compute_shap_lime(row)
    shap_cached, lime_cached, nn_dist = get_cached_explanation(row)

    shap_cos = cosine_sim(shap_direct, shap_cached)
    lime_cos = cosine_sim(lime_direct, lime_cached)
    shap_top_overlap = top_k_overlap(shap_direct, shap_cached, k=6)
    shap_sign_agree = sign_agreement(shap_direct, shap_cached)
    rank_corr, _ = spearmanr(np.abs(shap_direct), np.abs(shap_cached))

    rows.append({
        "test_idx": idx,
        "nn_distance": round(nn_dist, 4),
        "shap_cosine_sim": round(shap_cos, 4),
        "lime_cosine_sim": round(lime_cos, 4),
        "shap_top6_overlap": round(shap_top_overlap, 4),
        "shap_sign_agreement": round(shap_sign_agree, 4) if not np.isnan(shap_sign_agree) else np.nan,
        "shap_rank_corr": round(rank_corr, 4) if rank_corr is not None else np.nan,
    })

df_fidelity = pd.DataFrame(rows)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print(" CACHE FIDELITY RESULTS")
print("=" * 65)
print(df_fidelity.describe().round(4).to_string())

print(f"""
=== SUMMARY ({len(val_idx)} alerts validated) ===
  Mean SHAP cosine similarity   : {df_fidelity['shap_cosine_sim'].mean():.4f}
  Mean LIME cosine similarity   : {df_fidelity['lime_cosine_sim'].mean():.4f}
  Mean top-6 feature overlap    : {df_fidelity['shap_top6_overlap'].mean():.4f}
  Mean sign agreement           : {df_fidelity['shap_sign_agreement'].mean():.4f}
  Mean |SHAP| rank correlation  : {df_fidelity['shap_rank_corr'].mean():.4f}
  Mean nearest-neighbour dist   : {df_fidelity['nn_distance'].mean():.4f}

  Alerts with SHAP cosine sim < 0.5 : {(df_fidelity['shap_cosine_sim'] < 0.5).sum()} / {len(val_idx)}
  Alerts with top-6 overlap < 0.5   : {(df_fidelity['shap_top6_overlap'] < 0.5).sum()} / {len(val_idx)}
""")

os.makedirs("experiments/results", exist_ok=True)
out_path = "experiments/results/cache_fidelity_results.csv"
df_fidelity.to_csv(out_path, index=False)
print(f"Saved -> {out_path}")
print("\nCache fidelity check complete.")