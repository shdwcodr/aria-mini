"""
seeded_variance_comparison.py
=============================================================================
Runs the baseline and ARIA (real-MTTR) simulations across N_SEEDS bootstrap
resamples of the 40,000-sample deployment test pool, using the SAME
resampled stream (same order, same rows) for both methods on a given seed.
This keeps the paired comparison (McNemar) valid seed-by-seed, and lets us
report mean +/- 95% CI on every headline number instead of a single
deterministic run.

Expensive one-time setup (model load, SHAP/LIME cache build) happens
ONCE before the seed loop -- the cache is treated as fixed infrastructure
built before deployment, matching the paper's framing (Section III-F).
Only the alert stream itself varies across seeds.

Usage:
    python seeded_variance_comparison.py

Outputs:
    experiments/results/seeded/per_seed_results.csv
    experiments/results/seeded/aggregate_summary.csv
    experiments/results/seeded/pooled_mcnemar.txt
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap
import warnings
from lime.lime_tabular import LimeTabularExplainer
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from data_loader import DataLoader  # noqa: E402

# =============================================================================
# CONFIG (unchanged from the two source scripts)
# =============================================================================
MODEL_PATH = "models/mlp_alert_train120k_4attack.pkl"
SCALER_PATH = "models/scaler_train120k_4attack.pkl"
TEST_SIZE = 40000
TEST_ATTACK_PRIOR = 0.05
TRAIN_ATTACK_PRIOR = 0.40

SIM_STEPS = 40
BATCH_SIZE = 1000
SOC_TOP_K = 1500
NUM_ANALYSTS = 6
WINDOW_MINUTES = 15
BASE_MTTR = 15.0
LOAD_MTTR_PENALTY = 0.60
FATIGUE_DECAY = 0.935
MIN_FATIGUE = 0.55
MAX_FATIGUE = 1.0

BASELINE_CAPACITY = 45          # baseline.py's fixed top-K
ARIA_TARGET_CAPACITY = 50       # aria_real_mttr_controller.py's controller target
ALERT_THRESHOLD_RAW = 0.5       # baseline's raw_alert_count reporting only

CACHE_SIZE = 500
BASE_HIGH_CUTOFF = 0.10
BASE_MED_CUTOFF = 0.03
LOAD_ADJUST = 0.01
BASE_THRESHOLD = 0.55
THRESHOLD_MAX = 0.70
THRESHOLD_MIN = 0.35
EMA_ALPHA = 0.7
BEST_W = (0.45, 0.40, 0.15)

N_SEEDS = 30
OUT_DIR = "experiments/results/seeded"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# SHARED SCORING FUNCTIONS
# =============================================================================
def prior_correct(p, train_prior=TRAIN_ATTACK_PRIOR, test_prior=TEST_ATTACK_PRIOR, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    odds = p / (1 - p)
    ratio = (test_prior / (1 - test_prior)) / (train_prior / (1 - train_prior))
    adj = odds * ratio
    return adj / (1 + adj)

def quick_triage_score(conf, eps=1e-9):
    conf = np.clip(conf, eps, 1 - eps)
    entropy = -(conf * np.log(conf) + (1 - conf) * np.log(1 - conf))
    return 0.65 * conf - 0.35 * (entropy / np.log(2))

def full_rrs_score(conf, shap_vals, lime_vals, w=BEST_W):
    abs_shap = np.abs(shap_vals)
    top5_mean = np.mean(np.sort(abs_shap)[-5:])
    shap_evidence = top5_mean / (np.max(abs_shap) + 1e-9)
    significant = (abs_shap > 1e-4) | (np.abs(lime_vals) > 1e-4)
    agreement = (np.mean(np.sign(shap_vals[significant]) == np.sign(lime_vals[significant]))
                 if np.any(significant) else 0.0)
    return float(np.clip(w[0] * conf + w[1] * shap_evidence + w[2] * agreement, 0.0, 1.0))

def investigation_time(conf, shap_vals, lime_vals, load_factor):
    abs_shap = np.abs(shap_vals)
    shap_strength = np.mean(np.sort(abs_shap)[-5:]) / (np.sum(abs_shap) + 1e-9)
    agreement = np.mean(np.sign(shap_vals) == np.sign(lime_vals))
    reduction = 0.40 * conf + 0.35 * shap_strength + 0.25 * agreement
    base = max(3.0, BASE_MTTR * (1 - reduction))
    return base * (1 + LOAD_MTTR_PENALTY * load_factor)

def quantile_threshold(scores, k):
    if k <= 0:
        return np.max(scores) + 1e-6
    if k >= len(scores):
        return np.min(scores)
    return np.sort(scores)[::-1][k - 1]

# =============================================================================
# ONE-TIME SETUP: model, data, cache
# =============================================================================
print("Loading model and full test pool...")
mlp = joblib.load(MODEL_PATH)
loader = DataLoader()
loader.scaler = joblib.load(SCALER_PATH)
loader.scaler_fitted = True

df_test = loader.load_test()
X_full, y_full, feature_names, df_test_bal = loader.preprocess(
    df_test, target_attack_ratio=TEST_ATTACK_PRIOR, target_total_samples=TEST_SIZE
)
X_full = np.nan_to_num(X_full, nan=0.0)
N = len(X_full)
print(f"Base pool: {N:,} samples, {int(y_full.sum())} attacks")

y_prob_raw_full = mlp.predict_proba(X_full)[:, 1]
y_prob_baseline_full = y_prob_raw_full                 # baseline: NO prior correction (by design)
y_prob_aria_full = prior_correct(y_prob_raw_full)      # ARIA: prior-corrected

print(f"\nBuilding explanation cache ({CACHE_SIZE} prototypes, fixed across all seeds)...")
np.random.seed(42)
cache_indices = np.random.choice(N, CACHE_SIZE, replace=False)
cache_X = X_full[cache_indices]
background = shap.sample(X_full, 100, random_state=42)
shap_explainer = shap.Explainer(mlp.predict_proba, background)
lime_explainer = LimeTabularExplainer(
    training_data=X_full[:CACHE_SIZE], feature_names=feature_names,
    class_names=["Normal", "Attack"], mode="classification", discretize_continuous=False
)

shap_cache_map, lime_cache_map = {}, {}
for idx, cidx in enumerate(cache_indices):
    if (idx + 1) % 100 == 0 or idx == 0:
        print(f"  cache {idx+1}/{CACHE_SIZE}...")
    sv = shap_explainer(X_full[cidx:cidx + 1]).values[0][:, 1]
    shap_cache_map[cidx] = sv
    exp = lime_explainer.explain_instance(
        X_full[cidx], mlp.predict_proba, num_features=10, num_samples=300, labels=(1,)
    )
    lime_vec = np.zeros(len(feature_names))
    for f, v in exp.as_list(label=1):
        if f in feature_names:
            lime_vec[feature_names.index(f)] = v
        else:
            f_clean = f.strip()
            for j, fname in enumerate(feature_names):
                if fname.strip().lower() == f_clean.lower():
                    lime_vec[j] = v
                    break
    lime_cache_map[cidx] = lime_vec

nn_cache = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(cache_X)
print("Cache ready.\n")

def get_explanations(row_vec):
    _, ni = nn_cache.kneighbors(row_vec.reshape(1, -1))
    cidx = cache_indices[ni[0][0]]
    return shap_cache_map[cidx], lime_cache_map[cidx]

# =============================================================================
# BASELINE SIM (single resampled stream)
# =============================================================================
def run_baseline(X, y, y_prob):
    queue_size, fatigue = 0, MIN_FATIGUE
    attack_arrival, attack_detection = {}, {}
    mttr_list_all = []
    for t in range(SIM_STEPS):
        s, e = t * BATCH_SIZE, min((t + 1) * BATCH_SIZE, len(y))
        by, bp = y[s:e], y_prob[s:e]
        load_factor = queue_size / SOC_TOP_K
        avg_mttr = BASE_MTTR * (1 + LOAD_MTTR_PENALTY * load_factor)
        ranked = np.argsort(bp)[::-1]
        top_k = np.zeros(len(by), dtype=bool)
        top_k[ranked[:BASELINE_CAPACITY]] = True
        alert_count = int(top_k.sum())
        for i in range(len(by)):
            if by[i] == 1:
                gidx = s + i
                attack_arrival.setdefault(gidx, t)
                if gidx not in attack_detection and top_k[i]:
                    attack_detection[gidx] = t + 1
        drain = NUM_ANALYSTS * WINDOW_MINUTES / avg_mttr
        queue_size = min(int(max(0.0, queue_size - drain) + alert_count), SOC_TOP_K)
        utilization = queue_size / SOC_TOP_K
        stress = min(1.0, utilization * (avg_mttr / BASE_MTTR))
        target = MIN_FATIGUE + (MAX_FATIGUE - MIN_FATIGUE) * stress
        fatigue = float(np.clip(FATIGUE_DECAY * fatigue + (1 - FATIGUE_DECAY) * target, MIN_FATIGUE, MAX_FATIGUE))
        mttr_list_all.append(avg_mttr)
    return {
        "final_fatigue": fatigue,
        "avg_mttr": float(np.mean(mttr_list_all)),
        "max_queue": queue_size if queue_size == SOC_TOP_K else max(0, queue_size),
        "attacks_detected": len(attack_detection),
        "detected_set": set(attack_detection.keys()),
    }

# =============================================================================
# ARIA SIM (single resampled stream)
# =============================================================================
def run_aria(X, y, y_prob):
    queue_size, fatigue = 0, MIN_FATIGUE
    attack_arrival, attack_detection = {}, {}
    alert_thresh = BASE_THRESHOLD
    mttr_list_all = []
    for t in range(SIM_STEPS):
        s, e = t * BATCH_SIZE, min((t + 1) * BATCH_SIZE, len(y))
        bx, by, bp = X[s:e], y[s:e], y_prob[s:e]
        load_factor = queue_size / SOC_TOP_K
        high_cutoff = BASE_HIGH_CUTOFF + LOAD_ADJUST * load_factor
        med_cutoff = BASE_MED_CUTOFF + LOAD_ADJUST * load_factor
        rrs_scores = np.zeros(len(bx))
        medium_count = 0
        expl_batch = {}
        for i in range(len(bx)):
            conf = bp[i]
            qs = quick_triage_score(conf)
            if qs > high_cutoff:
                sv, lv = get_explanations(bx[i])
                expl_batch[i] = (sv, lv)
                rrs_scores[i] = full_rrs_score(conf, sv, lv)
            elif qs > med_cutoff:
                rrs_scores[i] = min(qs * 0.90, BASE_THRESHOLD * 0.90)
                medium_count += 1
            else:
                rrs_scores[i] = qs * 0.70
        escalated = rrs_scores >= alert_thresh
        alert_count = int(escalated.sum())
        mttr_list = []
        for i in np.where(escalated)[0]:
            sv, lv = expl_batch[i] if i in expl_batch else get_explanations(bx[i])
            mttr_list.append(investigation_time(bp[i], sv, lv, load_factor))
        avg_mttr = np.mean(mttr_list) if mttr_list else BASE_MTTR
        drain = NUM_ANALYSTS * WINDOW_MINUTES / avg_mttr
        for i in range(len(by)):
            if by[i] == 1:
                gidx = s + i
                attack_arrival.setdefault(gidx, t)
                if gidx not in attack_detection and escalated[i]:
                    attack_detection[gidx] = t + 1
        queue_size = min(int(max(0.0, queue_size - drain) + alert_count + 0.25 * medium_count), SOC_TOP_K)
        utilization = queue_size / SOC_TOP_K
        stress = min(1.0, utilization * (avg_mttr / BASE_MTTR))
        target = MIN_FATIGUE + (MAX_FATIGUE - MIN_FATIGUE) * stress
        fatigue = float(np.clip(FATIGUE_DECAY * fatigue + (1 - FATIGUE_DECAY) * target, MIN_FATIGUE, MAX_FATIGUE))
        mttr_list_all.append(avg_mttr)
        window_target = quantile_threshold(rrs_scores, ARIA_TARGET_CAPACITY)
        alert_thresh = float(np.clip(EMA_ALPHA * alert_thresh + (1 - EMA_ALPHA) * window_target,
                                      THRESHOLD_MIN, THRESHOLD_MAX))
    return {
        "final_fatigue": fatigue,
        "avg_mttr": float(np.mean(mttr_list_all)),
        "max_queue": max(0, queue_size),
        "attacks_detected": len(attack_detection),
        "detected_set": set(attack_detection.keys()),
    }

# =============================================================================
# SEED LOOP: bootstrap-resample the SAME stream for both methods per seed
# =============================================================================
per_seed_rows = []
pooled_b, pooled_c = 0, 0   # aggregate McNemar counts across seeds

print(f"Running {N_SEEDS} bootstrap-resampled seeds (baseline + ARIA, paired)...")
for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=N, replace=True)   # bootstrap resample, same order used by both methods

    X_s, y_s = X_full[idx], y_full[idx]
    prob_base_s = y_prob_baseline_full[idx]
    prob_aria_s = y_prob_aria_full[idx]

    base_res = run_baseline(X_s, y_s, prob_base_s)
    aria_res = run_aria(X_s, y_s, prob_aria_s)

    # paired detection outcome per resampled row (local index space, valid within this seed only)
    detected_by_baseline_only = len(base_res["detected_set"] - aria_res["detected_set"])
    detected_by_aria_only = len(aria_res["detected_set"] - base_res["detected_set"])
    pooled_b += detected_by_aria_only     # ARIA caught, baseline missed
    pooled_c += detected_by_baseline_only  # baseline caught, ARIA missed

    per_seed_rows.append({
        "seed": seed,
        "baseline_final_fatigue": base_res["final_fatigue"],
        "aria_final_fatigue": aria_res["final_fatigue"],
        "baseline_avg_mttr": base_res["avg_mttr"],
        "aria_avg_mttr": aria_res["avg_mttr"],
        "baseline_max_queue": base_res["max_queue"],
        "aria_max_queue": aria_res["max_queue"],
        "baseline_attacks_detected": base_res["attacks_detected"],
        "aria_attacks_detected": aria_res["attacks_detected"],
        "b_aria_only": detected_by_aria_only,
        "c_baseline_only": detected_by_baseline_only,
    })
    print(f"  seed {seed+1}/{N_SEEDS} done "
          f"(baseline fatigue={base_res['final_fatigue']:.3f}, aria fatigue={aria_res['final_fatigue']:.3f})")

df_seeds = pd.DataFrame(per_seed_rows)
df_seeds.to_csv(os.path.join(OUT_DIR, "per_seed_results.csv"), index=False)

# =============================================================================
# AGGREGATE: mean +/- 95% CI (percentile method) per metric
# =============================================================================
def summarize(col):
    vals = df_seeds[col].to_numpy()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"mean": vals.mean(), "std": vals.std(ddof=1), "ci_lo": lo, "ci_hi": hi}

metrics = [
    ("baseline_final_fatigue", "aria_final_fatigue", "Final fatigue"),
    ("baseline_avg_mttr", "aria_avg_mttr", "Avg MTTR (min)"),
    ("baseline_attacks_detected", "aria_attacks_detected", "Attacks detected"),
    ("baseline_max_queue", "aria_max_queue", "Max queue size"),
]

summary_rows = []
for base_col, aria_col, label in metrics:
    b_summary = summarize(base_col)
    a_summary = summarize(aria_col)
    summary_rows.append({
        "metric": label,
        "baseline_mean": round(b_summary["mean"], 4),
        "baseline_ci": f"[{b_summary['ci_lo']:.4f}, {b_summary['ci_hi']:.4f}]",
        "aria_mean": round(a_summary["mean"], 4),
        "aria_ci": f"[{a_summary['ci_lo']:.4f}, {a_summary['ci_hi']:.4f}]",
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_DIR, "aggregate_summary.csv"), index=False)
print("\n=== AGGREGATE SUMMARY (mean, 95% CI across {} seeds) ===".format(N_SEEDS))
print(df_summary.to_string(index=False))

# =============================================================================
# POOLED McNEMAR (aggregated b, c across all seeds)
# =============================================================================
if pooled_b + pooled_c > 0:
    chi_sq = (abs(pooled_b - pooled_c) - 1) ** 2 / (pooled_b + pooled_c)
    p_value = 1 - chi2.cdf(chi_sq, df=1)
else:
    chi_sq, p_value = 0.0, 1.0

mcnemar_text = (
    f"Pooled McNemar's test across {N_SEEDS} bootstrap-resampled seeds\n"
    f"  b (ARIA caught, baseline missed), summed across seeds : {pooled_b}\n"
    f"  c (baseline caught, ARIA missed), summed across seeds : {pooled_c}\n"
    f"  chi-squared (continuity corrected) : {chi_sq:.4f}\n"
    f"  p-value : {p_value:.6g}\n"
)
print("\n" + mcnemar_text)
with open(os.path.join(OUT_DIR, "pooled_mcnemar.txt"), "w") as f:
    f.write(mcnemar_text)

print(f"\nAll outputs saved to {OUT_DIR}/")