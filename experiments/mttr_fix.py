"""
mttr_decomposition_standalone.py
=============================================================================
Standalone experiment — does NOT import or modify your main simulation
script. It reloads the model/scaler/data itself and rebuilds its own
explanation cache, so nothing in your existing pipeline is touched.

Purpose
-------
Reviewer 2's objection: ARIA's reported -51.4% MTTR improvement (Table IX)
is partly "baked in" because investigation_time() assumes stronger evidence
(confidence + SHAP + LIME) speeds up investigation, then reports that it
sped up investigation. This script isolates how much of that MTTR drop is
due to that assumption versus how much is structural (ARIA simply queues
fewer / better-prioritised alerts than the baseline).

Method
------
Run the identical simulation loop twice, changing ONLY how `reduction` is
computed inside investigation_time():
  - "full"  mode: reduction = 0.40*conf + 0.35*shap_strength + 0.25*agreement
                  (exactly as in your paper's Eq. 8 / original script)
  - "fixed" mode: reduction = constant, set to the mean reduction observed
                  in the full run. Every escalated alert gets the same
                  per-alert MTTR regardless of its own evidence strength.

Triage tiers, thresholds, queue dynamics, and fatigue are IDENTICAL in both
modes. Only the evidence-dependence of MTTR is switched off in "fixed" mode.

Outputs
-------
Saves to experiments/results/mttr_decomposition/:
  - full_run.csv, fixed_run.csv     (per-window simulation logs)
  - decomposition_summary.csv        (side-by-side comparison table)
Prints an interpretation block you can drop straight into Section IV-B.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import warnings
from lime.lime_tabular import LimeTabularExplainer
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG — mirror your main script's constants exactly
# =============================================================================
MODEL_PATH = "models/mlp_alert_train120k_4attack.pkl"
SCALER_PATH = "models/scaler_train120k_4attack.pkl"
TEST_SIZE = 40000
SIM_STEPS = 40
CACHE_SIZE = 500

TRAIN_ATTACK_PRIOR = 0.40
TEST_ATTACK_PRIOR = 0.05

BASE_HIGH_CUTOFF = 0.15
BASE_MED_CUTOFF = 0.02
LOAD_ADJUST = 0.01

BASE_THRESHOLD = 0.55
THRESHOLD_MAX = 0.70
THRESHOLD_GAIN = 0.05

SOC_TOP_K = 1500
NUM_ANALYSTS = 6
WINDOW_MINUTES = 15
BASE_MTTR = 15.0
LOAD_MTTR_PENALTY = 0.60

FATIGUE_DECAY = 0.935
MIN_FATIGUE = 0.55
MAX_FATIGUE = 1.0

BEST_W = (0.45, 0.40, 0.15)

OUT_DIR = "experiments/results/mttr_decomposition"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print(" MTTR DECOMPOSITION — standalone, independent of main script")
print("=" * 65)

# =============================================================================
# LOAD MODEL, SCALER, DATA  (self-contained — does not touch DataLoader class
# instance from your other script; imports the same DataLoader class fresh)
# =============================================================================
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from data_loader import DataLoader  # noqa: E402

print("Loading model, scaler, and test data…")
mlp = joblib.load(MODEL_PATH)
loader = DataLoader()
loader.scaler = joblib.load(SCALER_PATH)
loader.scaler_fitted = True

df_test = loader.load_test()
X_test, y_test, feature_names, df_test_bal = loader.preprocess(
    df_test,
    target_attack_ratio=TEST_ATTACK_PRIOR,
    target_total_samples=TEST_SIZE
)
X_test = np.nan_to_num(X_test, nan=0.0)
print(f"✅ Loaded {X_test.shape[0]} test samples, {X_test.shape[1]} features")

# =============================================================================
# PRIOR-SHIFT CORRECTION + TRIAGE SCORING  (identical formulas to main script)
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
    entropy_norm = entropy / np.log(2)
    return 0.65 * conf - 0.35 * entropy_norm

def full_rrs_score(conf, shap_vals, lime_vals, w=BEST_W):
    abs_shap = np.abs(shap_vals)
    top5_mean = np.mean(np.sort(abs_shap)[-5:])
    shap_evidence = top5_mean / (np.max(abs_shap) + 1e-9)
    significant = (abs_shap > 1e-4) | (np.abs(lime_vals) > 1e-4)
    agreement = (np.mean(np.sign(shap_vals[significant]) == np.sign(lime_vals[significant]))
                 if np.any(significant) else 0.0)
    rrs = w[0] * conf + w[1] * shap_evidence + w[2] * agreement
    return float(np.clip(rrs, 0.0, 1.0))

def investigation_time(conf, shap_vals, lime_vals, load_factor, mode="full", fixed_reduction=None):
    """
    mode="full"  -> reduction computed per-alert from conf/SHAP/LIME (Eq. 8, as published)
    mode="fixed" -> reduction pinned to a constant, removing evidence-dependence
    """
    if mode == "full":
        abs_shap = np.abs(shap_vals)
        shap_strength = np.mean(np.sort(abs_shap)[-5:]) / (np.sum(abs_shap) + 1e-9)
        agreement = np.mean(np.sign(shap_vals) == np.sign(lime_vals))
        reduction = 0.40 * conf + 0.35 * shap_strength + 0.25 * agreement
    elif mode == "fixed":
        assert fixed_reduction is not None
        reduction = fixed_reduction
    else:
        raise ValueError(mode)

    base = max(3.0, BASE_MTTR * (1 - reduction))
    return base * (1 + LOAD_MTTR_PENALTY * load_factor), reduction

y_prob_raw = mlp.predict_proba(X_test)[:, 1]
y_prob = prior_correct(y_prob_raw)

# =============================================================================
# EXPLANATION CACHE (rebuilt independently — same method as main script)
# =============================================================================
print(f"\nBuilding explanation cache ({CACHE_SIZE} prototypes)…")
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

shap_cache_map = {}
lime_cache_map = {}
for idx, cidx in enumerate(cache_indices):
    if (idx + 1) % 100 == 0 or idx == 0:
        print(f"  {idx+1}/{CACHE_SIZE}…")
    sv = shap_explainer(X_test[cidx:cidx + 1]).values[0][:, 1]
    shap_cache_map[cidx] = sv

    exp = lime_explainer.explain_instance(
        X_test[cidx], mlp.predict_proba,
        num_features=10, num_samples=300, labels=(1,)
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
print("✅ Cache ready\n")

def get_explanations(row_vec):
    _, ni = nn_cache.kneighbors(row_vec.reshape(1, -1))
    cidx = cache_indices[ni[0][0]]
    return shap_cache_map[cidx], lime_cache_map[cidx]

# =============================================================================
# SIMULATION LOOP (mode-switchable)
# =============================================================================
def run_simulation(mode="full", fixed_reduction=None):
    results = []
    queue_size = 0
    fatigue = MIN_FATIGUE
    BATCH_SIZE = TEST_SIZE // SIM_STEPS
    attack_arrival = {}
    attack_detection = {}
    all_reductions = []

    print(f"Running simulation — mode={mode}"
          + (f" (fixed_reduction={fixed_reduction:.4f})" if mode == "fixed" else ""))

    for t in range(SIM_STEPS):
        start = t * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(X_test))
        batch_x = X_test[start:end]
        batch_y = y_test[start:end]
        batch_prob = y_prob[start:end]

        load_factor = queue_size / SOC_TOP_K
        high_cutoff = BASE_HIGH_CUTOFF + LOAD_ADJUST * load_factor
        med_cutoff = BASE_MED_CUTOFF + LOAD_ADJUST * load_factor
        alert_thresh = min(THRESHOLD_MAX, BASE_THRESHOLD + THRESHOLD_GAIN * load_factor)

        rrs_scores = np.zeros(len(batch_x))
        mttr_list = []
        high_count = 0
        medium_count = 0

        for i in range(len(batch_x)):
            conf = batch_prob[i]
            quick_score = quick_triage_score(conf)
            if quick_score > high_cutoff:
                shap_vals, lime_vals = get_explanations(batch_x[i])
                rrs = full_rrs_score(conf, shap_vals, lime_vals)
                rrs_scores[i] = rrs
                high_count += 1
            elif quick_score > med_cutoff:
                rrs_scores[i] = min(quick_score * 0.90, BASE_THRESHOLD * 0.90)
                medium_count += 1
            else:
                rrs_scores[i] = quick_score * 0.70

        rrs_alerts = (rrs_scores > alert_thresh).sum()
        batch_pred = mlp.predict(batch_x)
        raw_alerts = (batch_pred == 1).sum()

        for i in np.where(rrs_scores > alert_thresh)[0]:
            shap_vals, lime_vals = get_explanations(batch_x[i])
            t_invest, reduction_used = investigation_time(
                batch_prob[i], shap_vals, lime_vals, load_factor,
                mode=mode, fixed_reduction=fixed_reduction
            )
            mttr_list.append(t_invest)
            all_reductions.append(reduction_used)

        avg_mttr = np.mean(mttr_list) if mttr_list else BASE_MTTR

        for i in range(len(batch_y)):
            if batch_y[i] == 1:
                gidx = start + i
                attack_arrival.setdefault(gidx, t)
                if gidx not in attack_detection and rrs_scores[i] > alert_thresh:
                    attack_detection[gidx] = t + 1

        drain = NUM_ANALYSTS * WINDOW_MINUTES / avg_mttr
        queue_size = max(0.0, queue_size - drain) + high_count + 0.25 * medium_count
        queue_size = min(int(queue_size), SOC_TOP_K)

        utilization = queue_size / SOC_TOP_K
        stress = min(1.0, utilization * (avg_mttr / BASE_MTTR))
        fatigue_target = MIN_FATIGUE + (MAX_FATIGUE - MIN_FATIGUE) * stress
        fatigue = FATIGUE_DECAY * fatigue + (1 - FATIGUE_DECAY) * fatigue_target
        fatigue = float(np.clip(fatigue, MIN_FATIGUE, MAX_FATIGUE))

        results.append({
            "window": t,
            "raw_alerts": int(raw_alerts),
            "rrs_alerts": int(rrs_alerts),
            "high_tier": high_count,
            "medium_tier": medium_count,
            "queue_size": queue_size,
            "fatigue": round(fatigue, 3),
            "avg_mttr": round(avg_mttr, 2),
        })

    df_sim = pd.DataFrame(results)
    return df_sim, all_reductions, len(attack_detection)

# =============================================================================
# RUN BOTH MODES
# =============================================================================
df_full, reductions_full, attacks_full = run_simulation(mode="full")
mean_reduction_full = float(np.mean(reductions_full))

df_fixed, reductions_fixed, attacks_fixed = run_simulation(
    mode="fixed", fixed_reduction=mean_reduction_full
)

summary = pd.DataFrame({
    "Metric": ["Final fatigue", "Avg MTTR (min)", "Attacks detected", "Max queue"],
    "ARIA (full, evidence-dependent)": [
        df_full["fatigue"].iloc[-1],
        round(df_full["avg_mttr"].mean(), 2),
        attacks_full,
        int(df_full["queue_size"].max()),
    ],
    "ARIA (fixed-reduction, evidence removed)": [
        df_fixed["fatigue"].iloc[-1],
        round(df_fixed["avg_mttr"].mean(), 2),
        attacks_fixed,
        int(df_fixed["queue_size"].max()),
    ],
})

print("\n" + "=" * 65)
print(" DECOMPOSITION RESULT")
print("=" * 65)
print(summary.to_string(index=False))

mttr_full = df_full["avg_mttr"].mean()
mttr_fixed = df_fixed["avg_mttr"].mean()
print(f"""
Interpretation:
  Mean evidence-based reduction pinned in fixed-mode: {mean_reduction_full:.4f}
  Avg MTTR, full (evidence-dependent):   {mttr_full:.2f} min
  Avg MTTR, fixed (evidence removed):    {mttr_fixed:.2f} min

  Compare BOTH against your existing baseline MTTR (Table IX: 18.70 min).
  Gap [baseline -> fixed]  = structural contribution (better prioritisation / queue effects only)
  Gap [fixed -> full]      = contribution from the per-alert evidence-based reduction assumption (Eq. 8)
""")

df_full.to_csv(os.path.join(OUT_DIR, "full_run.csv"), index=False)
df_fixed.to_csv(os.path.join(OUT_DIR, "fixed_run.csv"), index=False)
summary.to_csv(os.path.join(OUT_DIR, "decomposition_summary.csv"), index=False)
print(f"Saved results to {OUT_DIR}/")
print("\n✅ MTTR decomposition complete — main simulation script untouched.")

df_zero, reductions_zero, attacks_zero = run_simulation(mode="fixed", fixed_reduction=0.0)
print(f"Avg MTTR, zero-evidence-effect: {df_zero['avg_mttr'].mean():.2f} min")
print(f"Final fatigue, zero-evidence-effect: {df_zero['fatigue'].iloc[-1]:.3f}")