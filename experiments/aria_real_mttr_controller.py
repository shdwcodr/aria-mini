"""
aria_real_mttr_controller.py
=============================================================================
This is the script for Table 16 / Appendix A.2 (admission-rate sensitivity
at matched low load — see REPRODUCIBILITY.md). It is the sibling of
aria_matched_controller_fix.py, with exactly ONE substantive difference:

  aria_matched_controller_fix.py -> FLAT MTTR (no evidence term)
                                     -> belongs in the sensitivity table
  aria_real_mttr_controller.py   -> REAL MTTR (Eq. 7, evidence-conditioned)
                                     -> belongs in Table 16

Everything else -- QTS, RRS, the cache, the quantile-tracking controller,
the queue/fatigue mechanics -- is identical between the two scripts. This
keeps the comparison to the flat-MTTR sensitivity run clean: the ONLY
thing that changes between the two tables is whether investigation time
depends on the strength of the supporting evidence.

CAPACITY: ANALYST_CAPACITY is set to 4 for this run, to target the
low admission-rate regime used in Table 16 (Appendix A.2), matching the
baseline's ~7 alerts/window drain capacity described in that section.
This is deliberately different from the main-result capacity used in
seeded_variance_comparison.py (Table 12), which targets ~50 alerts/window
for the reported ARIA run (close to, but not identical to, the baseline's
45-alert fixed capacity — see Section 3.9.2).
Do not confuse the two — they answer different questions (high-load vs.
low-load regime) and are not meant to use the same constant.

Run this alongside the baseline run for the same low-load regime and
compare the printed summaries directly. These two outputs are what go
into Table 16.

--- LOGGING ADDED ---
attack_rrs_log now records, for every true attack, the confidence score,
RRS score, and the escalation threshold that was active at the moment
that attack was evaluated. This is written to aria_attack_rrs_log.csv
and is used to check whether the McNemar b=71,c=0 subset pattern
(baseline's detections are a strict subset of ARIA's) is explained by
confidence and RRS being tightly correlated among true attacks
specifically, rather than being a fluke of this one seed/run.
=============================================================================
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

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
MODEL_PATH = "models/mlp_alert_train120k_4attack.pkl"
SCALER_PATH = "models/scaler_train120k_4attack.pkl"   # matched pair, not the 200k/60pct one
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

CACHE_SIZE = 500
BASE_HIGH_CUTOFF = 0.10   # theta0_high, matches paper Table VII
BASE_MED_CUTOFF = 0.03    # theta0_med, matches paper Table VII
LOAD_ADJUST = 0.01

BASE_THRESHOLD = 0.55       # theta0_alert -- "Medium" from Table XII
THRESHOLD_MAX = 0.70
THRESHOLD_GAIN = 0.05
THRESHOLD_MIN = 0.35

# --- Low-load target for the Table 16 / Appendix A.2 sensitivity run
# (paired with a baseline drain capacity of ~7 alerts/window). Do not
# confuse with the Table 12 main-result capacity (~45/window), which is
# set separately in seeded_variance_comparison.py. ---
ANALYST_CAPACITY = 4
EMA_ALPHA = 0.7

def quantile_threshold(scores, k):
    """Score value that admits ~k of len(scores) alerts when using >= ."""
    if k <= 0:
        return np.max(scores) + 1e-6
    if k >= len(scores):
        return np.min(scores)
    return np.sort(scores)[::-1][k - 1]

BEST_W = (0.45, 0.40, 0.15)

OUT_DIR = "experiments/results/aria_real_mttr_controller"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print(" ARIA simulation — REAL evidence-conditioned MTTR (Eq. 7)")
print(f" Quantile-tracking controller, target admission rate: {ANALYST_CAPACITY}/window")
print("=" * 70)

# =============================================================================
# LOAD MODEL, SCALER, DATA
# =============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from data_loader import DataLoader  # noqa: E402

print("Loading model and test data...")
mlp = joblib.load(MODEL_PATH)

loader = DataLoader()
loader.scaler = joblib.load(SCALER_PATH)
loader.scaler_fitted = True
print(f"✅ Scaler loaded from {SCALER_PATH}")

df_test = loader.load_test()
X_test, y_test, feature_names, df_test_bal = loader.preprocess(
    df_test, target_attack_ratio=TEST_ATTACK_PRIOR, target_total_samples=TEST_SIZE
)
X_test = np.nan_to_num(X_test, nan=0.0)
print(f"Test set loaded: {len(X_test):,} samples")
print(f"Attacks: {int(y_test.sum())} ({y_test.mean()*100:.1f}%)")

# =============================================================================
# ARIA scoring functions (identical to published formulas)
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

# --- Eq. 7: REAL, evidence-conditioned investigation time ---
# Identical to soc_simulation.py's investigation_time(). This is the ONLY
# functional difference from aria_matched_controller_fix.py.
def investigation_time(conf, shap_vals, lime_vals, load_factor):
    abs_shap = np.abs(shap_vals)
    shap_strength = (np.mean(np.sort(abs_shap)[-5:]) / (np.sum(abs_shap) + 1e-9))
    agreement = np.mean(np.sign(shap_vals) == np.sign(lime_vals))
    reduction = 0.40 * conf + 0.35 * shap_strength + 0.25 * agreement
    base = max(3.0, BASE_MTTR * (1 - reduction))
    return base * (1 + LOAD_MTTR_PENALTY * load_factor)

y_prob_raw = mlp.predict_proba(X_test)[:, 1]
y_prob = prior_correct(y_prob_raw)

# =============================================================================
# EXPLANATION CACHE
# =============================================================================
print(f"\nBuilding explanation cache ({CACHE_SIZE} prototypes)…")
np.random.seed(42)
cache_indices = np.random.choice(len(X_test), CACHE_SIZE, replace=False)
cache_X = X_test[cache_indices]
background = shap.sample(X_test, 100, random_state=42)

shap_explainer = shap.Explainer(mlp.predict_proba, background)
lime_explainer = LimeTabularExplainer(
    training_data=X_test[:CACHE_SIZE], feature_names=feature_names,
    class_names=["Normal", "Attack"], mode="classification", discretize_continuous=False
)

shap_cache_map, lime_cache_map = {}, {}
for idx, cidx in enumerate(cache_indices):
    if (idx + 1) % 100 == 0 or idx == 0:
        print(f"  {idx+1}/{CACHE_SIZE}…")
    sv = shap_explainer(X_test[cidx:cidx + 1]).values[0][:, 1]
    shap_cache_map[cidx] = sv
    exp = lime_explainer.explain_instance(
        X_test[cidx], mlp.predict_proba, num_features=10, num_samples=300, labels=(1,)
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
# SIMULATION — REAL MTTR, controller-based threshold
# =============================================================================
results = []
queue_size = 0
fatigue = MIN_FATIGUE
attack_arrival = {}
attack_detection = {}
attack_rrs_log = {}   # gidx -> (confidence, rrs_score, threshold_at_time)

alert_thresh = BASE_THRESHOLD

print("Running ARIA simulation (real MTTR, rate-controlled threshold)...")

for t in range(SIM_STEPS):
    start = t * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(y_test))

    batch_x = X_test[start:end]
    batch_y = y_test[start:end]
    batch_prob = y_prob[start:end]

    load_factor = queue_size / SOC_TOP_K

    # ---- QTS cutoffs (unchanged) ----
    high_cutoff = BASE_HIGH_CUTOFF + LOAD_ADJUST * load_factor
    med_cutoff = BASE_MED_CUTOFF + LOAD_ADJUST * load_factor

    rrs_scores = np.zeros(len(batch_x))
    medium_count = 0
    # cache explanations per-instance this window so we don't recompute
    # the nearest-neighbour lookup twice (once for RRS, once for MTTR)
    explanations_this_batch = {}

    for i in range(len(batch_x)):
        conf = batch_prob[i]
        quick_score = quick_triage_score(conf)
        if quick_score > high_cutoff:
            shap_vals, lime_vals = get_explanations(batch_x[i])
            explanations_this_batch[i] = (shap_vals, lime_vals)
            rrs_scores[i] = full_rrs_score(conf, shap_vals, lime_vals)
        elif quick_score > med_cutoff:
            rrs_scores[i] = min(quick_score * 0.90, BASE_THRESHOLD * 0.90)
            medium_count += 1
        else:
            rrs_scores[i] = quick_score * 0.70

    # ---- admit at CURRENT threshold (set by controller from prior window) ----
    escalated_mask = rrs_scores >= alert_thresh
    alert_count = int(escalated_mask.sum())
    raw_alert_count = int((batch_prob > 0.5).sum())

    # ---- REAL MTTR: computed per escalated alert using Eq. 7, then averaged ----
    mttr_list = []
    for i in np.where(escalated_mask)[0]:
        if i in explanations_this_batch:
            shap_vals, lime_vals = explanations_this_batch[i]
        else:
            # escalated without explanations (shouldn't normally happen,
            # since medium-tier scores are capped below alert_thresh, but
            # guard against it so the sim doesn't crash if thresholds drift)
            shap_vals, lime_vals = get_explanations(batch_x[i])
        mttr_list.append(investigation_time(batch_prob[i], shap_vals, lime_vals, load_factor))
    avg_mttr = np.mean(mttr_list) if mttr_list else BASE_MTTR

    drain_capacity = NUM_ANALYSTS * WINDOW_MINUTES / avg_mttr

    for i in range(len(batch_y)):
        if batch_y[i] == 1:
            gidx = start + i
            attack_arrival.setdefault(gidx, t)
            # log confidence, RRS score, and the threshold active at this
            # moment for every true attack, regardless of outcome -- lets
            # us check post hoc why baseline's detections are a subset of
            # ARIA's (McNemar b=71, c=0)
            attack_rrs_log[gidx] = (float(batch_prob[i]), float(rrs_scores[i]), float(alert_thresh))
            if gidx not in attack_detection and escalated_mask[i]:
                attack_detection[gidx] = t + 1

    # ---- queue/fatigue update (identical mechanism to baseline) ----
    queue_size = max(0.0, queue_size - drain_capacity) + alert_count + 0.25 * medium_count
    queue_size = min(int(queue_size), SOC_TOP_K)

    utilization = queue_size / SOC_TOP_K
    stress = min(1.0, utilization * (avg_mttr / BASE_MTTR))
    fatigue_target = MIN_FATIGUE + (MAX_FATIGUE - MIN_FATIGUE) * stress
    fatigue = FATIGUE_DECAY * fatigue + (1 - FATIGUE_DECAY) * fatigue_target
    fatigue = float(np.clip(fatigue, MIN_FATIGUE, MAX_FATIGUE))

    results.append({
        "window": t,
        "raw_alerts": raw_alert_count,
        "aria_alerts": alert_count,
        "medium_tier": medium_count,
        "drain_capacity": round(drain_capacity, 2),
        "alert_thresh": round(alert_thresh, 3),
        "queue_size": queue_size,
        "fatigue": round(fatigue, 3),
        "avg_mttr": round(avg_mttr, 2),
    })

    # ---- CONTROLLER STEP: quantile-tracking, EMA-smoothed, targets 50 ----
    window_target_thresh = quantile_threshold(rrs_scores, ANALYST_CAPACITY)
    alert_thresh = EMA_ALPHA * alert_thresh + (1 - EMA_ALPHA) * window_target_thresh
    alert_thresh = float(np.clip(alert_thresh, THRESHOLD_MIN, THRESHOLD_MAX))

df_aria = pd.DataFrame(results)
print(df_aria)

detected = len(attack_detection)
delays = [attack_detection[idx] - attack_arrival[idx] for idx in attack_detection]
avg_delay = np.mean(delays) if delays else np.nan
avg_delay_minutes = avg_delay * WINDOW_MINUTES

mean_admitted = df_aria["aria_alerts"].mean()

print("\n=== ARIA (real MTTR, controller-fixed) SUMMARY ===")
print("Attacks detected      :", detected)
print("Final fatigue         :", fatigue)
print("Avg MTTR              :", df_aria["avg_mttr"].mean())
print("Max queue             :", df_aria["queue_size"].max())
print("Final threshold       :", alert_thresh)
print("Mean admitted/window  :", round(mean_admitted, 2),
      f"(target was {ANALYST_CAPACITY} — check this is close before using in Table IX)")
print("Detection delay       :", avg_delay_minutes, "minutes")

# detected-attack-ID export (unchanged from before)
detected_ids = sorted(attack_detection.keys())
pd.Series(detected_ids, name="global_idx").to_csv(
    os.path.join(OUT_DIR, "aria_detected_attack_ids.csv"), index=False
)

# NEW: per-attack confidence/RRS/threshold export
rrs_log_path = os.path.join(OUT_DIR, "aria_attack_rrs_log.csv")
pd.DataFrame([
    {"global_idx": gidx, "confidence": c, "rrs": r, "threshold": th}
    for gidx, (c, r, th) in attack_rrs_log.items()
]).to_csv(rrs_log_path, index=False)
print(f"✅ Attack-level RRS log saved: {rrs_log_path} ({len(attack_rrs_log)} rows)")

csv_path = os.path.join(OUT_DIR, "aria_real_mttr_simulation.csv")
df_aria.to_csv(csv_path, index=False)
print(f"\n✅ Saved to: {csv_path}")
print("Compare directly against baseline.py's printed summary (TOP_K=50).")