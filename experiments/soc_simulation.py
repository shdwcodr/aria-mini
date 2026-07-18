import numpy as np
import pandas as pd
import joblib
import warnings
import os
import sys
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.neighbors import NearestNeighbors
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data_loader import DataLoader
from data_loader import DataLoader   # We'll fix the import below

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
MODEL_PATH = "models/mlp_alert_train300k_5attack.pkl"
TEST_SIZE = 40000
SIM_STEPS = 40
CACHE_SIZE = 500

TRAIN_ATTACK_PRIOR = 0.55
TEST_ATTACK_PRIOR = 0.05

BASE_HIGH_CUTOFF = 0.03
BASE_MED_CUTOFF = 0.01
LOAD_ADJUST = 0.03

RAW_ALERT_FLOOR = 0.01

BASE_THRESHOLD = 0.19
THRESHOLD_MAX = 0.27
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

print("=" * 65)
print(" ARIA SOC Simulation — RRS + Entropy QTS")
print("=" * 65)

# =============================================================================
# FUNCTIONS (same as before)
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

def investigation_time(conf, shap_vals, lime_vals, load_factor):
    abs_shap = np.abs(shap_vals)
    shap_strength = (np.mean(np.sort(abs_shap)[-5:]) / (np.sum(abs_shap) + 1e-9))
    agreement = np.mean(np.sign(shap_vals) == np.sign(lime_vals))
    reduction = 0.40 * conf + 0.35 * shap_strength + 0.25 * agreement
    base = max(3.0, BASE_MTTR * (1 - reduction))
    return base * (1 + LOAD_MTTR_PENALTY * load_factor)

def drill_down(idx_in_batch, conf, rrs, shap_vals, lime_vals, feature_names, row_series):
    abs_shap = np.abs(shap_vals)
    top_idx = np.argsort(abs_shap)[-6:][::-1]
    top_features = [feature_names[i].lower() for i in top_idx]
    patterns = set()
    if any("syn" in f for f in top_features): patterns.add("SYN behaviour anomaly")
    if any("ack" in f for f in top_features): patterns.add("ACK pattern deviation")
    if any("urg" in f for f in top_features): patterns.add("Urgency flag abnormality")
    if any("header" in f for f in top_features): patterns.add("Packet structure deviation")
    if any("byte" in f for f in top_features): patterns.add("Abnormal byte volume")
    if any("dur" in f for f in top_features): patterns.add("Unusual flow duration")
    if any("pkt" in f for f in top_features): patterns.add("Packet rate anomaly")
    if any("port" in f for f in top_features): patterns.add("Suspicious port usage")
    
    drivers = []
    for i in top_idx:
        fname = feature_names[i]
        fval = row_series.iloc[i] if not pd.isna(row_series.iloc[i]) else 0.0
        s_sign = np.sign(shap_vals[i])
        l_sign = np.sign(lime_vals[i])
        agree = "✓" if s_sign == l_sign else "✗"
        direction = "→ attack" if shap_vals[i] > 0 else "→ benign"
        drivers.append({
            "feature": fname,
            "value": round(float(fval), 4),
            "shap": round(float(shap_vals[i]), 4),
            "lime": round(float(lime_vals[i]), 4),
            "agree": agree,
            "direction": direction
        })
    
    positive_drivers = [d for d in drivers if d["shap"] > 0]
    if len(positive_drivers) >= 4:
        verdict = "HIGH CONFIDENCE ATTACK"
    elif len(positive_drivers) >= 2:
        verdict = "MEDIUM — SUSPICIOUS ACTIVITY"
    else:
        verdict = "LOW SIGNAL — REVIEW RECOMMENDED"
    
    return {
        "batch_idx": idx_in_batch,
        "conf": round(conf, 3),
        "rrs": round(rrs, 3),
        "verdict": verdict,
        "patterns": sorted(patterns),
        "drivers": drivers
    }

def format_drill_down(d):
    lines = [
        f"\n CASE (batch_idx={d['batch_idx']})",
        f" CONF: {d['conf']} RRS: {d['rrs']} VERDICT: {d['verdict']}",
        " ATTACK PATTERNS DETECTED:"
    ]
    for p in d["patterns"]:
        lines.append(f" • {p}")
    lines.append(" TOP FEATURE DRIVERS:")
    for dr in d["drivers"]:
        lines.append(
            f" {dr['feature']:35s} val={dr['value']:>8.4f} "
            f"shap={dr['shap']:>7.4f} lime={dr['lime']:>7.4f} {dr['agree']}"
        )
    lines.append("-" * 60)
    return "\n".join(lines)

# =============================================================================
# LOAD MODEL + DATA
# =============================================================================
print("Loading model and test data...")
mlp = joblib.load(MODEL_PATH)
loader = DataLoader()

if os.path.exists("models/scaler_train200k_60pct.pkl"):
    loader.scaler = joblib.load("models/scaler_train200k_60pct.pkl")
    loader.scaler_fitted = True
    print("✅ Scaler loaded successfully")

df_test = loader.load_test()

X_test, y_test, feature_names, df_test_bal = loader.preprocess(
    df_test,
    target_attack_ratio=TEST_ATTACK_PRIOR,
    target_total_samples=TEST_SIZE
)

X_test = np.nan_to_num(X_test, nan=0.0)

print(f"Test set: {len(X_test):,} samples | Attacks: {y_test.mean():.1%}")

y_prob_raw = mlp.predict_proba(X_test)[:, 1]
y_prob = prior_correct(y_prob_raw)

# =============================================================================
# EXPLANATION CACHE + SIMULATION (rest of the code)
# =============================================================================
print(f"\nBuilding explanation cache ({CACHE_SIZE} samples)...")
np.random.seed(42)
cache_indices = np.random.choice(len(X_test), CACHE_SIZE, replace=False)
cache_X = X_test[cache_indices]
background = shap.sample(X_test, 100, random_state=42)

shap_explainer = shap.Explainer(mlp.predict_proba, background)
lime_explainer = LimeTabularExplainer(
    training_data=X_test[:CACHE_SIZE],
    feature_names=feature_names,
    class_names=["Normal", "Attack"],
    mode="classification"
)

shap_cache_map = {}
lime_cache_map = {}
for idx, cidx in enumerate(cache_indices):
    if (idx + 1) % 100 == 0 or idx == 0:
        print(f" {idx+1}/{CACHE_SIZE}...")
    sv = shap_explainer(X_test[cidx:cidx+1]).values[0][:, 1]
    shap_cache_map[cidx] = sv
    exp = lime_explainer.explain_instance(X_test[cidx], mlp.predict_proba,
                                        num_features=10, num_samples=300, labels=(1,))
    lime_vec = np.zeros(len(feature_names))
    for f, v in exp.as_list(label=1):
        try:
            lime_vec[feature_names.index(f)] = v
        except ValueError:
            pass
    lime_cache_map[cidx] = lime_vec

nn_cache = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(cache_X)
print("✅ Cache ready\n")

def get_explanations(row_vec):
    _, ni = nn_cache.kneighbors(row_vec.reshape(1, -1))
    cidx = cache_indices[ni[0][0]]
    return shap_cache_map[cidx], lime_cache_map[cidx]

# SIMULATION LOOP (same as previous full version)
results = []
drill_log = []
queue_size = 0
fatigue = MIN_FATIGUE
BATCH_SIZE = TEST_SIZE // SIM_STEPS
attack_arrival = {}
attack_detection = {}

print("Running ARIA SOC Simulation...\n")

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
            if rrs > alert_thresh:
                dd = drill_down(
                    start + i, conf, rrs, shap_vals, lime_vals,
                    feature_names, pd.Series(batch_x[i])
                )
                dd["window"] = t
                drill_log.append(dd)
        elif quick_score > med_cutoff:
            rrs_scores[i] = min(quick_score * 0.90, BASE_THRESHOLD * 0.90)
            medium_count += 1
        else:
            rrs_scores[i] = quick_score * 0.70
    
    raw_alerts = (batch_prob > RAW_ALERT_FLOOR).sum()
    rrs_alerts = (rrs_scores > alert_thresh).sum()
    
    for i in np.where(rrs_scores > alert_thresh)[0]:
        shap_vals, lime_vals = get_explanations(batch_x[i])
        mttr_list.append(investigation_time(batch_prob[i], shap_vals, lime_vals, load_factor))
    avg_mttr = np.mean(mttr_list) if mttr_list else BASE_MTTR
    
    for i in range(len(batch_y)):
        if batch_y[i] == 1:
            gidx = start + i
            attack_arrival.setdefault(gidx, t)
            if gidx not in attack_detection and rrs_scores[i] > alert_thresh:
                attack_detection[gidx] = t + 1
    
    drain = NUM_ANALYSTS * WINDOW_MINUTES / avg_mttr
    queue_size = max(0.0, queue_size - drain) + high_count + 0.05 * medium_count
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
        "avg_mttr": round(avg_mttr, 2)
    })

df_sim = pd.DataFrame(results)

# Summary, save, plot (same as before)
delays = [attack_detection[idx] - arr for idx, arr in attack_arrival.items() if idx in attack_detection]
avg_delay = np.mean(delays) if delays else 0

print("\n" + "=" * 65)
print("ARIA SIMULATION RESULTS")
print("=" * 65)
print(df_sim.round(3).to_string())

print(f"""
=== FINAL SUMMARY ===
  Attacks in test set     : {int(y_test.sum())}
  Attacks detected        : {len(attack_detection)}
  Avg detection delay     : {avg_delay:.2f} windows
  Final fatigue           : {df_sim['fatigue'].iloc[-1]:.3f}
  Average MTTR            : {df_sim['avg_mttr'].mean():.2f} min
  Max queue size          : {df_sim['queue_size'].max()}
  High-tier explanations  : {len(drill_log)}
""")

os.makedirs("experiments/results", exist_ok=True)

with open("experiments/results/aria_drill_down_report.txt", "w") as f:
    f.write("ARIA SOC — HIGH-ALERT DRILL-DOWN REPORT\n")
    f.write("=" * 65 + "\n")
    f.write(f"Total HIGH alerts : {len(drill_log)}\n\n")
    prev = -1
    for d in drill_log:
        if d["window"] != prev:
            f.write(f"\n{'─'*65}\n WINDOW {d['window']}\n{'─'*65}\n")
            prev = d["window"]
        f.write(format_drill_down(d))

df_sim.to_csv("experiments/results/aria_soc_simulation.csv", index=False)

# Plot
fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.plot(df_sim["window"], df_sim["raw_alerts"], label="Raw Alerts (p > 0.01)", color="steelblue", linewidth=2)
ax1.plot(df_sim["window"], df_sim["rrs_alerts"], label="ARIA RRS Alerts", color="seagreen", linewidth=2)
ax1.plot(df_sim["window"], df_sim["queue_size"], label="Queue Size", color="darkorange", linewidth=2)
ax1.set_xlabel("Time Window")
ax1.set_ylabel("Alerts / Queue")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(df_sim["window"], df_sim["fatigue"], label="Fatigue", color="crimson", linewidth=2.5)
ax2.set_ylabel("Fatigue Level")
ax2.set_ylim(0, 1.05)
ax2.legend(loc="upper right")

plt.title("ARIA SOC Simulation — RRS + Entropy-based QTS")
plt.tight_layout()
plt.savefig("experiments/results/aria_simulation_final.png", dpi=300)
plt.show()

print("\n✅ ARIA Simulation Complete!")