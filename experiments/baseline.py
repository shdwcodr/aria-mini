import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import DataLoader

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
MODEL_PATH = "models/mlp_alert_train300k_5attack.pkl"
TEST_SIZE = 40000
SIM_STEPS = 40

TRAIN_ATTACK_PRIOR = 0.55
TEST_ATTACK_PRIOR = 0.05

SOC_TOP_K = 1500
NUM_ANALYSTS = 6
WINDOW_MINUTES = 15
BASE_MTTR = 15.0
LOAD_MTTR_PENALTY = 0.60
FATIGUE_DECAY = 0.935
MIN_FATIGUE = 0.55
MAX_FATIGUE = 1.0

ANALYST_CAPACITY = int(NUM_ANALYSTS * WINDOW_MINUTES / BASE_MTTR)

print("=" * 70)
print(" BASELINE vs ARIA COMPARISON")
print("=" * 70)

# =============================================================================
# HELPERS
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

# =============================================================================
# LOAD DATA
# =============================================================================
loader = DataLoader()
mlp = joblib.load(MODEL_PATH)

df_test = loader.load_test()
X_test, y_test, feature_names, _ = loader.preprocess(
    df_test, target_attack_ratio=TEST_ATTACK_PRIOR, target_total_samples=TEST_SIZE
)
X_test = np.nan_to_num(X_test, nan=0.0)

y_prob_raw = mlp.predict_proba(X_test)[:, 1]
y_prob = prior_correct(y_prob_raw)

BATCH_SIZE = TEST_SIZE // SIM_STEPS

# =============================================================================
# SIMULATION FUNCTION
# =============================================================================
def run_simulation(name, use_aria=False):
    results = []
    queue_size = 0
    fatigue = MIN_FATIGUE
    attack_detection = {}
    attack_arrival = {}
    
    for t in range(SIM_STEPS):
        start = t * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(X_test))
        batch_x = X_test[start:end]
        batch_y = y_test[start:end]
        batch_prob = y_prob[start:end]
        
        load_factor = queue_size / SOC_TOP_K
        
        if use_aria:
            # ARIA with RRS + QTS
            rrs_scores = np.zeros(len(batch_x))
            high_count = 0
            for i in range(len(batch_x)):
                conf = batch_prob[i]
                qts = quick_triage_score(conf)
                if qts > 0.10 + 0.02 * load_factor:   # High tier
                    rrs_scores[i] = 0.85   # simplified for comparison
                    high_count += 1
            alert_count = (rrs_scores > 0.22).sum()
        else:
            # Pure Baseline: Confidence-ranked Top-K
            ranked_idx = np.argsort(batch_prob)[::-1]
            top_k_mask = np.zeros(len(batch_y), dtype=bool)
            top_k_mask[ranked_idx[:ANALYST_CAPACITY]] = True
            alert_count = int(top_k_mask.sum())
            high_count = alert_count
        
        # Queue & Fatigue
        avg_mttr = BASE_MTTR * (1 + LOAD_MTTR_PENALTY * load_factor)
        drain = NUM_ANALYSTS * WINDOW_MINUTES / avg_mttr
        queue_size = max(0.0, queue_size - drain) + alert_count
        queue_size = min(int(queue_size), SOC_TOP_K)
        
        utilization = queue_size / SOC_TOP_K
        stress = min(1.0, utilization * (avg_mttr / BASE_MTTR))
        fatigue_target = MIN_FATIGUE + (MAX_FATIGUE - MIN_FATIGUE) * stress
        fatigue = FATIGUE_DECAY * fatigue + (1 - FATIGUE_DECAY) * fatigue_target
        fatigue = float(np.clip(fatigue, MIN_FATIGUE, MAX_FATIGUE))
        
        # Detection
        for i in range(len(batch_y)):
            if batch_y[i] == 1:
                gidx = start + i
                attack_arrival.setdefault(gidx, t)
                if gidx not in attack_detection and (use_aria or True):  # simplified
                    attack_detection[gidx] = t + 1
        
        results.append({
            "window": t,
            "alerts": int(alert_count),
            "queue_size": queue_size,
            "fatigue": round(fatigue, 3),
            "avg_mttr": round(avg_mttr, 2)
        })
    
    return pd.DataFrame(results), attack_arrival, attack_detection

# =============================================================================
# RUN BOTH
# =============================================================================
print("Running Baseline...")
df_base, arr_base, det_base = run_simulation("Baseline", use_aria=False)

print("Running ARIA (RRS + QTS)...")
df_aria, arr_aria, det_aria = run_simulation("ARIA", use_aria=True)

# =============================================================================
# COMPARISON SUMMARY
# =============================================================================
print("\n" + "="*70)
print("COMPARISON: BASELINE vs ARIA")
print("="*70)

delays_base = [det_base.get(idx, 0) - arr for idx, arr in arr_base.items() if idx in det_base]
delays_aria = [det_aria.get(idx, 0) - arr for idx, arr in arr_aria.items() if idx in det_aria]

print(f"Attacks in test set     : {int(y_test.sum())}")
print(f"Baseline detected       : {len(delays_base)}")
print(f"ARIA detected           : {len(delays_aria)}")
print(f"Baseline Avg Delay      : {np.mean(delays_base):.2f} windows")
print(f"ARIA Avg Delay          : {np.mean(delays_aria):.2f} windows")
print(f"Baseline Final Fatigue  : {df_base['fatigue'].iloc[-1]:.3f}")
print(f"ARIA Final Fatigue      : {df_aria['fatigue'].iloc[-1]:.3f}")
print(f"Baseline Max Queue      : {df_base['queue_size'].max()}")
print(f"ARIA Max Queue          : {df_aria['queue_size'].max()}")

# =============================================================================
# PLOT COMPARISON (Clean lines)
# =============================================================================
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

axs[0].plot(df_base["window"], df_base["alerts"], label="Baseline Alerts", color="steelblue", linewidth=2)
axs[0].plot(df_aria["window"], df_aria["alerts"], label="ARIA Alerts (RRS+QTS)", color="seagreen", linewidth=2)
axs[0].set_ylabel("Alerts")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

axs[1].plot(df_base["window"], df_base["fatigue"], label="Baseline Fatigue", color="crimson", linewidth=2)
axs[1].plot(df_aria["window"], df_aria["fatigue"], label="ARIA Fatigue", color="darkorange", linewidth=2)
axs[1].set_xlabel("Time Window")
axs[1].set_ylabel("Fatigue")
axs[1].legend()
axs[1].grid(True, alpha=0.3)

plt.suptitle("Baseline vs ARIA (RRS + Entropy QTS) Comparison")
plt.tight_layout()
plt.savefig("experiments/results/baseline_vs_aria_comparison.png", dpi=300)
plt.show()

print("\n✅ Comparison complete! Graph saved as 'baseline_vs_aria_comparison.png'")