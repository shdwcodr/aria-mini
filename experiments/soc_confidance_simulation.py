import numpy as np
import pandas as pd
import joblib
import warnings
import os
import sys

warnings.filterwarnings("ignore")

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data_loader import DataLoader

print("=" * 65)
print(" Confidence-Ranked Baseline SOC Simulation")
print("=" * 65)

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

def prior_correct(p, train_prior=TRAIN_ATTACK_PRIOR,
                 test_prior=TEST_ATTACK_PRIOR, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    odds = p / (1 - p)
    ratio = (test_prior / (1 - test_prior)) / (train_prior / (1 - train_prior))
    adj = odds * ratio
    return adj / (1 + adj)

# =============================================================================
# LOAD MODEL + TEST DATA (Using New Loader)
# =============================================================================
print("Loading model and test data...")
mlp = joblib.load(MODEL_PATH)

loader = DataLoader()
loader.scaler = joblib.load("models/scaler_train200k_60pct.pkl")
loader.scaler_fitted = True
print("✅ Scaler loaded successfully")

# Use proper held-out test set
df_test = loader.load_test()

X_test, y_test, feature_names, df_test_bal = loader.preprocess(
    df_test,
    target_attack_ratio=TEST_ATTACK_PRIOR,
    target_total_samples=TEST_SIZE
)

X_test = np.nan_to_num(X_test, nan=0.0)

print(f"Test set loaded: {len(X_test):,} samples")
print(f"Attacks: {int(y_test.sum())} ({y_test.mean()*100:.1f}%)")

# =============================================================================
# PREDICTIONS
# =============================================================================
y_prob_raw = mlp.predict_proba(X_test)[:, 1]
y_prob = prior_correct(y_prob_raw)

BATCH_SIZE = TEST_SIZE // SIM_STEPS

# =============================================================================
# SIMULATION — Confidence-Ranked Top-K
# =============================================================================
results = []
queue_size = 0
fatigue = MIN_FATIGUE
attack_arrival = {}
attack_detection = {}

print("\nRunning confidence-ranked baseline simulation...")

for t in range(SIM_STEPS):
    start = t * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(X_test))
    
    batch_y = y_test[start:end]
    batch_prob = y_prob[start:end]
    
    load_factor = queue_size / SOC_TOP_K
    avg_mttr = BASE_MTTR * (1 + LOAD_MTTR_PENALTY * load_factor)
    
    # Rank by confidence, take top-K
    ranked_idx = np.argsort(batch_prob)[::-1]
    top_k_mask = np.zeros(len(batch_y), dtype=bool)
    top_k_mask[ranked_idx[:ANALYST_CAPACITY]] = True
    
    alert_count = int(top_k_mask.sum())
    raw_alert_count = int((batch_prob > TEST_ATTACK_PRIOR).sum())
    
    # Track detection
    for i in range(len(batch_y)):
        if batch_y[i] == 1:
            gidx = start + i
            attack_arrival.setdefault(gidx, t)
            if gidx not in attack_detection and top_k_mask[i]:
                attack_detection[gidx] = t + 1
    
    # Update queue
    drain = NUM_ANALYSTS * WINDOW_MINUTES / avg_mttr
    queue_size = max(0.0, queue_size - drain) + alert_count
    queue_size = min(int(queue_size), SOC_TOP_K)
    
    # Update fatigue
    utilization = queue_size / SOC_TOP_K
    stress = min(1.0, utilization * (avg_mttr / BASE_MTTR))
    fatigue_target = MIN_FATIGUE + (MAX_FATIGUE - MIN_FATIGUE) * stress
    fatigue = FATIGUE_DECAY * fatigue + (1 - FATIGUE_DECAY) * fatigue_target
    fatigue = float(np.clip(fatigue, MIN_FATIGUE, MAX_FATIGUE))
    
    results.append({
        "window": t,
        "raw_alerts": raw_alert_count,
        "top_k_alerts": alert_count,
        "queue_size": queue_size,
        "fatigue": round(fatigue, 3),
        "avg_mttr": round(avg_mttr, 2),
    })

df_conf = pd.DataFrame(results)

# =============================================================================
# SUMMARY
# =============================================================================
delays = [
    attack_detection[idx] - arr 
    for idx, arr in attack_arrival.items() 
    if idx in attack_detection
]
avg_delay = np.mean(delays) if delays else 0

print(df_conf.round(3).to_string())

print(f"""
=== CONFIDENCE-RANKED BASELINE SUMMARY ===
  Attacks in test set : {int(y_test.sum())}
  Attacks detected    : {len(attack_detection)}
  Avg detection delay : {avg_delay:.2f} windows
  Final fatigue       : {df_conf['fatigue'].iloc[-1]:.3f}
  Average MTTR        : {df_conf['avg_mttr'].mean():.2f} min
  Max queue size      : {df_conf['queue_size'].max()}
""")

# =============================================================================
# SAVE
# =============================================================================
os.makedirs("experiments/results", exist_ok=True)
df_conf.to_csv("experiments/results/confidence_ranked_baseline.csv", index=False)

print("Saved → experiments/results/confidence_ranked_baseline.csv")
print("\n✅ Confidence-ranked baseline complete!")