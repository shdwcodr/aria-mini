import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
import os
import sys
# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data_loader import DataLoader
MODEL_PATH = "models/mlp_alert_train120k_4attack.pkl"
SCALER_PATH = "models/scaler_train120k_4attack.pkl"
TEST_SIZE = 40000
TEST_ATTACK_PRIOR = 0.05
TRAIN_ATTACK_PRIOR = 0.50
# Assuming constants from ARIA
SIM_STEPS = 40
BATCH_SIZE = 1000  # TEST_SIZE // SIM_STEPS, assuming TEST_SIZE=40000
SOC_TOP_K = 1500
NUM_ANALYSTS = 6
WINDOW_MINUTES = 15
BASE_MTTR = 15.0
LOAD_MTTR_PENALTY = 0.60
FATIGUE_DECAY = 0.935
MIN_FATIGUE = 0.55
MAX_FATIGUE = 1.0
ANALYST_CAPACITY = 7   # Match ARIA average alert volume
TEST_ATTACK_PRIOR = 0.05
ALERT_THRESHOLD = 0.5

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
y_prob = mlp.predict_proba(X_test)[:, 1]

print(f"Test set loaded: {len(X_test):,} samples")
print(f"Attacks: {int(y_test.sum())} ({y_test.mean()*100:.1f}%)")

results = []
queue_size = 0
fatigue = MIN_FATIGUE
attack_arrival = {}
attack_detection = {}

print("Running confidence-ranked baseline simulation...")

for t in range(SIM_STEPS):
    start = t * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(y_test))
    
    batch_y = y_test[start:end]
    batch_prob = y_prob[start:end]
    
    load_factor = queue_size / SOC_TOP_K
    avg_mttr = BASE_MTTR * (1 + LOAD_MTTR_PENALTY * load_factor)
    
    # Rank by confidence, take top-K
    ranked_idx = np.argsort(batch_prob)[::-1]
    top_k_mask = np.zeros(len(batch_y), dtype=bool)
    top_k_mask[ranked_idx[:ANALYST_CAPACITY]] = True  # Use ANALYST_CAPACITY as proxy
    
    alert_count = int(top_k_mask.sum())
    raw_alert_count = int((batch_prob > ALERT_THRESHOLD).sum())
    
    # Track detection (simplified)
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
print(df_conf)

# Detection statistics
detected = len(attack_detection)

delays = [
    attack_detection[idx] - attack_arrival[idx]
    for idx in attack_detection
]

avg_delay = np.mean(delays) if delays else np.nan
avg_delay_minutes = avg_delay * WINDOW_MINUTES

print("\n=== BASELINE SUMMARY ===")

print("Attacks detected :", detected)
print("Final fatigue    :", fatigue)
print("Avg MTTR         :", df_conf["avg_mttr"].mean())
print("Max queue        :", df_conf["queue_size"].max())
print("Detection delay :", avg_delay_minutes, "minutes")

# -----------------------------
# Save baseline simulation results
# -----------------------------
# -----------------------------
# Save baseline simulation results
# -----------------------------
os.makedirs("experiments/results", exist_ok=True)

csv_path = "experiments/results/baseline_simulation.csv"
df_conf.to_csv(csv_path, index=False)

print(f"\n✅ Baseline simulation saved to: {csv_path}")

# --- add this ---
detected_ids = sorted(attack_detection.keys())
pd.Series(detected_ids, name="global_idx").to_csv(
    "experiments/results/baseline_detected_attack_ids.csv", index=False
)
print(f"✅ Baseline detected-attack IDs saved: {len(detected_ids)} ids")