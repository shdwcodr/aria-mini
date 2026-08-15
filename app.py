"""
check_pool_overlap.py
=============================================================================
Verifies whether the 60,000-sample trainer-holdout pool (used for Fig. 4's
confusion matrix) and the 40,000-sample simulation/eval pool (used for
Table VIII/IX and the SOC simulation) share any rows.

Both pools are drawn independently from the same underlying loader.load_test()
dataframe (see the training script and the per-source eval script), so
overlap is possible in principle. This script checks it directly rather
than just disclosing it as "possible."

Method: rather than relying on dataframe index (which we don't know is
preserved through DataLoader.preprocess), each row's feature vector is
hashed. Two rows with identical hashes are almost certainly the same
underlying flow (collisions are astronomically unlikely for float feature
vectors of this dimensionality). This makes the check robust regardless
of how indices are handled internally.

Usage:
    python check_pool_overlap.py
"""

import os
import sys
import hashlib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from data_loader import DataLoader  # noqa: E402

TRAIN_SIZE_HOLDOUT = 60000   # matches the training script's TEST_SIZE
SIM_POOL_SIZE = 40000        # matches the per-source eval script's TEST_SIZE
ATTACK_PRIOR = 0.05          # both scripts use this same target_attack_ratio

def hash_rows(X):
    """Hash each row of a 2D array to a short hex digest for fast set comparison."""
    # round to a fixed precision first so floating-point noise from repeated
    # preprocessing runs doesn't cause identical rows to hash differently
    X_rounded = np.round(X, decimals=6)
    return [hashlib.md5(row.tobytes()).hexdigest() for row in X_rounded]

print("Loading test-source dataframe once (both pools are drawn from this)...")
loader = DataLoader()
df_test = loader.load_test()

print(f"\nSampling the {TRAIN_SIZE_HOLDOUT:,}-sample trainer-holdout pool "
      f"(same call as the training script)...")
X_holdout, y_holdout, feature_names, _ = loader.preprocess(
    df_test, target_attack_ratio=ATTACK_PRIOR, target_total_samples=TRAIN_SIZE_HOLDOUT
)
X_holdout = np.nan_to_num(X_holdout, nan=0.0)

print(f"Sampling the {SIM_POOL_SIZE:,}-sample simulation/eval pool "
      f"(same call as the per-source eval script)...")
X_sim, y_sim, _, _ = loader.preprocess(
    df_test, target_attack_ratio=ATTACK_PRIOR, target_total_samples=SIM_POOL_SIZE
)
X_sim = np.nan_to_num(X_sim, nan=0.0)

print("\nHashing rows for overlap comparison...")
holdout_hashes = set(hash_rows(X_holdout))
sim_hashes = hash_rows(X_sim)  # keep as list to count duplicates properly
sim_hash_set = set(sim_hashes)

overlap_hashes = holdout_hashes & sim_hash_set
n_overlap_unique = len(overlap_hashes)

# how many rows IN THE SIM POOL specifically are affected (a hash in holdout
# could in principle match more than one sim-pool row if sim pool itself has
# near-duplicate flows, so count actual affected sim rows, not just unique hashes)
n_overlap_rows_in_sim = sum(1 for h in sim_hashes if h in holdout_hashes)

pct_of_sim = 100 * n_overlap_rows_in_sim / len(sim_hashes)

print("\n" + "=" * 65)
print("POOL OVERLAP RESULT")
print("=" * 65)
print(f"Trainer-holdout pool size      : {len(X_holdout):,}")
print(f"Simulation/eval pool size      : {len(X_sim):,}")
print(f"Unique overlapping row-hashes  : {n_overlap_unique:,}")
print(f"Sim-pool rows affected by overlap : {n_overlap_rows_in_sim:,} "
      f"({pct_of_sim:.3f}% of sim pool)")

if n_overlap_rows_in_sim == 0:
    print("\n✅ No overlap detected — the two pools are effectively disjoint.")
else:
    print(f"\n⚠️  {n_overlap_rows_in_sim} rows in the sim pool also appear in the "
          f"trainer holdout. Consider whether this warrants re-running with "
          f"an explicit disjointness constraint.")

# save the result so it's citable / reproducible
os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/pool_overlap_check.txt", "w") as f:
    f.write(f"Trainer-holdout pool size: {len(X_holdout)}\n")
    f.write(f"Simulation/eval pool size: {len(X_sim)}\n")
    f.write(f"Unique overlapping row-hashes: {n_overlap_unique}\n")
    f.write(f"Sim-pool rows affected: {n_overlap_rows_in_sim} ({pct_of_sim:.3f}%)\n")

print("\nSaved to experiments/results/pool_overlap_check.txt")