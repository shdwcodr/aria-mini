"""
dataset_id_probe.py
=============================================================================
Reviewer check: does the FeatureHarmonizer's 20-feature canonical
representation still let a classifier tell CIC-IDS2017 / UNSW-NB15 /
CIC-IoT2023 apart? If yes, ARIA's model might partly be learning dataset
provenance rather than genuine attack signal (since CIC-IoT2023 is
~100% attack in the raw pool -- see Table IV -- source could act as a
proxy label).

Method:
  1. Load the TRAIN split (same DataLoader/FeatureHarmonizer pipeline
     used everywhere else in the paper), so this uses the exact same
     20 canonical features the MLP is trained on.
  2. Use "source" (CIC-IDS2017 / UNSW-NB15 / CIC-IoT2023) as the label
     instead of attack/benign.
  3. Balance classes so a chance baseline is a clean 1/3, then train a
     simple classifier (LogisticRegression, no hyperparameter search --
     this experiment is asking "is source trivially recoverable?", not
     "what's the best possible source-classifier?").
  4. Report accuracy + per-class metrics + confusion matrix.

Interpretation:
  - Accuracy near chance (~33%) => harmonisation is doing its job value;
    the 20 features don't strongly encode which raw dataset a row came from.
  - Accuracy near 100% => the model has an easy shortcut available; a
    reviewer would be right to worry ARIA's attack detector could be
    partly relying on dataset identity rather than universal attack
    signal.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from data_loader import DataLoader  # noqa: E402

RANDOM_STATE = 42
SAMPLES_PER_SOURCE = 20000  # balanced per-class sample size (capped by smallest source)

print("=" * 70)
print(" DATASET-ID PROBE — does the harmonized feature space leak source?")
print("=" * 70)

# =============================================================================
# LOAD RAW (unharmonized-label) TRAIN DATA — same loader as everywhere else
# =============================================================================
loader = DataLoader()
df_train = loader.load_train()

print(f"\nRaw source counts:\n{df_train['source'].value_counts()}")

# =============================================================================
# HARMONIZE FEATURES (identical 20-feature schema used by the MLP)
# =============================================================================
print("\nHarmonising features via FeatureHarmonizer...")
df_train = df_train.reset_index(drop=True)
X_all = loader.harmonizer.transform(df_train)
X_all = X_all.reset_index(drop=True)
X_all.replace([np.inf, -np.inf], 0, inplace=True)
X_all.fillna(0, inplace=True)

sources_all = df_train["source"].to_numpy()

# =============================================================================
# BALANCE CLASSES (equal samples per source => clean 1/3 chance baseline)
# =============================================================================
rng = np.random.RandomState(RANDOM_STATE)
idx_by_source = {src: np.where(sources_all == src)[0] for src in np.unique(sources_all)}
min_available = min(len(idx) for idx in idx_by_source.values())
n_per_source = min(SAMPLES_PER_SOURCE, min_available)

print(f"\nBalancing to {n_per_source:,} samples per source "
      f"(smallest source has {min_available:,} available)")

selected_idx = np.concatenate([
    rng.choice(idx, size=n_per_source, replace=False)
    for idx in idx_by_source.values()
])
rng.shuffle(selected_idx)

X = X_all.iloc[selected_idx].to_numpy()
y = sources_all[selected_idx]

print(f"Balanced probe set: {len(X):,} rows, classes = {sorted(set(y))}")

# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# =============================================================================
# CLASSIFIER — deliberately simple; this is a "can source be recovered
# at all", not a "what's the best possible source classifier" question
# =============================================================================
clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, multi_class="multinomial")
clf.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)

acc = accuracy_score(y_test, y_pred)
n_classes = len(set(y))
chance = 1.0 / n_classes

print("\n" + "=" * 70)
print(" RESULTS")
print("=" * 70)
print(f"Chance-level accuracy ({n_classes} balanced classes): {chance:.4f}")
print(f"Dataset-ID classifier accuracy:                       {acc:.4f}")
print(f"Margin over chance:                                   {acc - chance:+.4f}")

print("\nPer-class report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion matrix (rows=true, cols=predicted):")
labels_sorted = sorted(set(y))
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
print(cm_df)

# =============================================================================
# SAVE
# =============================================================================
OUT_DIR = "experiments/results"
os.makedirs(OUT_DIR, exist_ok=True)

summary = pd.DataFrame([{
    "n_classes": n_classes,
    "chance_accuracy": chance,
    "dataset_id_accuracy": acc,
    "margin_over_chance": acc - chance,
    "n_per_source": n_per_source,
}])
summary.to_csv(os.path.join(OUT_DIR, "dataset_id_probe_summary.csv"), index=False)
cm_df.to_csv(os.path.join(OUT_DIR, "dataset_id_probe_confusion_matrix.csv"))

print(f"\n✅ Saved to: {OUT_DIR}/dataset_id_probe_summary.csv")
print(f"✅ Saved to: {OUT_DIR}/dataset_id_probe_confusion_matrix.csv")

if acc > 0.9:
    print("\n⚠️  Source is trivially recoverable from the harmonized features.")
    print("    This is a real limitation worth disclosing/investigating further.")
elif acc > 0.5:
    print("\n⚠️  Source is recoverable well above chance. Worth discussing as a")
    print("    partial limitation, even if not as strong as full leakage.")
else:
    print("\n✅ Accuracy is close to chance — the harmonized features do not")
    print("   strongly encode dataset provenance. Good news for the paper.")