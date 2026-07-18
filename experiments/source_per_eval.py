import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data_loader import DataLoader

# =============================================================================
# CONFIG
# =============================================================================
MODEL_PATH = "models/mlp_alert_train300k_5attack.pkl"
SCALER_PATH = "models/scaler_train200k_60pct.pkl"
TEST_SIZE = 40000
TEST_ATTACK_PRIOR = 0.05
TRAIN_ATTACK_PRIOR = 0.55

# =============================================================================
# PRIOR-SHIFT CORRECTION
# =============================================================================
def prior_correct(p, train_prior=TRAIN_ATTACK_PRIOR, test_prior=TEST_ATTACK_PRIOR, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    odds = p / (1 - p)
    ratio = (test_prior / (1 - test_prior)) / (train_prior / (1 - train_prior))
    adj = odds * ratio
    return adj / (1 + adj)

def youden_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    best = np.argmax(j)
    return float(thresholds[best])

# =============================================================================
# LOAD MODEL + DATA
# =============================================================================
print("Loading model and test data...")
mlp = joblib.load(MODEL_PATH)
loader = DataLoader()

loader.scaler = joblib.load(SCALER_PATH)
loader.scaler_fitted = True
print(f"✅ Scaler loaded from {SCALER_PATH}")

df_test = loader.load_test()

X_test, y_test, feature_names, df_test_bal = loader.preprocess(
    df_test,
    target_attack_ratio=TEST_ATTACK_PRIOR,
    target_total_samples=TEST_SIZE
)

X_test = np.nan_to_num(X_test, nan=0.0)
sources = df_test_bal["source"].to_numpy()

print(f"\nTest set : {len(X_test):,} samples")
print(f"Attacks : {int(y_test.sum())} ({y_test.mean()*100:.1f}%)")
print(f"Sources : {sorted(set(sources))}")

# =============================================================================
# PROBABILITIES + THRESHOLDS
# =============================================================================
y_prob_raw = mlp.predict_proba(X_test)[:, 1]
y_prob_corr = prior_correct(y_prob_raw)

thresh_youden = youden_threshold(y_test, y_prob_corr)
print(f"\nYouden threshold (corrected): {thresh_youden:.4f}")

y_pred_corr = (y_prob_corr > thresh_youden).astype(int)

# =============================================================================
# OVERALL REPORT
# =============================================================================
print("\n" + "="*65)
print("ARIA-MLP — Prior-Corrected Performance")
print("="*65)
print(classification_report(y_test, y_pred_corr,
                            target_names=["Benign", "Attack"], digits=4))

overall_auc = roc_auc_score(y_test, y_prob_corr)
print(f"AUC-ROC (corrected): {overall_auc:.4f}")

# =============================================================================
# PER-SOURCE TABLE
# =============================================================================
print("\n" + "="*65)
print("PER-SOURCE RESULTS")
print(f"{'Source':<20} {'AUC':>7} {'P':>7} {'R':>7} {'F1':>7} {'Support':>9}")
print("-"*65)

rows = []
unique_sources = sorted(set(sources))
for src in unique_sources:
    mask = sources == src
    y_s = y_test[mask]
    yp_s = y_pred_corr[mask]
    ys_s = y_prob_corr[mask]
    
    if len(y_s) == 0 or y_s.sum() == 0 or (1 - y_s).sum() == 0:
        print(f"{src:<20} (single class — skipped)")
        continue
    
    auc = roc_auc_score(y_s, ys_s)
    p = precision_score(y_s, yp_s, zero_division=0)
    r = recall_score(y_s, yp_s, zero_division=0)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    sup = int(mask.sum())
    
    print(f"{src:<20} {auc:>7.4f} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {sup:>9,d}")
    rows.append({"Source": src, "AUC-ROC": round(auc,4), "Precision": round(p,4),
                 "Recall": round(r,4), "F1": round(f1,4), "Support": sup})

# Overall row
p_all = precision_score(y_test, y_pred_corr, zero_division=0)
r_all = recall_score(y_test, y_pred_corr, zero_division=0)
f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0.0

print("-"*65)
print(f"{'Overall':<20} {overall_auc:>7.4f} {p_all:>7.4f} {r_all:>7.4f} "
      f"{f1_all:>7.4f} {len(X_test):>9,d}")

# =============================================================================
# FIGURES — Per-Source + Overall Confusion Matrices
# =============================================================================
n_src = len(unique_sources)

# Per-source confusion matrices
fig_cm, axes_cm = plt.subplots(1, n_src, figsize=(4 * n_src, 4), constrained_layout=True)
if n_src == 1:
    axes_cm = [axes_cm]

for ax, src in zip(axes_cm, unique_sources):
    mask = sources == src
    y_s = y_test[mask]
    yp_s = y_pred_corr[mask]
    
    if y_s.sum() == 0 or (1 - y_s).sum() == 0:
        ax.set_title(f"{src}\n(single class)")
        continue
    
    cm = confusion_matrix(y_s, yp_s)
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Attack"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    
    p = precision_score(y_s, yp_s, zero_division=0)
    r = recall_score(y_s, yp_s, zero_division=0)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    ax.set_title(f"{src}\nF1={f1:.2f} Recall={r:.2f}", fontsize=9)

fig_cm.suptitle(
    "ARIA-MLP — Per-Source Confusion Matrices\n"
    f"Prior-corrected | Youden threshold = {thresh_youden:.4f}\n"
    f"Test: {TEST_SIZE:,} samples (~{TEST_ATTACK_PRIOR*100:.0f}% attack)",
    fontsize=10
)

# Overall confusion matrix
fig_all, ax_all = plt.subplots(figsize=(6, 5))
cm_all = confusion_matrix(y_test, y_pred_corr)
ConfusionMatrixDisplay(cm_all, display_labels=["Benign", "Attack"]).plot(
    ax=ax_all, colorbar=True, cmap="Blues")
ax_all.set_title(
    f"ARIA-MLP Overall — Deployment Test Set\n"
    f"{TEST_SIZE:,} samples | 5% attack prevalence | AUC-ROC = {overall_auc:.4f}"
)
fig_all.tight_layout()

# =============================================================================
# SAVE
# =============================================================================
os.makedirs("experiments/results", exist_ok=True)

fig_cm.savefig("experiments/results/confusion_matrix_per_source_FINAL.png",
               dpi=300, bbox_inches="tight")
fig_all.savefig("experiments/results/confusion_matrix_overall_FINAL.png",
                dpi=300, bbox_inches="tight")

df_ps = pd.DataFrame(rows)
df_ps.to_csv("experiments/results/per_source_classification_FINAL.csv", index=False)

plt.show()

print("\nSaved to experiments/results/")
print("✅ Per-source evaluation complete.")