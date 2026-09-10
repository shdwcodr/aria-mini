"""
calibration_roc_probe.py
=============================================================================
Reviewer check: the paper leans on a single AUC number to reconcile the
"moderate AUC, low recall" pattern (Section IV-A) but never shows the ROC
curve. Separately, the entire prior-shift contribution (Section III-E /
III-H) is a calibration claim with no calibration evidence (no reliability
diagram, no ECE/Brier score, before vs. after correction).

This generates both:
  1. ROC curve + PR curve (overall, on the 40k deployment test set)
  2. Reliability diagram + ECE + Brier score, BEFORE and AFTER
     prior-shift correction, so the calibration claim in Section III-E
     has an actual figure behind it.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, brier_score_loss
)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
from data_loader import DataLoader  # noqa: E402

RANDOM_STATE = 42
TRAIN_PRIOR = 0.40
TEST_PRIOR = 0.05
N_BINS = 10  # for ECE / reliability diagram

# The final deployed model selected in Table 3 (120k samples, 40% attack
# prevalence). This is the same model loaded by source_per_eval.py and
# ablation.py — keep this in sync with those if the deployed model ever
# changes.
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mlp_alert_train120k_4attack.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler_train120k_4attack.pkl")

print("=" * 70)
print(" ROC / PR / CALIBRATION PROBE")
print("=" * 70)


def load_trained_pipeline():
    import joblib
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Update MODEL_PATH/SCALER_PATH "
            "above if the deployed model filename has changed."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Update MODEL_PATH/SCALER_PATH "
            "above if the deployed model filename has changed."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def raw_probabilities(X, model, scaler):
    X_s = scaler.transform(X)
    return model.predict_proba(X_s)[:, 1]


def apply_prior_shift(p, train_prior=TRAIN_PRIOR, test_prior=TEST_PRIOR):
    r = (test_prior / (1 - test_prior)) / (train_prior / (1 - train_prior))
    return (p * r) / (1 - p + p * r)


def expected_calibration_error(y_true, y_prob, n_bins=N_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    ece = 0.0
    bin_stats = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            bin_stats.append((bins[b], bins[b + 1], np.nan, np.nan, 0))
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        weight = mask.sum() / len(y_true)
        ece += weight * abs(acc - conf)
        bin_stats.append((bins[b], bins[b + 1], conf, acc, int(mask.sum())))
    return ece, bin_stats


# =============================================================================
# LOAD 40k DEPLOYMENT TEST SET
# =============================================================================
loader = DataLoader()
df_test = loader.load_deployment_test(prevalence=TEST_PRIOR, n=40000)  # ADAPT if needed
df_test = df_test.reset_index(drop=True)

X_test = loader.harmonizer.transform(df_test)
X_test.replace([np.inf, -np.inf], 0, inplace=True)
X_test.fillna(0, inplace=True)

y_test = df_test["label"].to_numpy()  # ADAPT column name

model, scaler = load_trained_pipeline()
p_raw = raw_probabilities(X_test.to_numpy(), model, scaler)
p_corrected = apply_prior_shift(p_raw)

# =============================================================================
# ROC + PR CURVES (on prior-corrected probabilities, matching Table V/VII)
# =============================================================================
fpr, tpr, _ = roc_curve(y_test, p_corrected)
auc = roc_auc_score(y_test, p_corrected)

precision, recall, _ = precision_recall_curve(y_test, p_corrected)
ap = average_precision_score(y_test, p_corrected)

OUT_DIR = "experiments/results"
os.makedirs(OUT_DIR, exist_ok=True)

# Saved as two separate figures (not a combined 1x2 subplot) because that is
# the artifact format shipped in "final figures and tables/" and referenced
# by name in REPRODUCIBILITY.md — roc_curve_FINAL.png and pr_curve_FINAL.png.
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(fpr, tpr, label=f"ARIA (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (40k deployment test set)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curve_FINAL.png"), dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(recall, precision, label=f"ARIA (AP = {ap:.4f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve (40k deployment test set)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pr_curve_FINAL.png"), dpi=200)
plt.close()

print(f"\nAUC-ROC: {auc:.4f}")
print(f"Average precision (PR-AUC): {ap:.4f}")

# =============================================================================
# CALIBRATION: BEFORE vs AFTER PRIOR-SHIFT CORRECTION
# =============================================================================
ece_before, bins_before = expected_calibration_error(y_test, p_raw)
ece_after, bins_after = expected_calibration_error(y_test, p_corrected)

brier_before = brier_score_loss(y_test, p_raw)
brier_after = brier_score_loss(y_test, p_corrected)

print("\n" + "=" * 70)
print(" CALIBRATION RESULTS")
print("=" * 70)
print(f"ECE before prior-shift correction: {ece_before:.4f}")
print(f"ECE after prior-shift correction:  {ece_after:.4f}")
print(f"Brier score before: {brier_before:.4f}")
print(f"Brier score after:  {brier_after:.4f}")

fig, ax = plt.subplots(figsize=(5.5, 5.5))
for bins_data, label, marker in [(bins_before, "Before correction", "o"),
                                   (bins_after, "After correction", "s")]:
    confs = [b[2] for b in bins_data if not np.isnan(b[2])]
    accs = [b[3] for b in bins_data if not np.isnan(b[3])]
    ax.plot(confs, accs, marker=marker, label=label)
ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
ax.set_xlabel("Mean predicted probability (bin)")
ax.set_ylabel("Observed attack frequency (bin)")
ax.set_title(f"Reliability Diagram\nECE before={ece_before:.3f}, after={ece_after:.3f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "calibration_reliability_FINAL.png"), dpi=200)
plt.close()

# =============================================================================
# SAVE NUMERIC SUMMARY
# =============================================================================
# Written directly in the long-format {variant, ECE, Brier} shape used by
# calibration_summary_FINAL.csv (Table 11), rather than a wide before/after
# row, so this script's output matches the shipped final file with no manual
# reshaping step.
summary = pd.DataFrame([
    {"variant": "raw", "ECE": ece_before, "Brier": brier_before},
    {"variant": "prior_corrected", "ECE": ece_after, "Brier": brier_after},
])
summary.to_csv(os.path.join(OUT_DIR, "calibration_summary_FINAL.csv"), index=False)

# ROC-AUC / Average Precision are printed (they aren't part of Table 11) and
# also stashed alongside the curve PNGs for anyone regenerating Fig. 4/6.
with open(os.path.join(OUT_DIR, "calibration_auc_ap_FINAL.txt"), "w") as f:
    f.write(f"auc_roc={auc:.4f}\naverage_precision={ap:.4f}\n")

print(f"\n✅ Saved figure: {OUT_DIR}/roc_curve_FINAL.png")
print(f"✅ Saved figure: {OUT_DIR}/pr_curve_FINAL.png")
print(f"✅ Saved figure: {OUT_DIR}/calibration_reliability_FINAL.png")
print(f"✅ Saved to: {OUT_DIR}/calibration_summary_FINAL.csv")
print(f"✅ Saved to: {OUT_DIR}/calibration_auc_ap_FINAL.txt")