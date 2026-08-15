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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from data_loader import DataLoader  # noqa: E402

RANDOM_STATE = 42
TRAIN_PRIOR = 0.40
TEST_PRIOR = 0.05
N_BINS = 10  # for ECE / reliability diagram

print("=" * 70)
print(" ROC / PR / CALIBRATION PROBE")
print("=" * 70)

# =============================================================================
# ADAPT THIS SECTION — same pipeline as the other two probes
# =============================================================================
def load_trained_pipeline():
    import joblib
    candidates_model = ["models/mlp_model.pkl", "model/mlp_model.pkl",
                         "artifacts/mlp_model.pkl", "mlp_model.pkl"]
    candidates_scaler = ["models/scaler.pkl", "model/scaler.pkl",
                          "artifacts/scaler.pkl", "scaler.pkl"]
    model, scaler = None, None
    for p in candidates_model:
        fp = os.path.join(PROJECT_ROOT, p)
        if os.path.exists(fp):
            model = joblib.load(fp)
            break
    for p in candidates_scaler:
        fp = os.path.join(PROJECT_ROOT, p)
        if os.path.exists(fp):
            scaler = joblib.load(fp)
            break
    if model is None or scaler is None:
        raise FileNotFoundError(
            "Edit load_trained_pipeline() to point at your saved model/scaler."
        )
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

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(fpr, tpr, label=f"ARIA (AUC = {auc:.4f})")
axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve (40k deployment test set)")
axes[0].legend()

axes[1].plot(recall, precision, label=f"ARIA (AP = {ap:.4f})")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve (40k deployment test set)")
axes[1].legend()

plt.tight_layout()
OUT_DIR = "experiments/results"
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUT_DIR, "roc_pr_curves.png"), dpi=200)
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
plt.savefig(os.path.join(OUT_DIR, "reliability_diagram.png"), dpi=200)
plt.close()

# =============================================================================
# SAVE NUMERIC SUMMARY
# =============================================================================
summary = pd.DataFrame([{
    "auc_roc": auc,
    "average_precision": ap,
    "ece_before": ece_before,
    "ece_after": ece_after,
    "brier_before": brier_before,
    "brier_after": brier_after,
}])
summary.to_csv(os.path.join(OUT_DIR, "calibration_roc_summary.csv"), index=False)

print(f"\n✅ Saved figure: {OUT_DIR}/roc_pr_curves.png")
print(f"✅ Saved figure: {OUT_DIR}/reliability_diagram.png")
print(f"✅ Saved to: {OUT_DIR}/calibration_roc_summary.csv")
print("\nPaste the printed numbers back (figures I can't see, but the ECE/Brier")
print("numbers are what go in the text) and I'll write the Section IV subsection + figure captions.")