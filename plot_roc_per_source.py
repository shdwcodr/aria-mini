

"""
plot_roc_per_source.py

Adds the per-source ROC overlay Reviewer 2 was actually asking for in
Section IV-A: a single plot with one curve per dataset source, so the
"moderate AUC but poor recall on CIC-IDS2017" argument is visible rather
than only asserted in prose.

Self-contained — reloads model/scaler/data the same way as your other
eval scripts. Run from the same location (same relative data_loader
import, same models/ and experiments/results/ paths).
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import sys

from sklearn.metrics import roc_curve, roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from data_loader import DataLoader

MODEL_PATH = "models/mlp_alert_train120k_4attack.pkl"
SCALER_PATH = "models/scaler_train120k_4attack.pkl"
TEST_SIZE = 40000
TEST_ATTACK_PRIOR = 0.05
TRAIN_ATTACK_PRIOR = 0.40
OUT_DIR = "experiments/results"
os.makedirs(OUT_DIR, exist_ok=True)

# Consistent colors/markers so this figure visually matches your other
# per-source figures (confusion matrices, ablation bar chart, etc.)
SOURCE_STYLE = {
    "CIC-IDS2017": {"color": "#1f77b4", "marker": "o"},
    "UNSW-NB15":   {"color": "#ff7f0e", "marker": "s"},
    "CIC-IoT2023": {"color": "#2ca02c", "marker": "^"},
}


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
    return float(thresholds[best]), fpr[best], tpr[best]


print("Loading model and test data...")
mlp = joblib.load(MODEL_PATH)
loader = DataLoader()
loader.scaler = joblib.load(SCALER_PATH)
loader.scaler_fitted = True

df_test = loader.load_test()
X_test, y_test, feature_names, df_test_bal = loader.preprocess(
    df_test, target_attack_ratio=TEST_ATTACK_PRIOR, target_total_samples=TEST_SIZE
)
X_test = np.nan_to_num(X_test, nan=0.0)
sources = df_test_bal["source"].to_numpy()

y_prob_raw = mlp.predict_proba(X_test)[:, 1]
y_prob_corr = prior_correct(y_prob_raw)

# The single Youden threshold selected on the COMBINED set — the same one
# used everywhere else in the paper (Table IX, Fig. 3). We overlay this
# single global operating point on every source's curve, since that is
# the actual threshold applied uniformly at deployment.
thresh_global, _, _ = youden_threshold(y_test, y_prob_corr)
print(f"Global Youden threshold (combined set): {thresh_global:.4f}")

fig, ax = plt.subplots(figsize=(7, 6.5))
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1, label="Chance")

unique_sources = sorted(set(sources))
rows = []
for src in unique_sources:
    mask = sources == src
    y_s = y_test[mask]
    p_s = y_prob_corr[mask]

    if y_s.sum() == 0 or (1 - y_s).sum() == 0:
        print(f"  {src}: single class in this source, skipping ROC")
        continue

    fpr_s, tpr_s, thr_s = roc_curve(y_s, p_s)
    auc_s = roc_auc_score(y_s, p_s)

    style = SOURCE_STYLE.get(src, {"color": None, "marker": None})
    ax.plot(fpr_s, tpr_s, lw=2, color=style["color"], label=f"{src} (AUC = {auc_s:.4f})")

    # Mark where the GLOBAL threshold lands on THIS source's curve —
    # this is the point that makes the reconciliation argument visible:
    # a source can have a curve that hugs the top-left (good ranking)
    # while the global operating point still sits at low recall for it.
    idx = np.searchsorted(thr_s[::-1], thresh_global)
    idx = len(thr_s) - 1 - idx
    idx = np.clip(idx, 0, len(thr_s) - 1)
    ax.scatter([fpr_s[idx]], [tpr_s[idx]], color=style["color"], marker=style["marker"],
               s=70, zorder=5, edgecolor="black", linewidth=0.5)

    rows.append({
        "source": src, "AUC": round(auc_s, 4),
        "FPR_at_global_threshold": round(fpr_s[idx], 4),
        "TPR_at_global_threshold_i.e._recall": round(tpr_s[idx], 4),
    })

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Per-Source ROC Curves — ARIA-MLP (Prior-Corrected)\n"
              f"Markers show recall at the single global Youden threshold ({thresh_global:.4f})")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/roc_curve_per_source_FINAL.png", dpi=300, bbox_inches="tight")

summary = pd.DataFrame(rows)
summary.to_csv(f"{OUT_DIR}/roc_per_source_summary_FINAL.csv", index=False)

print("\n" + "=" * 65)
print("PER-SOURCE ROC @ GLOBAL THRESHOLD")
print("=" * 65)
print(summary.to_string(index=False))
print(f"\nSaved: {OUT_DIR}/roc_curve_per_source_FINAL.png")
print(f"Saved: {OUT_DIR}/roc_per_source_summary_FINAL.csv")

plt.show()