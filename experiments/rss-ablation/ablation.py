import numpy as np
import pandas as pd
import joblib
import shap
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
from scipy.stats import spearmanr
from lime.lime_tabular import LimeTabularExplainer
from data_loader import DataLoader

warnings.filterwarnings("ignore")

# =============================================================================
# WHAT THIS SCRIPT DOES
# Four ablation variants of RRS, each removing one component at a time.
# Story: Full RRS > Conf+SHAP > Conf only, proving each component adds value.
#
# Variant A — Conf only        : w = (1.00, 0.00, 0.00)  baseline
# Variant B — Conf + SHAP      : w = (0.55, 0.45, 0.00)  add SHAP evidence
# Variant C — Conf + Agreement : w = (0.60, 0.00, 0.40)  add LIME agreement
# Variant D — Full RRS         : w = (0.45, 0.40, 0.15)  all three components
# =============================================================================

print("\n=== ARIA RRS ABLATION STUDY ===\n")

# =============================================================================
# CONFIG — must match simulation exactly
# =============================================================================
MODEL_PATH         = "models/mlp_alert_calibrated_train200k_1attack.pkl"
TRAIN_ATTACK_PRIOR = 0.60
TEST_ATTACK_PRIOR  = 0.05
SAMPLE_SIZE        = 800     # samples to run SHAP+LIME on (expensive)
RRS_ALERT_THRESH   = 0.20    # same as BASE_THRESHOLD in simulation
K                  = 50      # top-K SOC queue size for precision@K

# =============================================================================
# PRIOR-SHIFT CORRECTION  (identical to simulation)
# =============================================================================
def prior_correct(p, train_prior=TRAIN_ATTACK_PRIOR,
                  test_prior=TEST_ATTACK_PRIOR, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    odds = p / (1 - p)
    ratio = (test_prior / (1 - test_prior)) / (train_prior / (1 - train_prior))
    adj = odds * ratio
    return adj / (1 + adj)

# =============================================================================
# RRS COMPONENTS  (identical formulas to simulation)
# =============================================================================
def shap_evidence(shap_vals):
    """mean(top-5 |SHAP|) / max(|SHAP|) — strength + concentration."""
    abs_s = np.abs(shap_vals)
    return np.mean(np.sort(abs_s)[-5:]) / (np.max(abs_s) + 1e-9)

def lime_agreement(shap_vals, lime_vals):
    """Fraction of significant features where SHAP and LIME agree on direction."""
    abs_s = np.abs(shap_vals)
    significant = (abs_s > 1e-4) | (np.abs(lime_vals) > 1e-4)
    if not np.any(significant):
        return 0.0
    return float(np.mean(
        np.sign(shap_vals[significant]) == np.sign(lime_vals[significant])
    ))

# =============================================================================
# RRS ABLATION VARIANTS
# Each weight tuple is (conf, shap_evidence, agreement) — sums to 1.0
# =============================================================================
VARIANTS = {
    "A — Conf only"        : (1.00, 0.00, 0.00),
    "B — Conf + SHAP"      : (0.55, 0.45, 0.00),
    "C — Conf + Agreement" : (0.60, 0.00, 0.40),
    "D — Full RRS"         : (0.45, 0.40, 0.15),
}

def compute_rrs(conf, shap_vals, lime_vals, w):
    ev  = shap_evidence(shap_vals)
    ag  = lime_agreement(shap_vals, lime_vals)
    rrs = w[0] * conf + w[1] * ev + w[2] * ag
    return float(np.clip(rrs, 0.0, 1.0))

# =============================================================================
# LOAD MODEL + DATA
# =============================================================================
print("Loading model and data…")
model  = joblib.load(MODEL_PATH)
loader = DataLoader()

df = loader.load_all_datasets(max_samples=150000, max_iot_parts=12)

X, y, feature_names = loader.preprocess(
    df,
    target_attack_ratio=TEST_ATTACK_PRIOR,
    target_total_samples=60000
)
X = np.nan_to_num(X, nan=0.0)

# Apply prior correction — same as simulation
probs_raw = model.predict_proba(X)[:, 1]
probs     = prior_correct(probs_raw)

print(f"Dataset: {len(X)} samples  |  Attacks: {int(y.sum())}  |  Benign: {int((y==0).sum())}")
print(f"Corrected prob range: [{probs.min():.4f}, {probs.max():.4f}]  median: {np.median(probs):.4f}\n")

# =============================================================================
# BUILD SHAP + LIME EXPLAINERS
# =============================================================================
print("Building explainers…")
background = shap.sample(X, 100, random_state=42)
shap_explainer = shap.Explainer(model.predict_proba, background)

lime_explainer = LimeTabularExplainer(
    training_data         = X[:500],
    feature_names         = feature_names,
    class_names           = ["Normal", "Attack"],
    mode                  = "classification",
    discretize_continuous = False   # keeps feature names exact
)

# =============================================================================
# SAMPLE — stratified so we get enough attacks
# =============================================================================
np.random.seed(42)
attack_idx = np.where(y == 1)[0]
benign_idx = np.where(y == 0)[0]

n_attack = min(int(SAMPLE_SIZE * TEST_ATTACK_PRIOR * 2), len(attack_idx))
n_benign = SAMPLE_SIZE - n_attack

sampled = np.concatenate([
    np.random.choice(attack_idx, n_attack, replace=False),
    np.random.choice(benign_idx, n_benign, replace=False)
])
np.random.shuffle(sampled)

print(f"Ablation sample: {len(sampled)} total  |  attacks: {n_attack}  |  benign: {n_benign}\n")

# =============================================================================
# COMPUTE SHAP + LIME FOR EACH SAMPLED INSTANCE
# (done once, reused across all variants — avoids 4x compute cost)
# =============================================================================
print(f"Computing SHAP + LIME for {len(sampled)} instances…")
shap_cache = {}
lime_cache = {}

for cnt, i in enumerate(sampled):
    if (cnt + 1) % 100 == 0 or cnt == 0:
        print(f"  {cnt+1}/{len(sampled)}…")

    # SHAP
    sv = shap_explainer(X[i:i+1]).values[0][:, 1]
    shap_cache[i] = sv

    # LIME — attack class coefficients explicitly
    exp = lime_explainer.explain_instance(
        X[i], model.predict_proba,
        num_features=10, num_samples=300,
        labels=(1,)
    )
    lime_vec = np.zeros(len(feature_names))
    for f, v in exp.as_list(label=1):
        f_clean = f.strip()
        if f_clean in feature_names:
            lime_vec[feature_names.index(f_clean)] = v
        else:
            for j, fname in enumerate(feature_names):
                if fname.strip().lower() == f_clean.lower():
                    lime_vec[j] = v
                    break
    lime_cache[i] = lime_vec

print("✅ Explanations ready\n")

# =============================================================================
# EVALUATE EACH VARIANT
# =============================================================================
rows   = []
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print("=" * 60)
print("  ABLATION RESULTS")
print("=" * 60)

for name, w in VARIANTS.items():

    scores = np.array([
        compute_rrs(probs[i], shap_cache[i], lime_cache[i], w)
        for i in sampled
    ])

    y_samp   = y[sampled]
    preds_rrs = (scores > RRS_ALERT_THRESH).astype(int)

    # Core classification metrics
    f1   = f1_score(y_samp, preds_rrs, zero_division=0)
    prec = precision_score(y_samp, preds_rrs, zero_division=0)
    rec  = recall_score(y_samp, preds_rrs, zero_division=0)

    alert_rate = np.mean(preds_rrs)

    # Score separation between attack and benign
    atk_scores = scores[y_samp == 1]
    ben_scores = scores[y_samp == 0]
    separation = float(np.mean(atk_scores) - np.mean(ben_scores))

    # Score stability
    stability = float(np.std(scores))

    # Rank correlation with model confidence
    corr = float(spearmanr(probs[sampled], scores).correlation)

    # Top-K SOC precision — what fraction of top-K alerts are real attacks
    top_k_idx       = np.argsort(scores)[-K:]
    top_k_labels    = y_samp[top_k_idx]
    top_k_precision = float(np.mean(top_k_labels == 1))
    recall_at_k     = float(np.sum(top_k_labels == 1) / max(np.sum(y_samp == 1), 1))

    print(f"\n{name}   weights={w}")
    print(f"  F1             : {f1:.4f}")
    print(f"  Precision      : {prec:.4f}")
    print(f"  Recall         : {rec:.4f}")
    print(f"  Alert rate     : {alert_rate:.4f}")
    print(f"  Separation     : {separation:.4f}   (attack mean − benign mean)")
    print(f"  Stability(std) : {stability:.4f}")
    print(f"  Corr(conf,RRS) : {corr:.4f}")
    print(f"  Top-{K} Prec   : {top_k_precision:.4f}")
    print(f"  Recall@{K}     : {recall_at_k:.4f}")
    print("-" * 60)

    rows.append({
        "run_id"          : run_id,
        "variant"         : name,
        "w_conf"          : w[0],
        "w_shap"          : w[1],
        "w_agreement"     : w[2],
        "f1"              : round(f1,   4),
        "precision"       : round(prec, 4),
        "recall"          : round(rec,  4),
        "alert_rate"      : round(alert_rate, 4),
        "separation"      : round(separation, 4),
        "stability"       : round(stability,  4),
        "corr_conf_rrs"   : round(corr, 4),
        "top_k_precision" : round(top_k_precision, 4),
        "recall_at_k"     : round(recall_at_k, 4),
    })

# =============================================================================
# DELTA TABLE  (Full RRS minus each ablated variant)
# =============================================================================
full = next(r for r in rows if r["variant"].startswith("D"))
metrics = ["f1", "precision", "recall", "separation", "top_k_precision", "recall_at_k"]

print("\n=== DELTA TABLE  (Full RRS − ablated variant) ===")
print(f"{'Variant':<30} " + "  ".join(f"{m:>12}" for m in metrics))
print("-" * 110)
for r in rows:
    if r["variant"].startswith("D"):
        continue
    deltas = [round(full[m] - r[m], 4) for m in metrics]
    print(f"{r['variant']:<30} " + "  ".join(f"{d:>+12.4f}" for d in deltas))

# =============================================================================
# SAVE
# =============================================================================
os.makedirs("experiments/rss_ablation/results", exist_ok=True)
csv_path = "experiments/rss_ablation/results/rss_ablation_results.csv"

df_out = pd.DataFrame(rows)
if os.path.exists(csv_path):
    df_out.to_csv(csv_path, mode="a", header=False, index=False)
else:
    df_out.to_csv(csv_path, index=False)

print(f"\nSaved → {csv_path}")

# =============================================================================
# FIGURE 1 — PER-SOURCE CONFUSION MATRICES
# Runs model on full dataset, splits results by source tag
# =============================================================================
print("\nGenerating per-source confusion matrices…")

# Get raw predictions on full dataset (no prior correction needed for CM)
y_pred_full = model.predict(X)
sources     = df["source"].values[-len(X):]   # source tag per sample

unique_sources = sorted(set(sources))
n_src = len(unique_sources)

fig_cm, axes_cm = plt.subplots(
    1, n_src, figsize=(4 * n_src, 4), constrained_layout=True
)
if n_src == 1:
    axes_cm = [axes_cm]

for ax, src in zip(axes_cm, unique_sources):
    mask   = sources == src
    y_s    = y[mask]
    yp_s   = y_pred_full[mask]
    cm     = confusion_matrix(y_s, yp_s)
    disp   = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Attack"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    f1_s   = f1_score(y_s, yp_s, zero_division=0)
    rec_s  = recall_score(y_s, yp_s, zero_division=0)
    ax.set_title(f"{src}\nF1={f1_s:.2f}  Recall={rec_s:.2f}", fontsize=9)

fig_cm.suptitle("ARIA Model — Confusion Matrix per Dataset Source", fontsize=11)
os.makedirs("experiments/rss_ablation/results", exist_ok=True)
cm_path = "experiments/rss_ablation/results/confusion_matrices_per_source.png"
fig_cm.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Confusion matrices → {cm_path}")

# Print per-source classification report to console too
print("\n=== PER-SOURCE CLASSIFICATION REPORT ===")
for src in unique_sources:
    mask = sources == src
    print(f"\n{src}  (n={mask.sum()})")
    print(classification_report(y[mask], y_pred_full[mask],
                                target_names=["Benign", "Attack"],
                                zero_division=0))

# =============================================================================
# FIGURE 2 — RRS ABLATION BAR CHART
# Shows F1, Separation, Top-K Precision across the four variants
# =============================================================================
print("\nGenerating RRS ablation bar chart…")

variant_labels  = [r["variant"].split("—")[1].strip() for r in rows]
metrics_plot    = {
    "F1 Score"         : [r["f1"]              for r in rows],
    "Separation"       : [r["separation"]      for r in rows],
    "Top-K Precision"  : [r["top_k_precision"] for r in rows],
    "Recall@K"         : [r["recall_at_k"]     for r in rows],
}

x      = np.arange(len(variant_labels))
n_met  = len(metrics_plot)
width  = 0.18
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig_abl, ax_abl = plt.subplots(figsize=(12, 5))

for idx, (met_name, vals) in enumerate(metrics_plot.items()):
    offset = (idx - n_met / 2 + 0.5) * width
    bars   = ax_abl.bar(x + offset, vals, width,
                        label=met_name, color=colors[idx], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax_abl.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

ax_abl.set_xticks(x)
ax_abl.set_xticklabels(variant_labels, fontsize=9)
ax_abl.set_ylabel("Score")
ax_abl.set_ylim(0, 1.05)
ax_abl.set_title("RRS Ablation Study — Component Contribution Analysis")
ax_abl.legend(loc="upper left", fontsize=8)
ax_abl.grid(axis="y", alpha=0.3)

fig_abl.tight_layout()
abl_path = "experiments/rss_ablation/results/rss_ablation_barchart.png"
fig_abl.savefig(abl_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Ablation bar chart → {abl_path}")

print("\n✅ RRS ablation complete!")