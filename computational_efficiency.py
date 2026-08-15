"""
ARIA Table 6 — SHAP/LIME Drill-Down Example

Generates a publication-ready LaTeX table showing the feature-level
explanation for a single high-confidence attack escalation, illustrating
what an analyst sees in the drill-down report.

Also generates a formatted CSV version for easy copy-paste.
"""

import numpy as np
import pandas as pd
import joblib
import shap
import os
import sys
import warnings
from lime.lime_tabular import LimeTabularExplainer
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from data_loader import DataLoader

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONFIG — must match simulation
# =============================================================================
MODEL_PATH = "models/mlp_alert_train120k_4attack.pkl"
SCALER_PATH = "models/scaler_train120k_4attack.pkl"
TEST_SIZE   = 40000
TRAIN_PRIOR = 0.55
TEST_ATTACK_PRIOR  = 0.05
N_FEATURES  = 6      # top features to show in the table
N_SAMPLES   = 300    # LIME perturbation samples

CANONICAL_FEATURES = [
    "duration","protocol","fwd_packets","bwd_packets","total_packets",
    "fwd_bytes","bwd_bytes","total_bytes","packet_rate","byte_rate",
    "mean_packet_size","std_packet_size","min_packet_size","max_packet_size",
    "iat_mean","iat_std","syn_count","ack_count","fin_count","rst_count",
]

def prior_correct(p, train=TRAIN_PRIOR, test=TEST_ATTACK_PRIOR, eps=1e-9):
    p = np.clip(p, eps, 1-eps)
    r = (test/(1-test)) / (train/(1-train))
    return (p*r) / (1-p+p*r)

# =============================================================================
# LOAD
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

y_prob_raw  = mlp.predict_proba(X_test)[:, 1]
y_prob_corr = prior_correct(y_prob_raw)

print(f"Test set: {len(X_test):,} | Attacks: {int(y_test.sum())}")

# =============================================================================
# FIND A HIGH-CONFIDENCE TRUE POSITIVE for the table
# We want: corrected prob > 0.80, true label = 1
# =============================================================================
attack_idx = np.where(
    (y_test == 1) & (y_prob_corr > 0.80)
)[0]

if len(attack_idx) == 0:
    # Fallback to highest-confidence attack
    attack_idx = np.where(y_test == 1)[0]

# Pick the sample with highest corrected confidence
best_i = attack_idx[np.argmax(y_prob_corr[attack_idx])]
instance = X_test[best_i]
raw_conf  = float(y_prob_raw[best_i])
corr_conf = float(y_prob_corr[best_i])

print(f"\nSelected instance {best_i}:")
print(f"  Raw confidence   : {raw_conf:.4f}")
print(f"  Corrected conf   : {corr_conf:.4f}")
print(f"  True label       : {'ATTACK' if y_test[best_i]==1 else 'BENIGN'}")

# =============================================================================
# SHAP
# =============================================================================
print("\nComputing SHAP values...")
background = shap.sample(X_test, 100, random_state=42)
explainer  = shap.Explainer(mlp.predict_proba, background)
shap_vals  = explainer(instance.reshape(1,-1)).values[0][:, 1]
print("  SHAP done.")

# =============================================================================
# LIME
# =============================================================================
print("Computing LIME values...")
lime_exp = LimeTabularExplainer(
    training_data         = X_test[:500],
    feature_names         = feature_names,
    class_names           = ["Benign","Attack"],
    mode                  = "classification",
    discretize_continuous = False
)
exp = lime_exp.explain_instance(
    instance, mlp.predict_proba,
    num_features=len(feature_names),
    num_samples=N_SAMPLES,
    labels=(1,)
)
lime_dict = dict(exp.as_list(label=1))
lime_vals = np.array([lime_dict.get(f, 0.0) for f in feature_names])
print("  LIME done.")

# =============================================================================
# BUILD TABLE
# Top-N features by absolute SHAP value
# =============================================================================
abs_shap  = np.abs(shap_vals)
top_idx   = np.argsort(abs_shap)[-N_FEATURES:][::-1]

# RRS components
top5_mean      = np.mean(np.sort(abs_shap)[-5:])
shap_evidence  = top5_mean / (np.max(abs_shap) + 1e-9)

sig = (abs_shap > 1e-4) | (np.abs(lime_vals) > 1e-4)
if np.any(sig):
    agreement_score = np.mean(
        np.sign(shap_vals[sig]) == np.sign(lime_vals[sig]))
else:
    agreement_score = 0.0

rrs = 0.45*corr_conf + 0.40*shap_evidence + 0.15*agreement_score

print(f"\n  SHAP evidence  : {shap_evidence:.4f}")
print(f"  LIME agreement : {agreement_score:.4f}")
print(f"  RRS score      : {rrs:.4f}")

# Verdict
pos = np.sum(shap_vals[top_idx] > 0)
verdict = ("HIGH CONFIDENCE ATTACK" if pos >= 4
           else "MEDIUM — SUSPICIOUS" if pos >= 2
           else "LOW SIGNAL")

# Get feature values in original (unscaled) space — raw canonical values
feature_values = instance  # already scaled; show scaled value, note in caption

rows = []
for rank, i in enumerate(top_idx, 1):
    fname   = feature_names[i]
    fval    = instance[i]
    sv      = shap_vals[i]
    lv      = lime_vals[i]
    agree   = (np.sign(sv) == np.sign(lv))
    direction = "$\\rightarrow$ Attack" if sv > 0 else "$\\rightarrow$ Benign"
    rows.append({
        "Rank"      : rank,
        "Feature"   : fname,
        "Value (scaled)": f"{fval:.4f}",
        "SHAP"      : sv,
        "LIME"      : lv,
        "Direction" : direction,
        "Agreement" : "\\checkmark" if agree else "$\\times$"
    })

df_table = pd.DataFrame(rows)

# =============================================================================
# PRINT FORMATTED TABLE
# =============================================================================
print(f"\n{'='*70}")
print(f"  DRILL-DOWN REPORT — Example HIGH-RISK Escalation")
print(f"{'='*70}")
print(f"  Instance index   : {best_i}")
print(f"  Corrected conf   : {corr_conf:.4f}")
print(f"  SHAP evidence    : {shap_evidence:.4f}")
print(f"  LIME agreement   : {agreement_score:.4f}")
print(f"  RRS score        : {rrs:.4f}")
print(f"  Verdict          : {verdict}")
print()
print(f"{'Rank':<5} {'Feature':<20} {'Value':>12} {'SHAP':>10} {'LIME':>10} {'Direction':>18} {'Agree'}")
print("-"*80)
for _, r in df_table.iterrows():
    agree_str = "✓" if "check" in r["Agreement"] else "✗"
    dir_str   = r["Direction"].replace("$\\rightarrow$","→")
    print(f"{r['Rank']:<5} {r['Feature']:<20} {r['Value (scaled)']:>12} "
          f"{r['SHAP']:>10.4f} {r['LIME']:>10.4f} "
          f"{dir_str:>18} {agree_str}")


os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/table6_drilldown.tex", "w") as f:
    f.write(latex)

df_table.to_csv("experiments/results/table6_drilldown.csv", index=False)

print(f"\n{'='*70}")
print(f"\nSaved → experiments/results/table6_drilldown.tex")
print(f"Saved → experiments/results/table6_drilldown.csv")
print("\n✅ SHAP/LIME table complete.")