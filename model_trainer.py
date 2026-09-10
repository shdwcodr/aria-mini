import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from data_loader import DataLoader   # Your new DataLoader

# =============================================================================
# CONFIG
# =============================================================================
TRAIN_SIZE = 120000
ATTACK_RATIO_TRAIN = 0.40
TEST_SIZE = 60000

print(f"=== ARIA Alert MLP Training ===\n"
      f"Train: {TRAIN_SIZE:,} ({ATTACK_RATIO_TRAIN:.0%} attack)\n"
      f"Test : {TEST_SIZE:,} (5% attack)\n")

loader = DataLoader()

# =============================================================================
# TRAINING DATA (Proper split)
# =============================================================================
print("Loading Training Data...")
df_train = loader.load_train()
X_train_full, y_train_full, feature_names, _ = loader.preprocess(
    df_train,
    target_attack_ratio=ATTACK_RATIO_TRAIN,
    target_total_samples=TRAIN_SIZE,
)

print("\nCanonical Features:")
for f in feature_names:
    print(" -", f)

os.makedirs("models", exist_ok=True)
joblib.dump(loader.scaler, f"models/scaler_train{TRAIN_SIZE//1000}k_{int(ATTACK_RATIO_TRAIN*10)}attack.pkl")
print(f"\n✅ Scaler saved: models/scaler_train{TRAIN_SIZE//1000}k_{int(ATTACK_RATIO_TRAIN*10)}attack.pkl")

# =============================================================================
# TEST DATA (Held-out)
# =============================================================================
print("\nLoading Test Data...")
df_test = loader.load_test()
X_test_full, y_test_full, _, _ = loader.preprocess(
    df_test,
    target_attack_ratio=0.05,          # Realistic low attack rate
    target_total_samples=TEST_SIZE,
)

# =============================================================================
# TRAIN / VALIDATION SPLIT
# =============================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full,
)

print(f"\nFinal Split → Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test_full):,}")

# =============================================================================
# MODEL
# Architecture and hyperparameters match paper Section 3.4:
#   4 hidden layers [512, 256, 128, 64], ReLU, Adam (eta=0.0012),
#   L2 regularisation (alpha=0.001), batch size 2048,
#   early stopping with a patience of 35 epochs,
#   class weights w_benign=1.0, w_attack=2.5.
# sklearn's MLPClassifier has no native class_weight parameter, so the
# class weights are applied via sample_weight at fit time instead.
# =============================================================================
CLASS_WEIGHT = {0: 1.0, 1: 2.5}   # {benign: 1.0, attack: 2.5}, per Section 3.4
sample_weight = np.where(y_train == 1, CLASS_WEIGHT[1], CLASS_WEIGHT[0])

mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.0012,
    alpha=0.001,
    batch_size=2048,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=35,
    random_state=42
)

print("\nTraining MLP...")
mlp.fit(X_train, y_train, sample_weight=sample_weight)

# =============================================================================
# EVALUATION
# =============================================================================
y_prob = mlp.predict_proba(X_test_full)[:, 1]
y_pred = (y_prob > 0.5).astype(int)

print("\n=== Classification Report ===")
print(classification_report(
    y_test_full, y_pred,
    target_names=["Benign", "Attack"],
    digits=4
))

print("\n=== Probability Distribution ===")
print(np.percentile(y_prob, [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]).round(4))

print(f"\nTraining Accuracy   : {mlp.score(X_train, y_train):.4f}")
print(f"Validation Accuracy : {mlp.score(X_val, y_val):.4f}")

# =============================================================================
# CONFUSION MATRIX
# =============================================================================
cm = confusion_matrix(y_test_full, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=["Benign", "Attack"]).plot(ax=ax, cmap="Blues")
plt.title(f"ARIA-MLP\nTrain: {TRAIN_SIZE:,} ({ATTACK_RATIO_TRAIN:.0%} attack) | Test: {TEST_SIZE:,}")
plt.savefig(f"models/cm_mlp_train{TRAIN_SIZE//1000}k.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# SAVE MODEL
# =============================================================================
model_path = f"models/mlp_alert_train{TRAIN_SIZE//1000}k_{int(ATTACK_RATIO_TRAIN*10)}attack.pkl"
joblib.dump(mlp, model_path)
print(f"\n✅ Model saved: {model_path}")
print(f"✅ Scaler saved: models/scaler_train{TRAIN_SIZE//1000}k_{int(ATTACK_RATIO_TRAIN*10)}attack.pkl")