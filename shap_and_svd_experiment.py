"""
SVD-SHAP: Low-Rank Approximation of SHAP Values for IDS
=========================================================
IEEE Conference Paper Experiment — UNSW-NB15 only
v6: adds a full-rank neural surrogate baseline ("FastSHAP-style, simplified")
    to give an honest accuracy comparison against a learned, full-rank
    amortized explainer — the same family of method as FastSHAP/PDD-SHAP/
    TN-SHAP, without reimplementing their exact training objectives. See
    the caveat printed in methodology_notes.txt: this is NOT a claim of
    reproducing [5]/[6]/[7]'s published results.

v5: fixes a real bug from v4 (KernelSHAP/PermutationSHAP/SamplingSHAP were
    silently dropped from baseline_comparison.csv due to an unsliced
    2-class predict_proba output causing a shape mismatch) and makes the
    Ridge-vs-MLPRegressor comparison fair by regularizing and
    early-stopping the MLPRegressor, which was badly overfitting 200
    training samples with ~2,500 unregularized parameters in v4.

v4: directly resolves two mentor-flagged issues from the v3 results —

  (1) WALL-CLOCK TIME IS HARDWARE-DEPENDENT. Replaced as the primary
      efficiency metric with MODEL QUERY COUNT: every predict_proba
      call is counted via QueryCounter, so "efficiency" is now stated
      as "number of model evaluations per explanation" — deterministic,
      hardware-independent, and the same convention used by FastSHAP
      and PDD-SHAP. Wall-clock time is kept ONLY as a secondary,
      informational column (single machine, no other change made
      mid-run) — never the headline number.

      We also separate ONE-TIME AMORTIZED TRAINING COST (queries spent
      computing full SHAP on the 200-sample training set, needed once
      to fit SVD-SHAP) from MARGINAL PER-EXPLANATION COST (queries
      needed to explain one NEW sample after training). SVD-SHAP's
      real claim is 0 model queries at inference — Ridge/MLP regressor
      predict + one matrix multiply, no calls to the original model at
      all — which is a strictly hardware-independent claim, stronger
      than any wall-clock speedup number.

  (2) IS THE LINEARITY ASSUMPTION (Ridge) COSTING US ACCURACY? Added a
      second coefficient-regressor variant using a small MLPRegressor
      alongside the original Ridge, compared at a fixed k across all
      seeds. Both variants still make ZERO calls to the original model
      at inference — swapping Ridge for MLPRegressor does not change
      the query-count story, only tests whether the mapping from raw
      features to low-rank SHAP coordinates benefits from nonlinearity.

Research Questions:
  Q1: How close is SVD-SHAP to full SHAP?
  Q2: How sensitive is the approximation to Ridge regularization?
  Q3: How stable are the results across random seeds?
  Q4: How does SVD-SHAP compare to other fast SHAP approximations on
      accuracy vs. MODEL QUERY COST (not wall-clock time)?
  Q5 (NEW): Does a nonlinear coefficient regressor (MLPRegressor)
      outperform Ridge for the feature -> low-rank-SHAP-coordinate
      mapping, i.e., is the linear-mapping assumption actually costing
      us accuracy?
"""

import numpy as np
import pandas as pd
import shap
import time
import os
import warnings
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

N_TRAIN_SHAP   = 200
N_TEST_SHAP    = 100
N_BACKGROUND   = 50
RANK_SWEEP     = [1, 2, 3, 4, 5, 7, 10, 15, 20]
RIDGE_ALPHAS   = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
SEEDS          = [42, 7, 123, 2024, 99]
ATTACK_RATIO   = 0.30

COMPARE_K              = 10
KERNEL_NSAMPLES        = 50
KERNEL_BACKGROUND      = 20
SAMPLING_NSAMPLES      = 100
SAMPLING_BACKGROUND    = 50
PERMUTATION_MAX_EVALS  = None
PERMUTATION_BACKGROUND = 50
PCA_KERNEL_NSAMPLES    = 50
PCA_KERNEL_BACKGROUND  = 20
MLP_REGRESSOR_HIDDEN   = (50,)   # small, deliberately lightweight nonlinear regressor

os.makedirs("experiments/svd_shap", exist_ok=True)

print("=" * 70)
print("  SVD-SHAP: Low-Rank SHAP Approximation for IDS")
print("  Dataset: UNSW-NB15 only")
print("=" * 70)


# =============================================================================
# QUERY COUNTER — hardware-independent efficiency metric
# =============================================================================
class QueryCounter:
    """
    Wraps a model's predict_proba and counts total ROWS evaluated (not
    calls — a call with a batch of 50 rows counts as 50 queries), which
    is the standard "model evaluations" convention in the efficient-SHAP
    literature (e.g. FastSHAP). Deterministic and hardware-independent:
    running on a different machine, or with other tabs/processes open,
    never changes this number.
    """
    def __init__(self, model):
        self.model = model
        self.count = 0

    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        self.count += n
        return self.model.predict_proba(X)

    def reset(self):
        self.count = 0


# =============================================================================
# DATA LOADING — UNSW-NB15 only
# =============================================================================
def load_unsw(n_samples=2000, attack_ratio=0.30, seed=42):
    print(f"\nLoading UNSW-NB15...")
    base_path = "data/UNSW-NB15"
    if not os.path.exists(base_path):
        base_path = "/aria-mini/data/UNSW-NB15"
    if not os.path.exists(base_path):
        raise FileNotFoundError("Could not find UNSW-NB15 folder")

    csv_files = [f for f in os.listdir(base_path) if f.lower().endswith(".csv")]
    print(f"  Found {len(csv_files)} CSV files")
    dfs = []
    for f in csv_files:
        path = os.path.join(base_path, f)
        print(f"    Reading: {f}")
        try:
            dfs.append(pd.read_csv(path, nrows=15000))
        except Exception as e:
            print(f"      Skip: {e}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {df.shape[0]:,} rows × {df.shape[1]} cols")

    if "label" in df.columns:
        y = df["label"].astype(int)
    else:
        label_cands = [c for c in df.columns if "label" in c.lower() or "attack" in c.lower()]
        y = df[label_cands[0]].astype(int) if label_cands else pd.Series(np.zeros(len(df)))

    drop_cols = ["srcip", "sport", "dstip", "dsport", "stime", "ltime"]
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.drop(columns=[c for c in drop_cols if c in numeric_df.columns], errors="ignore")

    X = numeric_df.values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    rng = np.random.RandomState(seed)
    if len(X) > n_samples:
        attack_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]
        n_attack = min(int(n_samples * attack_ratio), len(attack_idx))
        n_normal = n_samples - n_attack
        if len(normal_idx) == 0 or len(attack_idx) == 0:
            idx = rng.choice(len(X), n_samples, replace=False)
        else:
            idx = np.concatenate([
                rng.choice(attack_idx, n_attack, replace=False),
                rng.choice(normal_idx, n_normal, replace=False)
            ])
            rng.shuffle(idx)
        X = X[idx]
        y = y.iloc[idx].values

    feat = numeric_df.columns.tolist()
    print(f"  Final → {X.shape[0]} samples, {X.shape[1]} features, "
          f"{int(y.sum())} attacks ({y.mean()*100:.1f}%)")
    return X, y, feat


# =============================================================================
# SHAP + SVD CORE
# =============================================================================
def compute_full_shap(model, X_train, X_bg, label=""):
    """Reference SHAP. Now instrumented with a QueryCounter."""
    print(f"  [FULL SHAP] {len(X_train)} samples {label}...", end=" ", flush=True)
    background = shap.sample(X_bg, min(N_BACKGROUND, len(X_bg)), random_state=42)
    counter = QueryCounter(model)
    explainer = shap.Explainer(counter.predict_proba, background)
    explainer_type = type(explainer).__name__
    t0 = time.perf_counter()
    sv = explainer(X_train).values[:, :, 1]
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.2f}s [{counter.count} model queries] [explainer={explainer_type}]")
    return sv, elapsed, explainer, explainer_type, counter.count


def fit_svd_shap(X_train, S_train, k, alpha=1.0):
    """Ridge (linear) coefficient regressor — the main method."""
    U, sigma, Vt = np.linalg.svd(S_train, full_matrices=False)
    explained_var = (sigma[:k]**2).sum() / (sigma**2).sum()
    targets_k = U[:, :k] * sigma[:k]
    reg = Ridge(alpha=alpha)
    reg.fit(X_train, targets_k)
    return Vt[:k, :], reg, explained_var


def fit_svd_shap_nonlinear(X_train, S_train, k, hidden=MLP_REGRESSOR_HIDDEN, seed=42):
    """
    Nonlinear coefficient regressor (MLPRegressor) mapping raw features
    -> low-rank SHAP coordinates, replacing Ridge. Tests whether the
    linear-mapping assumption in the main method is costing accuracy.
    Still makes ZERO calls to the original classifier at inference —
    only the internal regressor changes.

    v5 fairness fix: the v4 run used an UNREGULARIZED MLPRegressor
    (default alpha=1e-4) with ~2,500 parameters fit on only 200 training
    samples — a near-guaranteed overfit, not a fair test of whether
    nonlinearity helps. Added explicit L2 regularization (alpha) and
    early_stopping with a held-out validation split, so a loss to Ridge
    here reflects the comparison honestly rather than an undertuned
    baseline.
    """
    U, sigma, Vt = np.linalg.svd(S_train, full_matrices=False)
    explained_var = (sigma[:k]**2).sum() / (sigma**2).sum()
    targets_k = U[:, :k] * sigma[:k]
    reg = MLPRegressor(
        hidden_layer_sizes=hidden,
        alpha=1.0,                 # meaningful L2 regularization, not sklearn's tiny 1e-4 default
        early_stopping=True,       # stop before overfitting the 200-sample training set
        validation_fraction=0.2,
        n_iter_no_change=15,
        max_iter=2000,
        random_state=seed,
    )
    reg.fit(X_train, targets_k)
    return Vt[:k, :], reg, explained_var


def predict_svd_shap(X_new, Vk, reg):
    """Zero model queries — Ridge/MLP predict + one matrix multiply, no calls to f."""
    t0 = time.perf_counter()
    coeff = reg.predict(X_new)
    if len(coeff.shape) == 1:
        coeff = coeff.reshape(-1, 1)
    if coeff.shape[1] != Vk.shape[0]:
        Vk = Vk.T
    shap_approx = coeff @ Vk
    elapsed = time.perf_counter() - t0
    return shap_approx, elapsed


def top_n_overlap(true_shap, approx_shap, n=10):
    overlaps = []
    for i in range(len(true_shap)):
        top_true   = set(np.argsort(np.abs(true_shap[i]))[-n:])
        top_approx = set(np.argsort(np.abs(approx_shap[i]))[-n:])
        overlaps.append(len(top_true & top_approx) / n)
    return np.mean(overlaps)


def feature_correlation_rank(X, threshold=0.95):
    _, sigma, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    cumvar = np.cumsum(sigma**2) / (sigma**2).sum()
    k = int(np.searchsorted(cumvar, threshold)) + 1
    return k, cumvar, sigma


# =============================================================================
# BASELINE SHAP METHODS — all instrumented with QueryCounter
# =============================================================================
def run_kernel_shap(model, X_test, X_bg, n_background=KERNEL_BACKGROUND, nsamples=KERNEL_NSAMPLES):
    print(f"  [Kernel SHAP] background={n_background}, nsamples={nsamples}...", end=" ", flush=True)
    background = shap.sample(X_bg, min(n_background, len(X_bg)), random_state=42)
    counter = QueryCounter(model)
    # BUGFIX (v5): must slice to the attack-class probability, or shap returns a
    # (n, d, 2)-shaped array that silently fails the shape check in
    # score_against_truth() and gets dropped from the results with no error.
    def f(x):
        return counter.predict_proba(x)[:, 1]
    explainer = shap.KernelExplainer(f, background)
    t0 = time.perf_counter()
    sv = explainer.shap_values(X_test, nsamples=nsamples, silent=True)
    elapsed = time.perf_counter() - t0
    sv = np.array(sv)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    print(f"done in {elapsed:.2f}s [{counter.count} model queries]")
    return sv, elapsed, counter.count


def run_permutation_shap(model, X_test, X_bg, n_background=PERMUTATION_BACKGROUND, max_evals=PERMUTATION_MAX_EVALS):
    print(f"  [Permutation SHAP] background={n_background}...", end=" ", flush=True)
    background = shap.sample(X_bg, min(n_background, len(X_bg)), random_state=42)
    counter = QueryCounter(model)
    def f(x):  # BUGFIX (v5): see note in run_kernel_shap
        return counter.predict_proba(x)[:, 1]
    kwargs = {} if max_evals is None else {"max_evals": max_evals}
    explainer = shap.PermutationExplainer(f, background)
    t0 = time.perf_counter()
    sv = explainer(X_test, **kwargs).values
    elapsed = time.perf_counter() - t0
    sv = np.array(sv)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    print(f"done in {elapsed:.2f}s [{counter.count} model queries]")
    return sv, elapsed, counter.count


def run_sampling_shap(model, X_test, X_bg, n_background=SAMPLING_BACKGROUND, nsamples=SAMPLING_NSAMPLES):
    print(f"  [Sampling SHAP] background={n_background}, nsamples={nsamples}...", end=" ", flush=True)
    background = shap.sample(X_bg, min(n_background, len(X_bg)), random_state=42)
    counter = QueryCounter(model)
    def f(x):  # BUGFIX (v5): see note in run_kernel_shap
        return counter.predict_proba(x)[:, 1]
    explainer = shap.SamplingExplainer(f, background)
    t0 = time.perf_counter()
    sv = explainer.shap_values(X_test, nsamples=nsamples)
    elapsed = time.perf_counter() - t0
    sv = np.array(sv)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    print(f"done in {elapsed:.2f}s [{counter.count} model queries]")
    return sv, elapsed, counter.count


def run_pca_then_shap(model, X_train, X_test, X_bg, k, n_background=PCA_KERNEL_BACKGROUND,
                       nsamples=PCA_KERNEL_NSAMPLES):
    from sklearn.decomposition import PCA
    print(f"  [PCA-then-SHAP] k={k}...", end=" ", flush=True)

    pca = PCA(n_components=k, random_state=42)
    X_tr_pca = pca.fit_transform(X_train)
    X_te_pca = pca.transform(X_test)
    X_bg_pca = pca.transform(X_bg)

    counter = QueryCounter(model)
    def f_pca(x_pca):
        x_orig = pca.inverse_transform(x_pca)
        return counter.predict_proba(x_orig)[:, 1]

    background = shap.sample(X_bg_pca, min(n_background, len(X_bg_pca)), random_state=42)
    explainer = shap.KernelExplainer(f_pca, background)

    t0 = time.perf_counter()
    sv_pca = explainer.shap_values(X_te_pca, nsamples=nsamples, silent=True)
    elapsed = time.perf_counter() - t0

    sv_pca = np.array(sv_pca)
    if sv_pca.ndim == 1:
        sv_pca = sv_pca.reshape(1, -1)

    S_approx = sv_pca @ pca.components_
    print(f"done in {elapsed:.2f}s [{counter.count} model queries]")
    return S_approx, elapsed, sv_pca, pca, counter.count


def diagnose_pca_component_alignment(S_true, sv_pca, pca):
    k = sv_pca.shape[1]
    rows = []
    for j in range(k):
        true_proj_j = S_true @ pca.components_[j]
        if np.std(true_proj_j) < 1e-12 or np.std(sv_pca[:, j]) < 1e-12:
            corr = np.nan
        else:
            corr, _ = pearsonr(true_proj_j, sv_pca[:, j])
        rows.append({"component": j, "component_pearson": corr,
                      "explained_var_ratio": pca.explained_variance_ratio_[j]})
    return pd.DataFrame(rows)


def score_against_truth(S_true, S_approx, t_true_ms, t_approx, n_test, queries):
    if S_approx.shape != S_true.shape:
        return None
    flat_t, flat_a = S_true.flatten(), S_approx.flatten()
    pearson, _ = pearsonr(flat_t, flat_a)
    top10 = top_n_overlap(S_true, S_approx, n=10)
    t_approx_ms = t_approx / n_test * 1000
    speedup = t_true_ms / max(t_approx_ms, 1e-9)
    queries_per_sample = queries / n_test
    return {"pearson": pearson, "top10_overlap": top10, "t_ms": t_approx_ms,
            "speedup_informational_only": speedup, "queries_per_sample": queries_per_sample}


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================
def run_one_seed(seed, ridge_alpha=1.0, run_baselines=True, run_pca_diagnostic=True,
                  run_nonlinear_comparison=True, run_surrogate_comparison=True):
    print(f"\n{'─'*60}")
    print(f"  Seed = {seed} | Ridge α = {ridge_alpha}")
    print(f"{'─'*60}")

    X, y, feat = load_unsw(n_samples=N_TRAIN_SHAP + N_TEST_SHAP + 200, attack_ratio=ATTACK_RATIO, seed=seed)

    X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_tr_m = scaler.fit_transform(X_tr_m)
    X_te_m = scaler.transform(X_te_m)

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=seed, learning_rate_init=0.001)
    mlp.fit(X_tr_m, y_tr_m)
    print(f"  Model Acc  train={mlp.score(X_tr_m, y_tr_m):.3f}  test={mlp.score(X_te_m, y_te_m):.3f}")

    X = scaler.transform(X)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    idx_tr = idx[:N_TRAIN_SHAP]
    idx_te = idx[N_TRAIN_SHAP:N_TRAIN_SHAP + N_TEST_SHAP]
    idx_bg = idx[N_TRAIN_SHAP + N_TEST_SHAP:]
    X_tr, X_te, X_bg = X[idx_tr], X[idx_te], X[idx_bg]

    S_train, _, explainer, explainer_type, train_queries = compute_full_shap(mlp, X_tr, X_bg, "(train, one-time amortized cost)")
    t0 = time.perf_counter()
    ref_counter = QueryCounter(mlp)
    explainer_test = shap.Explainer(ref_counter.predict_proba, shap.sample(X_bg, min(N_BACKGROUND, len(X_bg)), random_state=42))
    S_test_true = explainer_test(X_te).values[:, :, 1]
    t_full = time.perf_counter() - t0
    t_full_ms = t_full / len(X_te) * 1000
    test_queries = ref_counter.count
    print(f"  [FULL SHAP] test set: {t_full:.2f}s ({t_full_ms:.1f} ms/sample) "
          f"[{test_queries} queries, {test_queries/len(X_te):.1f}/sample]")

    d = X.shape[1]
    k_feat_95, cumvar_feat, sigma_feat = feature_correlation_rank(X_tr, 0.95)
    _, sigma_S, _ = np.linalg.svd(S_train, full_matrices=False)
    cumvar_S = np.cumsum(sigma_S**2) / (sigma_S**2).sum()
    k_shap_95 = int(np.searchsorted(cumvar_S, 0.95)) + 1

    rank_rows = []
    pca_rows = []
    pca_diag_rows = []
    for k in RANK_SWEEP:
        if k > min(d, N_TRAIN_SHAP):
            continue
        Vk, reg, expl_var = fit_svd_shap(X_tr, S_train, k, alpha=ridge_alpha)
        S_approx, t_approx = predict_svd_shap(X_te, Vk, reg)  # 0 model queries — no counter needed
        t_approx_ms = t_approx / len(X_te) * 1000

        flat_t, flat_a = S_test_true.flatten(), S_approx.flatten()
        pearson, _ = pearsonr(flat_t, flat_a)
        spearman, _ = spearmanr(flat_t, flat_a)
        mae = mean_absolute_error(flat_t, flat_a)

        rank_agree = np.mean([
            len(set(np.argsort(np.abs(S_test_true[i]))[-5:]) & set(np.argsort(np.abs(S_approx[i]))[-5:])) / 5
            for i in range(len(X_te))
        ])
        top10 = top_n_overlap(S_test_true, S_approx, n=10)
        speedup = t_full_ms / max(t_approx_ms, 1e-9)

        rank_rows.append({
            "seed": seed, "alpha": ridge_alpha, "k": k, "expl_var": expl_var,
            "pearson": pearson, "spearman": spearman, "mae": mae,
            "rank_agree_top5": rank_agree, "top10_overlap": top10,
            "t_full_ms": t_full_ms, "t_approx_ms": t_approx_ms,
            "speedup_informational_only": speedup, "queries_per_sample": 0
        })
        print(f"  k={k:2d} | Pearson={pearson:.4f} | Top10={top10:.3f} | "
              f"queries/sample=0 | speedup(informational)={speedup:.0f}x")

        if k in [5, 10, 15]:
            try:
                S_pca, t_pca, sv_pca, pca_obj, pca_queries = run_pca_then_shap(mlp, X_tr, X_te, X_bg, k=k)
                if S_pca.shape == S_test_true.shape:
                    p_pca, _ = pearsonr(S_test_true.flatten(), S_pca.flatten())
                    t10_pca = top_n_overlap(S_test_true, S_pca, n=10)
                    t_pca_ms = t_pca / len(X_te) * 1000
                    pca_rows.append({
                        "seed": seed, "k": k, "pearson": p_pca, "top10_overlap": t10_pca,
                        "t_ms": t_pca_ms, "speedup_informational_only": t_full_ms / max(t_pca_ms, 1e-9),
                        "queries_per_sample": pca_queries / len(X_te)
                    })
                    print(f"    PCA-then-SHAP k={k} → Pearson={p_pca:.4f} Top10={t10_pca:.3f} "
                          f"queries/sample={pca_queries/len(X_te):.1f}")

                    if run_pca_diagnostic:
                        diag = diagnose_pca_component_alignment(S_test_true, sv_pca, pca_obj)
                        diag["seed"] = seed
                        diag["k"] = k
                        n_aligned = (diag["component_pearson"] > 0).sum()
                        print(f"      diagnostic: {n_aligned}/{k} components positively aligned "
                              f"(mean component_pearson={diag['component_pearson'].mean():.3f})")
                        pca_diag_rows.append(diag)
            except Exception as e:
                print(f"    PCA-then-SHAP failed at k={k}: {e}")

    baseline_rows = []
    nonlinear_rows = []
    if run_baselines:
        svd_at_k = next((r for r in rank_rows if r["k"] == COMPARE_K), None)

        try:
            S_kernel, t_kernel, kernel_queries = run_kernel_shap(mlp, X_te, X_bg)
            res = score_against_truth(S_test_true, S_kernel, t_full_ms, t_kernel, len(X_te), kernel_queries)
            if res:
                baseline_rows.append({"seed": seed, "method": "KernelSHAP", **res})
        except Exception as e:
            print(f"  KernelSHAP failed: {e}")

        try:
            S_perm, t_perm, perm_queries = run_permutation_shap(mlp, X_te, X_bg)
            res = score_against_truth(S_test_true, S_perm, t_full_ms, t_perm, len(X_te), perm_queries)
            if res:
                baseline_rows.append({"seed": seed, "method": "PermutationSHAP", **res})
        except Exception as e:
            print(f"  Permutation SHAP failed: {e}")

        try:
            S_samp, t_samp, samp_queries = run_sampling_shap(mlp, X_te, X_bg)
            res = score_against_truth(S_test_true, S_samp, t_full_ms, t_samp, len(X_te), samp_queries)
            if res:
                baseline_rows.append({"seed": seed, "method": "SamplingSHAP", **res})
        except Exception as e:
            print(f"  Sampling SHAP failed: {e}")

        if svd_at_k:
            baseline_rows.append({
                "seed": seed, "method": f"SVD-SHAP-Ridge (k={COMPARE_K})",
                "pearson": svd_at_k["pearson"], "top10_overlap": svd_at_k["top10_overlap"],
                "t_ms": svd_at_k["t_approx_ms"],
                "speedup_informational_only": svd_at_k["speedup_informational_only"],
                "queries_per_sample": 0,
            })

        pca_at_k = next((r for r in pca_rows if r["k"] == COMPARE_K), None)
        if pca_at_k:
            baseline_rows.append({
                "seed": seed, "method": f"PCA-then-SHAP (k={COMPARE_K})",
                "pearson": pca_at_k["pearson"], "top10_overlap": pca_at_k["top10_overlap"],
                "t_ms": pca_at_k["t_ms"], "speedup_informational_only": pca_at_k["speedup_informational_only"],
                "queries_per_sample": pca_at_k["queries_per_sample"],
            })

    # =========================================================================
    # NEW: Nonlinear coefficient-regressor comparison (Ridge vs MLPRegressor)
    # at a fixed k, to directly test the linear-mapping assumption.
    # =========================================================================
    if run_nonlinear_comparison:
        Vk_lin, reg_lin, _ = fit_svd_shap(X_tr, S_train, COMPARE_K, alpha=ridge_alpha)
        S_lin, _ = predict_svd_shap(X_te, Vk_lin, reg_lin)
        p_lin, _ = pearsonr(S_test_true.flatten(), S_lin.flatten())
        t10_lin = top_n_overlap(S_test_true, S_lin, n=10)

        Vk_nl, reg_nl, _ = fit_svd_shap_nonlinear(X_tr, S_train, COMPARE_K, seed=seed)
        S_nl, _ = predict_svd_shap(X_te, Vk_nl, reg_nl)
        p_nl, _ = pearsonr(S_test_true.flatten(), S_nl.flatten())
        t10_nl = top_n_overlap(S_test_true, S_nl, n=10)

        nonlinear_rows.append({"seed": seed, "regressor": "Ridge (linear)", "k": COMPARE_K,
                                "pearson": p_lin, "top10_overlap": t10_lin, "queries_per_sample": 0})
        nonlinear_rows.append({"seed": seed, "regressor": "MLPRegressor (nonlinear)", "k": COMPARE_K,
                                "pearson": p_nl, "top10_overlap": t10_nl, "queries_per_sample": 0})
        print(f"  Regressor comparison @ k={COMPARE_K}: Ridge Pearson={p_lin:.4f}  "
              f"MLPRegressor Pearson={p_nl:.4f}  (both 0 model queries)")

    # =========================================================================
    # NEW: Full-rank neural surrogate baseline — "FastSHAP-style, simplified"
    # Trains a single regularized MLP to predict the FULL d-dimensional SHAP
    # vector directly from raw features (no SVD compression), amortizing
    # explanation cost the same way FastSHAP [5] does: train once, then
    # zero model queries per explanation thereafter. This is NOT a
    # reimplementation of FastSHAP's actual weighted least-squares training
    # objective (which enforces the Shapley efficiency property) — it is a
    # plain regression to reference SHAP values, offered as an honest,
    # feasible stand-in for "what a learned full-rank surrogate can achieve
    # here," not a claim of matching [5]'s published method or results.
    # =========================================================================
    if run_surrogate_comparison:
        surrogate_reg = MLPRegressor(
            hidden_layer_sizes=(100,),
            alpha=1.0,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=15,
            max_iter=2000,
            random_state=seed,
        )
        surrogate_reg.fit(X_tr, S_train)          # trained once — amortized cost
        S_surrogate = surrogate_reg.predict(X_te)  # 0 model queries at inference
        p_sur, _ = pearsonr(S_test_true.flatten(), S_surrogate.flatten())
        t10_sur = top_n_overlap(S_test_true, S_surrogate, n=10)
        print(f"  Full-rank surrogate (FastSHAP-style, simplified): "
              f"Pearson={p_sur:.4f}  Top10={t10_sur:.3f}  (0 model queries)")
        if baseline_rows or run_baselines:
            baseline_rows.append({
                "seed": seed, "method": "Surrogate-NN (FastSHAP-style, simplified)",
                "pearson": p_sur, "top10_overlap": t10_sur,
                "t_ms": 0.0, "speedup_informational_only": np.nan,
                "queries_per_sample": 0,
            })

    return (pd.DataFrame(rank_rows),
            pd.DataFrame(baseline_rows) if baseline_rows else None,
            pd.DataFrame(pca_rows) if pca_rows else None,
            pd.concat(pca_diag_rows, ignore_index=True) if pca_diag_rows else None,
            pd.DataFrame(nonlinear_rows) if nonlinear_rows else None,
            {
                "d": d, "k_feat_95": k_feat_95, "k_shap_95": k_shap_95,
                "cumvar_feat": cumvar_feat, "cumvar_S": cumvar_S,
                "explainer_type": explainer_type, "model_type": type(mlp).__name__,
                "train_queries": train_queries,
            })


# =============================================================================
# RUN EVERYTHING
# =============================================================================
all_rank, all_baselines, all_pca, all_pca_diag, all_nonlinear = [], [], [], [], []
meta = None

print("\n>>> Phase 1: Multiple seeds with default Ridge α=1.0 (+ baselines + nonlinear comparison)")
for seed in SEEDS:
    df_rank, df_base, df_pca, df_pca_diag, df_nl, meta = run_one_seed(
        seed, ridge_alpha=1.0, run_baselines=True, run_pca_diagnostic=True,
        run_nonlinear_comparison=True, run_surrogate_comparison=True
    )
    all_rank.append(df_rank)
    if df_base is not None and len(df_base) > 0:
        all_baselines.append(df_base)
    if df_pca is not None and len(df_pca) > 0:
        all_pca.append(df_pca)
    if df_pca_diag is not None and len(df_pca_diag) > 0:
        all_pca_diag.append(df_pca_diag)
    if df_nl is not None and len(df_nl) > 0:
        all_nonlinear.append(df_nl)

df_seeds = pd.concat(all_rank, ignore_index=True)
df_seeds.to_csv("experiments/svd_shap/unsw_seeds.csv", index=False)

print("\n" + "="*70)
print("  MEAN ± STD across seeds (Ridge α = 1.0)")
print("="*70)
summary = df_seeds.groupby("k").agg({
    "pearson": ["mean", "std"], "top10_overlap": ["mean", "std"],
    "queries_per_sample": ["mean"], "speedup_informational_only": ["mean", "std"]
}).round(4)
print(summary)
summary.to_csv("experiments/svd_shap/unsw_mean_std.csv")

if all_baselines:
    df_baselines = pd.concat(all_baselines, ignore_index=True)
    df_baselines.to_csv("experiments/svd_shap/baseline_comparison.csv", index=False)
    print("\n" + "="*70)
    print(f"  BASELINE COMPARISON — mean ± std over {len(SEEDS)} seeds (anchor k={COMPARE_K})")
    print("  PRIMARY METRIC: queries_per_sample (hardware-independent)")
    print("="*70)
    baseline_summary = df_baselines.groupby("method").agg({
        "pearson": ["mean", "std"], "top10_overlap": ["mean", "std"],
        "queries_per_sample": ["mean"], "speedup_informational_only": ["mean", "std"]
    }).round(4)
    print(baseline_summary)
    baseline_summary.to_csv("experiments/svd_shap/baseline_comparison_summary.csv")

# =============================================================================
# NEW: NONLINEAR VS LINEAR REGRESSOR COMPARISON
# =============================================================================
if all_nonlinear:
    df_nonlinear = pd.concat(all_nonlinear, ignore_index=True)
    df_nonlinear.to_csv("experiments/svd_shap/regressor_comparison.csv", index=False)
    print("\n" + "="*70)
    print(f"  LINEAR (Ridge) vs NONLINEAR (MLPRegressor) coefficient regressor @ k={COMPARE_K}")
    print("="*70)
    reg_summary = df_nonlinear.groupby("regressor").agg({
        "pearson": ["mean", "std"], "top10_overlap": ["mean", "std"]
    }).round(4)
    print(reg_summary)
    print("\n  Interpretation for the paper:")
    print("  - If MLPRegressor meaningfully beats Ridge, the linear-mapping assumption")
    print("    is costing accuracy — report this as a limitation resolved / extension point.")
    print("  - If they're close, the linear assumption is NOT a meaningful bottleneck —")
    print("    report this as justification for keeping the simpler, cheaper Ridge version.")
    reg_summary.to_csv("experiments/svd_shap/regressor_comparison_summary.csv")

if all_pca_diag:
    df_pca_diag = pd.concat(all_pca_diag, ignore_index=True)
    df_pca_diag.to_csv("experiments/svd_shap/pca_component_diagnostic.csv", index=False)
    diag_summary = df_pca_diag.groupby("k").agg(
        mean_component_pearson=("component_pearson", "mean"),
        frac_positively_aligned=("component_pearson", lambda s: (s > 0).mean())
    ).round(3)
    diag_summary.to_csv("experiments/svd_shap/pca_component_diagnostic_summary.csv")

# =============================================================================
# RIDGE ALPHA ABLATION
# =============================================================================
print("\n>>> Phase 2: Ridge Alpha Ablation (seed=42)")
ablation_rows = []
for alpha in RIDGE_ALPHAS:
    df_a, _, _, _, _, _ = run_one_seed(seed=42, ridge_alpha=alpha, run_baselines=False,
                                        run_pca_diagnostic=False, run_nonlinear_comparison=False,
                                        run_surrogate_comparison=False)
    for k in [5, 10]:
        row = df_a[df_a["k"] == k].iloc[0]
        ablation_rows.append({"alpha": alpha, "k": k, "pearson": row["pearson"],
                               "top10_overlap": row["top10_overlap"], "mae": row["mae"]})
df_ablation = pd.DataFrame(ablation_rows)
df_ablation.to_csv("experiments/svd_shap/unsw_ridge_ablation.csv", index=False)
print(df_ablation.to_string(index=False))

if all_pca:
    df_pca_all = pd.concat(all_pca, ignore_index=True)
    df_pca_all.to_csv("experiments/svd_shap/unsw_pca_baseline.csv", index=False)

# =============================================================================
# METHODOLOGY NOTES
# =============================================================================
methods_config = pd.DataFrame([
    {"method": "Full SHAP (reference)", "algorithm": meta["explainer_type"],
     "background_size": N_BACKGROUND, "note": f"Not exact 2^d enumeration — intractable at d={meta['d']}."},
    {"method": "SVD-SHAP-Ridge", "algorithm": "Truncated SVD + Ridge", "note": "Ours (main)"},
    {"method": "SVD-SHAP-MLP", "algorithm": "Truncated SVD + MLPRegressor", "note": "Ours (nonlinear variant)"},
    {"method": "PCA-then-SHAP", "algorithm": "PCA + KernelExplainer", "note": "Naive baseline"},
    {"method": "KernelSHAP", "algorithm": "shap.KernelExplainer", "note": ""},
    {"method": "PermutationSHAP", "algorithm": "shap.PermutationExplainer", "note": "May coincide with reference"},
    {"method": "SamplingSHAP", "algorithm": "shap.SamplingExplainer", "note": ""},
])
methods_config.to_csv("experiments/svd_shap/methods_config.csv", index=False)

with open("experiments/svd_shap/methodology_notes.txt", "w") as f:
    f.write("SVD-SHAP experiment — methodology notes for paper writing (v4)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"'Full SHAP' reference explainer: {meta['explainer_type']}. Not exact enumeration "
            f"(intractable at d={meta['d']}) — state this explicitly.\n\n")
    f.write("EFFICIENCY METRIC: use queries_per_sample (model evaluations), NOT wall-clock time, "
            "as the primary efficiency claim in the paper. It is hardware-independent and matches "
            "the convention used by FastSHAP/PDD-SHAP. Wall-clock time (t_ms, "
            "speedup_informational_only columns) is kept only as secondary/illustrative context — "
            "state your machine spec once in Experimental Setup if you report it at all, and note "
            "it is not the basis for any comparative claim.\n\n")
    f.write(f"AMORTIZED TRAINING COST: computing full SHAP on the {N_TRAIN_SHAP}-sample training "
            f"set to fit SVD-SHAP costs {meta['train_queries']} model queries — a ONE-TIME cost "
            "paid once regardless of how many future samples are explained. State this explicitly: "
            "SVD-SHAP's headline claim is 0 queries at INFERENCE time (per new explanation), not "
            "0 total cost ever. This is the same amortization framing FastSHAP and PDD-SHAP use "
            "for their own training phase — cite both when making this argument.\n\n")
    f.write("LINEARITY ASSUMPTION: see regressor_comparison_summary.csv for Ridge vs MLPRegressor "
            "at k=10. Report whichever direction the result goes — both are legitimate findings; "
            "state the comparison honestly rather than picking the outcome that looks better.\n\n")
    f.write("FULL-RANK SURROGATE BASELINE (v6): 'Surrogate-NN (FastSHAP-style, simplified)' in "
            "baseline_comparison.csv is a plain regularized MLP trained to predict the FULL "
            "d-dimensional SHAP vector directly (no SVD compression), included to give an honest "
            "accuracy comparison against the 'train once, explain free' family of methods "
            "(FastSHAP/PDD-SHAP/TN-SHAP). IMPORTANT FOR THE PAPER: this is NOT a reimplementation "
            "of FastSHAP's actual training objective (a weighted least-squares loss enforcing the "
            "Shapley efficiency property) or of PDD-SHAP's ANOVA decomposition or TN-SHAP's tensor "
            "network — state this explicitly wherever this baseline is reported, e.g.: 'a simplified "
            "full-rank neural surrogate inspired by the amortization principle in FastSHAP [5], not "
            "a reproduction of its published training objective or results.' Do not caption this row "
            "as 'FastSHAP' alone in Table II — use the full qualified name.\n")

print("\n✅ Experiment complete (v4). Key new outputs:")
print("   - baseline_comparison.csv / baseline_comparison_summary.csv (queries_per_sample is now primary)")
print("   - regressor_comparison.csv / regressor_comparison_summary.csv (NEW — Ridge vs MLPRegressor)")
print("   - methodology_notes.txt (explains both mentor-flag fixes)")

# =============================================================================
# PLOTS
# =============================================================================
print("\nGenerating plots...")
mean_df = df_seeds.groupby("k").mean(numeric_only=True).reset_index()
std_df = df_seeds.groupby("k").std(numeric_only=True).reset_index()

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("SVD-SHAP on UNSW-NB15 — Key Results v4 (mean ± std over 5 seeds)", fontsize=14, fontweight="bold")

ax = axes[0, 0]
ax.errorbar(mean_df["k"], mean_df["pearson"], yerr=std_df["pearson"], fmt="o-", color="#1f77b4",
            linewidth=2, capsize=4, label="SVD-SHAP (Ridge)")
ax.axhline(0.95, color="gray", linestyle="--", linewidth=1, label="0.95 threshold")
if all_pca:
    pca_mean = df_pca_all.groupby("k")["pearson"].mean()
    ax.plot(pca_mean.index, pca_mean.values, "s--", color="#d62728", linewidth=2, label="PCA-then-SHAP")
ax.set_xlabel("Rank k"); ax.set_ylabel("Pearson Correlation"); ax.set_title("Q1: Approximation Accuracy")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0.5, 1.02)

ax = axes[0, 1]
ax.errorbar(mean_df["k"], mean_df["top10_overlap"], yerr=std_df["top10_overlap"], fmt="s-",
            color="#ff7f0e", linewidth=2, capsize=4, label="SVD-SHAP (Ridge)")
if all_pca:
    pca_t10 = df_pca_all.groupby("k")["top10_overlap"].mean()
    ax.plot(pca_t10.index, pca_t10.values, "D--", color="#d62728", linewidth=2, label="PCA-then-SHAP")
ax.set_xlabel("Rank k"); ax.set_ylabel("Top-10 Feature Overlap"); ax.set_title("Q1: Feature Ranking Preservation")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0.3, 1.0)

ax = axes[0, 2]
if all_baselines:
    method_means = df_baselines.groupby("method").agg({"queries_per_sample": "mean"}).reset_index()
    method_means = method_means.sort_values("queries_per_sample")
    colors = ["#1f77b4" if "SVD-SHAP" in m else "#999999" for m in method_means["method"]]
    ax.barh(method_means["method"], method_means["queries_per_sample"] + 0.01, color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Model queries per explanation (log)")
    ax.set_title("Computational Cost — Hardware-Independent")
    ax.grid(True, alpha=0.3, axis="x")

ax = axes[1, 0]
if meta is not None:
    k_max = min(25, len(meta["cumvar_feat"]), len(meta["cumvar_S"]))
    ax.plot(range(1, k_max+1), meta["cumvar_feat"][:k_max], "o-", color="#1f77b4", linewidth=2, label="Feature covariance")
    ax.plot(range(1, k_max+1), meta["cumvar_S"][:k_max], "s--", color="#ff7f0e", linewidth=2, label="SHAP matrix")
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Number of components k"); ax.set_ylabel("Cumulative explained variance")
ax.set_title("Eigenvalue Decay: Features vs SHAP"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
if all_nonlinear:
    reg_means = df_nonlinear.groupby("regressor").agg({"pearson": "mean", "top10_overlap": "mean"}).reset_index()
    xpos = np.arange(len(reg_means))
    width = 0.35
    ax.bar(xpos - width/2, reg_means["pearson"], width, label="Pearson", color="#9467bd")
    ax.bar(xpos + width/2, reg_means["top10_overlap"], width, label="Top-10 Overlap", color="#8c564b")
    ax.set_xticks(xpos); ax.set_xticklabels(reg_means["regressor"], fontsize=8, rotation=10)
    ax.set_title(f"Ridge vs MLPRegressor @ k={COMPARE_K}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1, 2]
ax.axis("off")
table_data = []
headers = ["k", "Pearson", "Top-10", "Queries/sample"]
for k in [5, 10, 15, 20]:
    if k in mean_df["k"].values:
        r = mean_df[mean_df["k"] == k].iloc[0]
        table_data.append([int(k), f"{r['pearson']:.3f}", f"{r['top10_overlap']:.3f}", "0"])
tbl = ax.table(cellText=table_data, colLabels=headers, cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.3, 1.8)
ax.set_title("Summary at key k values", pad=20)

plt.tight_layout()
plt.savefig("experiments/svd_shap/unsw_svd_shap_results.png", dpi=300, bbox_inches="tight")
plt.show()
print("Plot saved → experiments/svd_shap/unsw_svd_shap_results.png")