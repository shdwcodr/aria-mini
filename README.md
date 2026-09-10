# ARIA — Rethinking Alert Triage in Security Operations Centres

Code accompanying the paper *"Automated Reversible Intelligence for Adaptive
Response (ARIA): Rethinking Alert Triage in Security Operations Centres"*
(Palav, Gavhale, Pant, Kumar). This repo contains the feature harmonisation
pipeline, MLP training, deployment-aware prior-shift calibration, the
twin-scoring (QTS/RRS) triage logic, the SOC discrete-event simulation, and
the ablation/validation experiments reported in the paper.

See `REPRODUCIBILITY.md` for the full script → paper-table/figure mapping.

## Repo layout

```
harmonizer.py                  Feature harmonisation into the common 20-feature schema (Table 2)
data_loader.py                 Dataset loading / sampling to a target attack prevalence
model_trainer.py                MLP training (Table 3 sweep; deployed model = 120k samples / 40% attack)
calibration.py                  Prior-shift calibration: ROC, PR, reliability diagram, ECE/Brier (Table 11, Fig. 6)

check_pool_overlap.py           Checks train/test pool overlap
dataset_id_probe.py             Checks whether source dataset is trivially recoverable from features
plot_roc_per_source.py          Per-source ROC overlay (Fig. 4)
descriptive_accuracy.py         SHAP faithfulness / AOPC (Section 4.5.2)
generate_drilldown_table.py     Drill-down report example (Table 8)

experiments/
  seeded_variance_comparison.py   Main SOC simulation: Table 12 + McNemar test, Fig. 7/8/9
  ablation.py                     RRS ablation (Table 13)
  cache.py                        Explanation cache fidelity (Table 14)
  aria_real_mttr_controller.py    Low-load admission-rate sensitivity (Appendix A.2, Table 16)
  mttr_fix.py                     MTTR decomposition supporting the Section 5.2 discussion
  source_per_eval.py              Per-source / overall classification performance (Table 10, Fig. 2, Fig. 3)
  results/                        Script outputs (CSVs, text reports, some figures)

models/
  mlp_alert_train120k_4attack.pkl + scaler_...   The deployed model used throughout Section 4
  sweep/                                          The other four Table 3 configurations

final figures and tables/       The exact figures and tables that appear in the compiled paper
explanations/                   Example SHAP/LIME drill-down text reports and force plots
```

## Run order (from scratch)

```bash
pip install -r requirements.txt

# 1. Train the model
python model_trainer.py

# 2. Per-source evaluation, confusion matrices, ROC
python experiments/source_per_eval.py
python plot_roc_per_source.py

# 3. Calibration — ECE, Brier score, reliability diagram (Table 11, Fig. 6)
python calibration.py

# 4. Main SOC simulation — Table 12, McNemar test, Fig. 7/8/9
python experiments/seeded_variance_comparison.py

# 5. RRS ablation (Table 13)
python experiments/ablation.py

# 6. Explanation cache validation (Table 14) and SHAP faithfulness (Section 4.5.2)
python experiments/cache.py
python descriptive_accuracy.py

# 7. Supporting checks
python check_pool_overlap.py
python dataset_id_probe.py
python experiments/mttr_fix.py
python generate_drilldown_table.py
```

Each script is self-contained (loads its own model/scaler/data) and writes
to `experiments/results/`. They can be run in any order; the sequence above
just follows the paper's section order.

## Data availability

The three source datasets (CIC-IDS2017, UNSW-NB15, CIC-IoT2023) are public
and available from their original sources (see the paper's reference list,
[26], [24], [27]). This repo does not redistribute the raw data — place the
downloaded CSVs where `data_loader.py` expects them before running any
script above.

## License / citation

MIT — see `LICENSE`. If this code is used, please cite the paper (see
`CITATION.cff`).
