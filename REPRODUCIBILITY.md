# Reproducibility Map

Which script produces each paper table/figure, and where its output lands.

| Paper item | Script | Output |
|---|---|---|
| Table 1 (dataset composition) | `data_loader.py` / `harmonizer.py` | printed at load time |
| Table 2 (feature schema) | `harmonizer.py` | `CANONICAL_COLUMNS` |
| Table 3 (training sweep) | `model_trainer.py`, rerun per row with different `TRAIN_SIZE` / `ATTACK_RATIO_TRAIN` | `models/sweep/*.pkl` |
| Table 4 (prior-shift effect) | `experiments/source_per_eval.py` | `experiments/results/table4_prior_shift_effect.csv` |
| Table 7 (SHAP/LIME timing) | `generate_drilldown_table.py` | `experiments/results/table7_shap_lime_timing.csv` (regenerate locally — see note below) |
| Table 8 (drill-down example) | `generate_drilldown_table.py` | `experiments/results/table6_drilldown.{tex,csv}` |
| Table 10 / Fig. 2 / Fig. 3 | `experiments/source_per_eval.py` | `per_source_classification_FINAL.csv`, `confusion_matrix_*_FINAL.png` |
| Table 11 / Fig. 6 (calibration) | `calibration.py` | `roc_curve_FINAL.png`, `pr_curve_FINAL.png`, `calibration_reliability_FINAL.png`, `calibration_summary_FINAL.csv` |
| Fig. 4 (per-source ROC) | `plot_roc_per_source.py` | `roc_curve_per_source_FINAL.png` |
| Fig. 5 (trainer holdout confusion matrix) | `model_trainer.py` | `final figures and tables/fig05_confusion_matrix_train_holdout.png` |
| Table 12 + McNemar test / Fig. 7 / Fig. 8 / Fig. 9 | `experiments/seeded_variance_comparison.py` | `experiments/results/seeded/{aggregate_summary,per_seed_results}.csv`, `fig07/08/09_*.png` |
| Table 13 (RRS ablation) | `experiments/ablation.py` | `experiments/rss_ablation/results/rss_ablation_FINAL.csv` |
| Table 14 (cache fidelity) | `experiments/cache.py` | `experiments/results/cache_fidelity_results.csv` |
| Table 15 (triage sensitivity, Appendix A.1) | `experiments/seeded_variance_comparison.py`, with `alert_thresh` fixed at 0.40 / 0.55 / 0.60 and the EMA controller update disabled | not persisted as a separate file |
| Table 16 (admission-rate sensitivity, Appendix A.2) | `experiments/aria_real_mttr_controller.py` | `experiments/results/aria_real_mttr_controller/*` |
| Section 4.5.2 (AOPC / SHAP faithfulness) | `descriptive_accuracy.py` | `experiments/results/descriptive_accuracy.csv` |

## Notes on exact reproduction

- **Table 7 (SHAP/LIME timing)** reports wall-clock measurements taken on
  the paper's hardware (Apple M4 Air, 16 GB RAM, single thread). Re-running
  `generate_drilldown_table.py` on different hardware will not reproduce the
  exact millisecond values, though the relative result (cache lookup ~3
  orders of magnitude faster than direct SHAP+LIME computation) should hold
  regardless of machine.
- **Table 1** row counts are printed by `data_loader.py` / `harmonizer.py`
  rather than written to a file; re-run those scripts to see them.
- All other tables and figures are written directly to the paths listed
  above with no manual post-processing required.
