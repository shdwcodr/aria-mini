# Final Figures and Tables — Mapping to Paper

This folder contains the figures and tables referenced in the manuscript,
renamed to match the paper's numbering for easy cross-referencing.

| File | Paper reference |
|---|---|
| `fig02_confusion_matrix_per_source.png` | Figure 2 — per-source confusion matrices (prior-corrected, Youden threshold = 0.2498) |
| `fig03_confusion_matrix_overall.png` | Figure 3 — overall confusion matrix, AUC-ROC = 0.9601 |
| `fig04_roc_curve_per_source.png` | Figure 4 — per-source ROC curves |
| `fig05_confusion_matrix_train_holdout.png` | Figure 5 — MLP confusion matrix on the trainer's held-out set, before deployment prior-shift correction |
| `fig06_calibration_reliability.png` | Figure 6 — reliability diagram before/after prior-shift correction |
| `fig07_admission_controller.png` | Figure 7 — adaptive admission controller behaviour |
| `fig08_fatigue_mttr.png` | Figure 8 — analyst fatigue and MTTR over the simulated shift |
| `fig09_investigation_queue.png` | Figure 9 — investigation queue size over the simulated shift |
| `supp_rss_ablation_barchart.png` | Supplementary — visual companion to Table 13 (RRS ablation) |
| `tables.csv` | Underlying numbers for Tables 1, 3, 4, 7, 8, 12, 13, and Appendix Table 15 (see in-file section headers) |

All confusion matrices and ROC curves use the Youden-optimal decision
threshold (0.2498), consistent with Table 10 in the manuscript.

Figure 1 (the ARIA pipeline/architecture diagram) was produced directly in
TikZ for the manuscript and is not a code-generated output, so it is not
included here.
