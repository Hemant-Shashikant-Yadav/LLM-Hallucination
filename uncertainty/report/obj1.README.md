# uncertainty/report/obj1.py — the Objective 1 report

Every number on the test split; dev only for thresholds.

| Part | What it does |
|---|---|
| `method_table` | AUROC with bootstrap CI, AUPRC, AURAC, PRR, accuracy at 80/90% for every model × set (and all sets) × method. |
| `delong_table` | Semantic entropy vs every other method, Holm-adjusted. |
| `profile_table` | Baseline uncertainty profile: accuracy, mean entropy, share very unsure, share sure, blind spot, confident-error rate. |
| `calibration_table` | ECE, Brier and reliability bins of P(True) and verbalised confidence. |
| `k_curve_table`, `size_table`, `cost_table` | AUROC vs samples; size, family and generation effects; AUROC vs extra calls and tokens. |
| `judge_agreement`, `layer1_pipeline` | Second-judge and alias agreement; AUROC of the pipeline's stored Layer 1 entropy on its own test runs. |
| `figures` | Heatmap, ROC, rejection curve, reliability, entropy histograms, AUROC vs K, AUROC vs cost, size effects. |
| `_per_model` | One model's tables, cached in `obj1_cache.pkl` beside its scores (keyed on the scores file), so rebuilding after every model does not redo earlier models' bootstraps. |
| `build` | Writes CSVs, LaTeX, figures, `summary.json` and `obj1.md`; `split='all'` for the pilot. |
