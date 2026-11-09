# uncertainty/framework/detector.py — identify: how unsure

Calibrated P(hallucination) from the Objective 1 scores (PLAN §9.5).

| Part | What it does |
|---|---|
| `make_model` | Logistic regression (imputed, with missing indicators, scaled), Platt-calibrated. |
| `evaluate` | Trained on every model's dev split, evaluated on test: AUROC with CI, AURAC, ECE, Brier; the best single method (chosen on dev) and semantic entropy, with DeLong tests; per model and leave-one-model-out; level thresholds from out-of-fold dev predictions. Skips when there is too little data. |
| `level_thresholds`, `level` | low / medium / high. |
| `predict` | One answer's probability and level (the analyze tool). |
