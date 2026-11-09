# uncertainty/metrics.py — how well a score predicts mistakes

Selective-prediction and calibration metrics (PLAN §8.2). Convention: higher uncertainty =
less sure; `wrong` = 1 for a wrong answer. Ties are averaged, which matters for coarse
scores such as the number of clusters.

| Part | What it does |
|---|---|
| `auroc`, `auprc` | scikit-learn, after dropping questions the method could not score. |
| `accuracy_curve`, `aurac`, `accuracy_at_coverage` | Accuracy of the r most certain answers; its mean (area under the rejection-accuracy curve, Farquhar et al.); accuracy at 80% / 90% kept. |
| `prr` | Prediction rejection ratio: 1 = oracle, 0 = random. |
| `ece`, `brier`, `reliability` | Calibration of confidences (P(True), verbalised, the detector). |
| `bootstrap_ci`, `bootstrap_auroc` | Percentile CIs over questions; the AUROC one is vectorised (rank formula) because Objective 1 needs thousands. |
| `delong` | DeLong test for two AUROCs on the same questions (fast version, Sun & Xu 2014). |
| `holm` | Holm-adjusted p-values across method comparisons. |
