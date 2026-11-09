# uncertainty/framework/classifier.py — classify: what kind

Three classifiers on the same labelled answers (PLAN §9.5), with grouped cross-validation so
the same question never sits in two folds.

| Part | What it does |
|---|---|
| `FEATURES`, `RULE_FEATURES` | Full and lite feature lists; the readable features the decision rules may use. |
| `make` | Rules (depth-3 tree), logistic regression, gradient-boosted trees; class-balanced. |
| `oof_predict`, `_folds` | Out-of-fold predictions, `StratifiedGroupKFold` by question. |
| `bootstrap_f1`, `_boot_macro_f1` | Vectorised bootstrap of macro-F1 (checked equal to scikit-learn), alone or as a paired difference. |
| `evaluate` | Baseline, rules, both learned models; best vs baseline CI; leave-one-dataset/model/family-out; printed rules; permutation importance (fit on dev, measured on test); aleatoric subtype classifier; the final fitted models. |
| `predict` | One answer's type, type probabilities and subtype (the analyze tool). |
