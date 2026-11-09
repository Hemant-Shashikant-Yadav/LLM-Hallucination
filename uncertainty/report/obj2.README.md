# uncertainty/report/obj2.py — the Objective 2 report and training

Trains and evaluates the framework, saves it for the analyze tool, writes the report.

| Part | What it does |
|---|---|
| `assemble_all`, `load_human` | Type labels for every in-scope answer; human labels (first annotator) when imported. |
| `_actionability` | Does the intervention matching the predicted type help more than the other? |
| `_human_agreement` | Inter-annotator kappa, human vs weak labels, human vs classifier. |
| `_figures`, `_examples` | Confusion matrix, per-type F1 of the three classifiers, importances; an example gallery. |
| `build` | Detector and classifier (full and lite), tables, rules, models in `results/models/`, `summary.json`, `obj2.md`. |
