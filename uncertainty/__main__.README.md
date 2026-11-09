# uncertainty/__main__.py — the command line

`python -m uncertainty <command>`; `run-all` is the one the Mac needs.

| Part | What it does |
|---|---|
| `run-all` / `check` | The whole study, or its first stage (see `orchestrate/run_all.py`). `--backend fake` for a dry run. |
| `build-sets` | Freeze the question sets. |
| `generate`, `nli`, `label`, `score`, `intervene` | One step for one model (debugging, re-runs). |
| `report`, `train` | Rebuild reports; retrain the Objective 2 framework. |
| `annotate` | Export and import human labels and the judge audit. |
| `analyze` | The uncertainty report for one question. |
| `_one_step` | The single-step commands. |
