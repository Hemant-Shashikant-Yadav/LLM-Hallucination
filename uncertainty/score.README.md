# uncertainty/score.py — every score per question

Runs every Objective 1 method and every Objective 2 feature on every answered question of
a model, offline, and writes `results/<model>/scores.csv`. Re-running it costs no model time.

| Part | What it does |
|---|---|
| `K_CURVE`, `LITE_K`, `LITE_METHODS` | Sample counts for the AUROC-vs-K analysis; the lite budget (4 samples, no self-evaluation calls). |
| `record_features(rec, view)` | One question's row: sample counts, normalised entropy, confidences, probes, cost, every method, entropy at K = 2…8, the pipeline's own K = 4 entropy, Objective 2 structure and question features, and the same signals at the lite budget (`lite__*`). Also used by the analyze tool. |
| `score_model` | Loads generations, NLI matrices and labels; one row per question with `correct`, `wrong` and `correct_source` added. |
| `load_scores` | Every model's scores in one table. |
