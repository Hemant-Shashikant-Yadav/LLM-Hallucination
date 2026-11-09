# uncertainty/framework/features.py — Objective 2 features

Signals available at inference time (no gold data, no dataset name), computed in the scoring
pass.

| Part | What it does |
|---|---|
| `structure_features` | Share of the largest and second cluster, greedy in majority, mean log-prob of the top two clusters (confident-but-different suggests aleatoric), log-prob spread, premise agreement (shared names or intermediate numbers) and premise minus conclusion agreement (suggests reasoning). |
| `question_features` | Answer kind, the pipeline's task type, years and time words, hedges, refusals, premise corrections, question length. |
| `pipeline_task_type` | The pipeline's routing rule as a type: the Objective 2 baseline. |
