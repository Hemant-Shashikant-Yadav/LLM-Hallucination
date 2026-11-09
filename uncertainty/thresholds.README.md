# uncertainty/thresholds.py — dev-tuned 'sure' and 'unsure' thresholds

Thresholds on normalised semantic entropy, tuned on each model's dev split only.

| Part | What it does |
|---|---|
| `tune` | `low`: the largest entropy below which dev answers are at least 90% right (errors below it are the blind spot); `high`: the smallest above which at least 50% are wrong. Defaults when fewer than 10 dev answers. |
| `tune_model`, `load` | Write and read `results/<model>/thresholds.json`. |
