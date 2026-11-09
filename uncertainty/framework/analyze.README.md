# uncertainty/framework/analyze.py — the tool

One question and a model in, an uncertainty report out (PLAN §9.6).

| Part | What it does |
|---|---|
| `infer_kind` | Options → choice; yes/no question; maths → number; else short. |
| `load_bundles` | The trained detector and classifier for a mode, with a clear error if training has not run. |
| `analyze` | Generates the record (full: K = 10 + self-evaluation + probes; lite: K = 4 only), computes NLI and features, applies both models, flags overconfidence (sure samples, doubtful self-check), and returns answer, identify, classify, signals, recommended action and cost. |
