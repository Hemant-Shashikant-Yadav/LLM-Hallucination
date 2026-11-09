# uncertainty/orchestrate/gates.py — checks that stop a bad run early

Readability of at least 90% of worked samples, at most 10% of samples cut off, at most 10%
empty greedy answers, a judge that answers (see `check_judge`).

| Part | What it does |
|---|---|
| `generation_stats`, `check_generation` | On the pilot (stops the run) and on each model's first 20 questions per set (skips that model). |
| `check_judge` | Above 2% judge failures: a recorded warning (those answers stay unlabelled). Above 20%: the judge is broken and the gate fails. One failure in a small batch must not throw away a finished model. |
