# uncertainty/orchestrate/system.py — the machine and the server

Checks `run-all` makes before and between models.

| Part | What it does |
|---|---|
| `RunStop` | A problem that must stop the run, with what to do. |
| `memory_gb`, `machine_info` | Unified memory on a Mac (picks the roster); also Linux and Windows. |
| `ensure_pulled` | Pulls missing models; returns those that failed. |
| `ensure_only` | Unloads this run's other models; stops if a model the run does not use is loaded (another job). |
| `logprob_check` | Are log-probs sent for greedy and sampled tokens, and at T = 0 are they the model's probabilities or about 0 (reported after temperature)? Sets the mode the runner uses. |
