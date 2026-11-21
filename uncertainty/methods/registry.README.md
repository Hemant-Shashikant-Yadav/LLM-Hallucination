# uncertainty/methods/registry.py — the method table

Every Objective 1 method with its label, family, what it needs and its reference; the order
of every report.

| Part | What it does |
|---|---|
| `Method`, `METHODS` | 17 methods, in the order of their files `01_…py` to `17_…py`; `confidence` marks the probability-valued ones (calibration); `Method.file` names the method's own file. |
| `PRIMARY` | `se_discrete`. |
| `score_view` | Every method on one View; a method that fails on one odd answer gives None there (first failures logged), so a long run never stops on it. |
