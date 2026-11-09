# uncertainty/methods/self_eval.py — self-evaluation

The model judges its own answer.

| Part | What it does |
|---|---|
| `p_true_confidence`, `p_true_unc` | P(True) (Kadavath et al., 2022) and 1 − P(True). |
| `verbal_confidence`, `verbal_unc` | Stated confidence / 100 (Tian et al., 2023) and 1 − it. |
