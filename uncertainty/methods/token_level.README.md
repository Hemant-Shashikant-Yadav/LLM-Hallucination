# uncertainty/methods/token_level.py — token-level uncertainty

From the model's own probabilities; None when the server sends no log-probs.

| Part | What it does |
|---|---|
| `nll`, `nll_norm` | Negative log-likelihood of the greedy answer, total and per token (Malinin & Gales, 2021). |
| `token_entropy_mean`, `token_entropy_max` | Entropy over the top-5 alternatives per token (a lower bound). |
| `predictive_entropy` | Mean per-token negative log-likelihood of the samples (Kuhn et al., 2023). |
