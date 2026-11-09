# uncertainty/methods/semantic_entropy.py — semantic entropy

The primary method of Objective 1 and its variants.

| Part | What it does |
|---|---|
| `se_discrete` | Farquhar et al. (2024): entropy of the samples' cluster frequencies. |
| `se_normalized` | The same scaled to [0, 1] by ln K; used for thresholds. |
| `se_pipeline` | Layer 1's variant (greedy + samples, scaled by ln n), to compare with the pipeline's operating point. |
| `se_weighted` | Kuhn et al. (2023): clusters weighted by length-normalised sequence probability. |
| `num_clusters` | Number of meanings (Lin et al., 2024). |
| `has_meaning`, `mean_logprob` | Helpers: equivalence data present; mean token log-prob of an answer. |
