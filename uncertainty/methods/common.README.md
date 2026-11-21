# uncertainty/methods/common.py — helpers shared by the method files

None of these is a method itself.

| Part | What it does |
|---|---|
| `has_meaning` | The answers can be grouped by meaning (read answers or an NLI matrix); meaning-based methods return None without it. |
| `entropy` | −Σ p ln p over a list of shares (methods 01–03). |
| `mean_logprob` | Average token log-probability of one answer (methods 02, 11, 14; Objective 2 features). |
| `samples_similarity` | W, the samples' pairwise similarity, for the graph methods 05–07. |
| `normalised_laplacian` | L = I − D^(−1/2) W D^(−1/2) (methods 05 and 07). |
| `token_entropies` | Entropy at each token of the greedy answer over its renormalised top alternatives (methods 12–13). |
