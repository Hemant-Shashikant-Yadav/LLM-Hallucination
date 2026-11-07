# app/layer1_triage/semantic_entropy.py — Layer 1: is the model sure?

Decides whether the draft can be trusted as is. It samples the model a few more times and
measures how much the answers **disagree in meaning** (semantic entropy, Kuhn et al. 2023;
Farquhar et al. 2024). Agreement → the fast path; disagreement → verification. The samples
are kept and reused by Layer 3 and the votes, so no work is wasted.

Entropy formula: `H_sem = −Σ p(c)·log p(c) / log(N)` over meaning clusters `c` of the
N answers (draft + samples), scaled to 0 (all agree) … 1 (all differ).

| Lines | What it does |
|---|---|
| 1–18 | Docstring: sample, cluster by meaning, compute entropy, decide. |
| 48–50 | **`math_sample_tokens`**: token budget for a maths sample: 1,536 for advanced maths, else 1,024. Cut-off samples can't vote, so short budgets silenced hard questions. |
| 53–71 | `SemanticEntropyEstimate`: entropy, cluster of each answer, the answers (draft first), the read answers; `draft_in_majority` = the draft is in the largest cluster. |
| 77–97 | Constructor: generator client, NLI model (for meaning clusters), K = 4 samples, temperature 0.7, `t_safe`. |
| 99–118 | **`triage`, sampling**: decides whether this is maths; draws K samples (maths and multiple choice get the bigger budget, and cut-off ones are dropped). |
| 120–132 | **Clustering**, by question kind: multiple choice → by the option picked; maths → by the canonical final answer (`0.5` = `1/2`); yes/no question whose draft and most samples say yes/no → by verdict; anything else → by meaning, with NLI. |
| 134–141 | Entropy and the estimate. |
| 143–152 | Threshold: `t_safe_reasoning` (0.2 in `.env`) for maths, `t_safe` (0.35) otherwise. Fewer than 2 successful samples → entropy is meaningless (draft alone = 0), so the fast path is blocked and a degradation is recorded. |
| 153–165 | **Decision**, in order: time_sensitive → always Layer 2 (agreeing samples can all be out of date); advanced maths → always Layer 3 (all samples can share one mistake); enough samples, entropy below threshold, draft in the majority → **SAFE**; maths → Layer 3; otherwise → Layer 2. |
| 167–197 | Builds the `TriageResult` (entropy doubles as hallucination probability) and logs decision, entropy, samples OK, cluster count, draft-in-majority. |
| 203–207 | `sample`: public entry for more samples (Layer 3 and votes), always dropping cut-off ones. |
| 209–233 | `_sample`: N samples in parallel, each with a fixed seed (`seed_base + i`) so runs repeat; a failed sample is recorded as a degradation and skipped. |
| 239–245 | `_cluster_numeric`: cluster by canonical final answer. |
| 247–253 | `_yes_no_readable`: use verdict clustering only if the draft and most answers say yes/no outright. |
| 255–265 | `_cluster_keys`: same key → same cluster; an unreadable answer is its own cluster. |
| 267–298 | **`_cluster_semantic`**: meaning clusters via NLI. Each answer (question + its first sentences) joins the first cluster whose representative it **entails both ways**, else starts a new cluster. Without the NLI model every answer is its own cluster (entropy 1, always verified) and a degradation is recorded. |
| 300–302 | `_entails`: entailment probability is the highest of the three labels. |
| 304–311 | `_discrete_entropy`: the formula above, scaled to [0, 1]. |
