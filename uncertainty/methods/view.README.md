# uncertainty/methods/view.py — one question's answers, ready for the methods

Meaning equivalence follows Layer 1: worked answers by their read answer, free text by
two-way NLI entailment. Clusters are formed greedily (Kuhn et al., 2023).

| Part | What it does |
|---|---|
| `View` | Kind, question, greedy answer, valid samples, read keys or NLI matrix, P(True), verbalised confidence. |
| `equivalent` | (n × n) same meaning; an unreadable answer is only equal to itself. |
| `similarity` | Soft similarity for the spectral methods: mean two-way entailment, or 0/1. |
| `clusters` | Cluster id of each sample (or of greedy + samples, for the pipeline variant). |
| `subset(k)` | The same question with the first k samples (AUROC-vs-K, lite features). |
| `view_from_record` | Builds a View from a stored record. |
