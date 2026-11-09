# uncertainty/methods/consistency.py — sampling consistency

Scores from agreement between answers.

| Part | What it does |
|---|---|
| `disagree_greedy` | Share of samples that do not mean what the greedy answer means. |
| `lexical_sim` | One minus mean pairwise ROUGE-L of the samples' first 64 words (ROUGE-L from `evaluation.truthfulqa_judge`, read-only). |
| `selfcheck_nli` | SelfCheckGPT-NLI (Manakul et al., 2023), free text only: mean probability a sample contradicts the greedy answer. |
