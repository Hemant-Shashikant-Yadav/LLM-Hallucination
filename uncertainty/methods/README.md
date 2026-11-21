# uncertainty/methods/ — the 17 Objective 1 methods, one file each

Each numbered file holds one method, with a note beside it (`NN_name.README.md`, linked below). The
file's docstring and its note give the same content: what the method asks, the formula, every
symbol, a worked example (the running example of
[Formulas_in_Simple_Words.md](../presentation/Formulas_in_Simple_Words.md): 7 "Canberra" and 3
"Sydney" samples) and how to read the score. The numbers are the order of every report
(`registry.METHODS`).

A file name that starts with a digit cannot be imported with a plain `import`, so
[`__init__.py`](__init__.py) loads the numbered files and re-exports their functions:
`from uncertainty.methods import se_discrete`.

| # | File | Function | Family | Reference |
|---|---|---|---|---|
| 01 | [`01_semantic_entropy.py`](01_semantic_entropy.README.md) | `se_discrete` (and `se_normalized`) | meaning of the samples | Farquhar et al., 2024; Kuhn et al., 2023 |
| 02 | [`02_semantic_entropy_weighted.py`](02_semantic_entropy_weighted.README.md) | `se_weighted` | meaning of the samples | Kuhn et al., 2023 |
| 03 | [`03_semantic_entropy_pipeline.py`](03_semantic_entropy_pipeline.README.md) | `se_pipeline` | meaning of the samples | pipeline Layer 1 |
| 04 | [`04_number_of_meaning_groups.py`](04_number_of_meaning_groups.README.md) | `num_clusters` | meaning of the samples | Lin et al., 2024 |
| 05 | [`05_spectral_eigv.py`](05_spectral_eigv.README.md) | `eigv` | meaning of the samples (graph) | Lin et al., 2024 |
| 06 | [`06_spectral_degree.py`](06_spectral_degree.README.md) | `degree` | meaning of the samples (graph) | Lin et al., 2024 |
| 07 | [`07_spectral_eccentricity.py`](07_spectral_eccentricity.README.md) | `eccentricity` | meaning of the samples (graph) | Lin et al., 2024 |
| 08 | [`08_disagreement_with_greedy.py`](08_disagreement_with_greedy.README.md) | `disagree_greedy` | meaning of the samples | Wang et al., 2023 |
| 09 | [`09_selfcheck_nli.py`](09_selfcheck_nli.README.md) | `selfcheck_nli` | meaning of the samples | Manakul et al., 2023 |
| 10 | [`10_sequence_nll.py`](10_sequence_nll.README.md) | `nll` | word probabilities | Malinin & Gales, 2021 |
| 11 | [`11_length_normalised_nll.py`](11_length_normalised_nll.README.md) | `nll_norm` | word probabilities | Malinin & Gales, 2021 |
| 12 | [`12_token_entropy_mean.py`](12_token_entropy_mean.README.md) | `token_entropy_mean` | word probabilities | Malinin & Gales, 2021 |
| 13 | [`13_token_entropy_max.py`](13_token_entropy_max.README.md) | `token_entropy_max` | word probabilities | Malinin & Gales, 2021 |
| 14 | [`14_predictive_entropy.py`](14_predictive_entropy.README.md) | `predictive_entropy` | word probabilities | Kuhn et al., 2023 |
| 15 | [`15_lexical_similarity.py`](15_lexical_similarity.README.md) | `lexical_sim` | word overlap | Lin et al., 2024 |
| 16 | [`16_p_true.py`](16_p_true.README.md) | `p_true_unc` (and `p_true_confidence`) | self-evaluation | Kadavath et al., 2022 |
| 17 | [`17_verbalised_confidence.py`](17_verbalised_confidence.README.md) | `verbal_unc` (and `verbal_confidence`) | self-evaluation | Tian et al., 2023; Xiong et al., 2024 |

Other files here: `common.py` (shared helpers), `view.py` (one question's answers),
`nli.py` (meaning equivalence for free text), `registry.py` (the method table), each with its own
note too.

`tests/test_methods.py` checks the numbers against hand-worked examples, that each method has its
own numbered file in registry order, and that every file documents its formula and example and has
its note.
