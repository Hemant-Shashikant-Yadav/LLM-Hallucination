"""
Objective 1: the 17 published uncertainty methods, one file each, as pure functions over stored
generations.

01_semantic_entropy.py … 17_verbalised_confidence.py hold one method each, in report order. Each
file's docstring gives the formula, every symbol, a worked example and how to read the score.
`common.py` has the helpers they share; `view.View` holds one question's greedy answer, valid
samples and meaning-equivalence data; `registry.METHODS` lists the methods with their labels and
references.

Every method maps a View to a score where **higher means more uncertain**, or None when it cannot
run (no log-probs, too few samples). A file name that starts with a digit cannot be imported with
a plain `import` statement, so this package loads the numbered files and re-exports their
functions:

    from uncertainty.methods import se_discrete, se_normalized
"""

from __future__ import annotations

from importlib import import_module

# file (without .py) -> the functions it defines, in method order
METHOD_FILES: dict[str, tuple[str, ...]] = {
    "01_semantic_entropy": ("se_discrete", "se_normalized"),
    "02_semantic_entropy_weighted": ("se_weighted",),
    "03_semantic_entropy_pipeline": ("se_pipeline",),
    "04_number_of_meaning_groups": ("num_clusters",),
    "05_spectral_eigv": ("eigv",),
    "06_spectral_degree": ("degree",),
    "07_spectral_eccentricity": ("eccentricity",),
    "08_disagreement_with_greedy": ("disagree_greedy",),
    "09_selfcheck_nli": ("selfcheck_nli",),
    "10_sequence_nll": ("nll",),
    "11_length_normalised_nll": ("nll_norm",),
    "12_token_entropy_mean": ("token_entropy_mean",),
    "13_token_entropy_max": ("token_entropy_max",),
    "14_predictive_entropy": ("predictive_entropy",),
    "15_lexical_similarity": ("lexical_sim",),
    "16_p_true": ("p_true_unc", "p_true_confidence"),
    "17_verbalised_confidence": ("verbal_unc", "verbal_confidence"),
}

for _file, _names in METHOD_FILES.items():
    _module = import_module(f"{__name__}.{_file}")
    for _name in _names:
        globals()[_name] = getattr(_module, _name)

from uncertainty.methods.common import has_meaning, mean_logprob  # noqa: E402

__all__ = ["METHOD_FILES", "has_meaning", "mean_logprob"] + [n for names in METHOD_FILES.values() for n in names]
