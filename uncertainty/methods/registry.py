"""
The Objective 1 method table: name, family, function, and what it needs (PLAN §8.1).
Order here is the order in every report and matches the file numbers 01-17 in this folder.
Semantic entropy (discrete) is the primary method.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from uncertainty.methods import (degree, disagree_greedy, eccentricity, eigv, lexical_sim, nll, nll_norm,
                                 num_clusters, p_true_confidence, p_true_unc, predictive_entropy, se_discrete,
                                 se_pipeline, se_weighted, selfcheck_nli, token_entropy_max, token_entropy_mean,
                                 verbal_confidence, verbal_unc)
from uncertainty.methods.view import View


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    family: str
    fn: Callable[[View], float | None]
    needs: str
    reference: str
    confidence: Callable[[View], float | None] | None = None  # probability-valued: used for calibration

    @property
    def file(self) -> str:
        """The method's own file in uncertainty/methods/, e.g. '01_semantic_entropy'."""
        return self.fn.__module__.rsplit(".", 1)[-1]


PRIMARY = "se_discrete"

METHODS: list[Method] = [
    Method("se_discrete", "Semantic entropy (discrete)", "meaning", se_discrete, "samples", "Farquhar et al., 2024"),
    Method("se_weighted", "Semantic entropy (weighted)", "meaning", se_weighted, "samples + log-probs",
           "Kuhn et al., 2023"),
    Method("se_pipeline", "Semantic entropy (pipeline Layer 1)", "meaning", se_pipeline, "samples",
           "Layer 1 of this project"),
    Method("num_clusters", "Number of meaning clusters", "meaning", num_clusters, "samples", "Lin et al., 2024"),
    Method("eigv", "Spectral EigV", "meaning", eigv, "samples", "Lin et al., 2024"),
    Method("degree", "Spectral Degree", "meaning", degree, "samples", "Lin et al., 2024"),
    Method("eccentricity", "Spectral Eccentricity", "meaning", eccentricity, "samples", "Lin et al., 2024"),
    Method("disagree_greedy", "Disagreement with greedy", "meaning", disagree_greedy, "samples",
           "Wang et al., 2023"),
    Method("selfcheck_nli", "SelfCheckGPT-NLI", "meaning", selfcheck_nli, "samples (free text)",
           "Manakul et al., 2023"),
    Method("nll", "Sequence NLL", "token", nll, "log-probs", "Malinin & Gales, 2021"),
    Method("nll_norm", "Length-normalised NLL", "token", nll_norm, "log-probs", "Malinin & Gales, 2021"),
    Method("token_entropy_mean", "Mean token entropy", "token", token_entropy_mean, "top log-probs",
           "Malinin & Gales, 2021"),
    Method("token_entropy_max", "Max token entropy", "token", token_entropy_max, "top log-probs",
           "Malinin & Gales, 2021"),
    Method("pred_entropy", "Predictive entropy", "token", predictive_entropy, "samples + log-probs",
           "Kuhn et al., 2023"),
    Method("lexical_sim", "Lexical similarity", "surface", lexical_sim, "samples", "Lin et al., 2024"),
    Method("p_true", "P(True)", "self-eval", p_true_unc, "1 extra call", "Kadavath et al., 2022",
           confidence=p_true_confidence),
    Method("verbal", "Verbalised confidence", "self-eval", verbal_unc, "1 extra call", "Tian et al., 2023",
           confidence=verbal_confidence),
]

BY_NAME = {m.name: m for m in METHODS}
SAMPLE_METHODS = [m.name for m in METHODS if "samples" in m.needs]


_failures: dict[str, int] = {}


def score_view(v: View) -> dict[str, float | None]:
    """Every method's score. One odd answer must not stop a long run: a failing method gives
    None for that question, and its first few failures are logged."""
    out = {}
    for m in METHODS:
        try:
            out[m.name] = m.fn(v)
        except Exception as exc:
            _failures[m.name] = _failures.get(m.name, 0) + 1
            if _failures[m.name] <= 3:
                logger.warning("Method {} failed on a question: {}: {}", m.name, type(exc).__name__, exc)
            out[m.name] = None
    return out
