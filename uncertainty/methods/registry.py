"""
The Objective 1 method table: name, family, function, and what it needs (PLAN §8.1).
Order here is the order in every report. Semantic entropy (discrete) is the primary method.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from uncertainty.methods import consistency, graph, self_eval, semantic_entropy, token_level
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


PRIMARY = "se_discrete"

METHODS: list[Method] = [
    Method("se_discrete", "Semantic entropy (discrete)", "meaning", semantic_entropy.se_discrete,
           "samples", "Farquhar et al., 2024"),
    Method("se_weighted", "Semantic entropy (weighted)", "meaning", semantic_entropy.se_weighted,
           "samples + log-probs", "Kuhn et al., 2023"),
    Method("se_pipeline", "Semantic entropy (pipeline Layer 1)", "meaning", semantic_entropy.se_pipeline,
           "samples", "Layer 1 of this project"),
    Method("num_clusters", "Number of meaning clusters", "meaning", semantic_entropy.num_clusters,
           "samples", "Lin et al., 2024"),
    Method("eigv", "Spectral EigV", "meaning", graph.eigv, "samples", "Lin et al., 2024"),
    Method("degree", "Spectral Degree", "meaning", graph.degree, "samples", "Lin et al., 2024"),
    Method("eccentricity", "Spectral Eccentricity", "meaning", graph.eccentricity, "samples", "Lin et al., 2024"),
    Method("disagree_greedy", "Disagreement with greedy", "meaning", consistency.disagree_greedy,
           "samples", "Wang et al., 2023"),
    Method("selfcheck_nli", "SelfCheckGPT-NLI", "meaning", consistency.selfcheck_nli,
           "samples (free text)", "Manakul et al., 2023"),
    Method("nll", "Sequence NLL", "token", token_level.nll, "log-probs", "Malinin & Gales, 2021"),
    Method("nll_norm", "Length-normalised NLL", "token", token_level.nll_norm, "log-probs",
           "Malinin & Gales, 2021"),
    Method("token_entropy_mean", "Mean token entropy", "token", token_level.token_entropy_mean,
           "top log-probs", "Malinin & Gales, 2021"),
    Method("token_entropy_max", "Max token entropy", "token", token_level.token_entropy_max,
           "top log-probs", "Malinin & Gales, 2021"),
    Method("pred_entropy", "Predictive entropy", "token", token_level.predictive_entropy,
           "samples + log-probs", "Kuhn et al., 2023"),
    Method("lexical_sim", "Lexical similarity", "surface", consistency.lexical_sim, "samples",
           "Lin et al., 2024"),
    Method("p_true", "P(True)", "self-eval", self_eval.p_true_unc, "1 extra call", "Kadavath et al., 2022",
           confidence=self_eval.p_true_confidence),
    Method("verbal", "Verbalised confidence", "self-eval", self_eval.verbal_unc, "1 extra call",
           "Tian et al., 2023", confidence=self_eval.verbal_confidence),
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
