"""
Sampling-consistency scores.

- Disagreement with the greedy answer: share of samples that do not mean the same as the
  greedy answer (self-consistency, Wang et al., 2023).
- Lexical similarity: 1 − mean pairwise ROUGE-L F1 between samples (Lin et al., 2024),
  over each sample's first 64 words so long worked answers stay cheap.
- SelfCheckGPT-NLI (Manakul et al., 2023), free text only: mean probability that a sample
  contradicts the greedy answer.
"""

from __future__ import annotations

from itertools import combinations

from evaluation.truthfulqa_judge import rouge_l_f1
from uncertainty.methods.nli import CONTRA
from uncertainty.methods.semantic_entropy import has_meaning
from uncertainty.methods.view import View


def disagree_greedy(v: View) -> float | None:
    if v.m < 1 or not has_meaning(v):
        return None
    eq = v.equivalent()
    return float(1 - eq[0, 1:].sum() / v.m)


def lexical_sim(v: View) -> float | None:
    if v.m < 2:
        return None
    texts = [" ".join(s["text"].split()[:64]) for s in v.samples]
    scores = [rouge_l_f1(a, b) for a, b in combinations(texts, 2)]
    return float(1 - sum(scores) / len(scores))


def selfcheck_nli(v: View) -> float | None:
    if v.nli is None or v.m < 1:
        return None
    return float(v.nli[1:, 0, CONTRA].mean())
