"""
Semantic entropy: uncertainty over meanings rather than wordings.

- Discrete SE (Farquhar et al., 2024, the primary method of Objective 1): entropy of the
  cluster frequencies of the K samples, H = −Σ_c p(c) ln p(c).
- Likelihood-weighted SE (Kuhn et al., 2023; Farquhar et al., 2024): cluster probability is
  the summed length-normalised sequence probability of its members.
- Pipeline SE: the variant Layer 1 uses (greedy + samples, scaled to [0, 1] by ln n), for
  comparison with the pipeline's operating point.
- Number of meaning clusters (Lin et al., 2024).
"""

from __future__ import annotations

import math
from collections import Counter

from uncertainty.methods.view import View


def has_meaning(v: View) -> bool:
    return v.keys is not None or v.nli is not None


def _entropy(probs: list[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


def se_discrete(v: View) -> float | None:
    if v.m < 2 or not has_meaning(v):
        return None
    counts = Counter(v.clusters())
    return _entropy([c / v.m for c in counts.values()])


def se_normalized(v: View) -> float | None:
    """Discrete SE scaled to [0, 1] by ln K, comparable across sample counts (used for thresholds)."""
    se = se_discrete(v)
    return None if se is None else se / math.log(v.m)


def se_pipeline(v: View) -> float | None:
    if v.m < 1 or not has_meaning(v):
        return None
    ids = v.clusters(with_greedy=True)
    counts = Counter(ids)
    return min(1.0, _entropy([c / v.n for c in counts.values()]) / math.log(v.n))


def mean_logprob(sample: dict) -> float | None:
    lps = sample.get("logprobs") or []
    return sum(lps) / len(lps) if lps else None


def se_weighted(v: View) -> float | None:
    if v.m < 2 or not has_meaning(v):
        return None
    lps = [mean_logprob(s) for s in v.samples]
    if any(lp is None for lp in lps):
        return None
    weights = [math.exp(lp) for lp in lps]
    mass: dict[int, float] = {}
    for c, w in zip(v.clusters(), weights):
        mass[c] = mass.get(c, 0.0) + w
    total = sum(mass.values())
    return _entropy([m / total for m in mass.values()]) if total > 0 else None


def num_clusters(v: View) -> float | None:
    if v.m < 1 or not has_meaning(v):
        return None
    return float(len(set(v.clusters())))
