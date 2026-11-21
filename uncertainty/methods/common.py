"""
Helpers shared by the 17 method files (01_*.py … 17_*.py). None of these is a method itself.

Symbols used in the method files:
    m     number of valid samples (10 in the study; cut-off or empty samples are dropped)
    n     m + 1: the samples plus the greedy answer
    L     number of tokens in one answer
    ln    natural logarithm
"""

from __future__ import annotations

import math

import numpy as np

from uncertainty.methods.view import View


def has_meaning(v: View) -> bool:
    """True when the answers can be grouped by meaning: read answers (maths, options, yes/no) or an
    NLI matrix (free text). Meaning-based methods return None without it."""
    return v.keys is not None or v.nli is not None


def entropy(probs: list[float]) -> float:
    """H = − Σ p · ln p over the given shares (shares of 0 are skipped)."""
    return -sum(p * math.log(p) for p in probs if p > 0)


def mean_logprob(sample: dict) -> float | None:
    """Average token log-probability of one answer, (1/L) · Σ ln p(token); None without log-probs."""
    lps = sample.get("logprobs") or []
    return sum(lps) / len(lps) if lps else None


def samples_similarity(v: View) -> np.ndarray | None:
    """W, the samples' pairwise similarity in [0, 1] (graph methods 05-07): mean two-way NLI entailment
    for free text, 1/0 for matching/different read answers. None with fewer than 2 samples."""
    if v.m < 2 or not has_meaning(v):
        return None
    return v.similarity()[1:, 1:]


def normalised_laplacian(w: np.ndarray) -> np.ndarray:
    """L = I − D^(−1/2) · W · D^(−1/2), where D is the diagonal matrix of degrees D_jj = Σ_k W_jk."""
    d = w.sum(axis=1)
    inv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return np.eye(len(w)) - (inv[:, None] * w * inv[None, :])


def token_entropies(v: View) -> list[float]:
    """Entropy at each token of the greedy answer (methods 12-13): over the top alternatives Ollama
    reports, renormalised to sum to 1. A lower bound on the full entropy, since only the top few are seen."""
    out = []
    for alts in v.greedy.get("top_logprobs") or []:
        ps = [math.exp(lp) for _, lp in alts]
        total = sum(ps)
        if total > 0:
            out.append(entropy([p / total for p in ps]))
    return out
