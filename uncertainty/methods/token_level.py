"""
Token-level uncertainty from the model's own probabilities (needs log-probs from the server).

- NLL and length-normalised NLL of the greedy answer (Malinin & Gales, 2021).
- Mean and max token entropy of the greedy answer, over the top-k alternatives renormalised
  (a lower bound on the full entropy, since only the top k are reported).
- Predictive entropy: Monte Carlo estimate −mean_s (1/L_s) ln p(s) over the samples
  (Kuhn et al., 2023, length-normalised).
"""

from __future__ import annotations

import math

from uncertainty.methods.semantic_entropy import mean_logprob
from uncertainty.methods.view import View


def nll(v: View) -> float | None:
    lps = v.greedy.get("logprobs") or []
    return -sum(lps) if lps else None


def nll_norm(v: View) -> float | None:
    lp = mean_logprob(v.greedy)
    return None if lp is None else -lp


def _token_entropies(v: View) -> list[float]:
    out = []
    for alts in v.greedy.get("top_logprobs") or []:
        ps = [math.exp(lp) for _, lp in alts]
        total = sum(ps)
        if total > 0:
            out.append(-sum((p / total) * math.log(p / total) for p in ps if p > 0))
    return out


def token_entropy_mean(v: View) -> float | None:
    h = _token_entropies(v)
    return sum(h) / len(h) if h else None


def token_entropy_max(v: View) -> float | None:
    h = _token_entropies(v)
    return max(h) if h else None


def predictive_entropy(v: View) -> float | None:
    lps = [mean_logprob(s) for s in v.samples]
    if not lps or any(lp is None for lp in lps):
        return None
    return -sum(lps) / len(lps)
