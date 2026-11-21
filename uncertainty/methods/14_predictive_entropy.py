"""
Method 14 of 17: Predictive entropy (length-normalised, estimated from the samples).

Family      the model's own word probabilities (token-level)
Reference   Kuhn et al., 2023 [2]
Needs       the token log-probabilities of every sample

What it asks
    On average, how unlikely were the sampled answers?

Formula
    PE = − (1/m) · Σ_s  (1/L_s) · Σ_i  ln p(t_i^s)

Symbols
    m          number of samples
    s          one sample;  L_s its number of tokens;  t_i^s its i-th token
    (1/L_s) Σ ln p(t_i^s)   the sample's average token log-probability

Worked example
    each "Canberra" sample: average log-prob −0.1;  each "Sydney" sample: −0.7
    PE = −(7 × (−0.1) + 3 × (−0.7)) / 10 = (0.7 + 2.1) / 10 = 0.28

Reading the score
    Higher = the samples were less likely. It mixes up wording with doubt: "Canberra" and "The
    capital is Canberra" get different probabilities but mean the same. Semantic entropy (method 01)
    groups by meaning to fix this. None without log-probs.

Code: predictive_entropy.
"""

from __future__ import annotations

from uncertainty.methods.common import mean_logprob
from uncertainty.methods.view import View


def predictive_entropy(v: View) -> float | None:
    lps = [mean_logprob(s) for s in v.samples]
    if not lps or any(lp is None for lp in lps):
        return None
    return -sum(lps) / len(lps)
