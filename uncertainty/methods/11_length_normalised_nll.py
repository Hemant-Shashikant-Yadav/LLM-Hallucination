"""
Method 11 of 17: Length-normalised NLL (per-token negative log-likelihood).

Family      the model's own word probabilities (token-level)
Reference   Malinin & Gales, 2021 [4]
Needs       the token log-probabilities of the greedy answer

What it asks
    The same as method 10, but per token, so long answers are not punished.

Formula
    NLL_norm = − (1/L) · Σ_i  ln p(t_i)  =  NLL / L

Symbols
    L        number of tokens in the greedy answer
    p(t_i)   probability of the i-th token

Worked example
    NLL = 0.167 over 3 tokens  ->  NLL_norm = 0.167 / 3 = 0.056

Reading the score
    0 = fully sure of every token; higher = less likely per token. None without log-probs.

Code: nll_norm.
"""

from __future__ import annotations

from uncertainty.methods.common import mean_logprob
from uncertainty.methods.view import View


def nll_norm(v: View) -> float | None:
    lp = mean_logprob(v.greedy)
    return None if lp is None else -lp
