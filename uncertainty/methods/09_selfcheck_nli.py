"""
Method 09 of 17: SelfCheckGPT-NLI.

Family      meaning of the samples
Reference   Manakul, Liusie & Gales, 2023 [8]
Needs       free-text answers and the NLI matrix (DeBERTa-v3); not used for maths, options or yes/no

What it asks
    On average, how strongly does each sample contradict the greedy answer?

Formula
    SelfCheck = (1/m) · Σ_j  P(contradiction : premise = sample j, hypothesis = greedy answer)

Symbols
    P(contradiction : …)   the NLI model's probability that sample j contradicts the greedy answer
    m                      number of samples

Worked example
    7 "Canberra" samples contradict the greedy "Canberra" with 0.02; 3 "Sydney" samples with 0.95
    SelfCheck = (7 × 0.02 + 3 × 0.95) / 10 = (0.14 + 2.85) / 10 = 0.299

Reading the score
    Close to 0 = the samples support the answer; close to 1 = they contradict it. None for worked
    answers (no NLI matrix) or without samples.

Code: selfcheck_nli.
"""

from __future__ import annotations

from uncertainty.methods.nli import CONTRA
from uncertainty.methods.view import View


def selfcheck_nli(v: View) -> float | None:
    if v.nli is None or v.m < 1:
        return None
    return float(v.nli[1:, 0, CONTRA].mean())
