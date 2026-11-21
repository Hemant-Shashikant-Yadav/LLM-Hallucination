"""
Method 03 of 17: Semantic entropy, pipeline Layer 1 variant.

Family      meaning of the samples
Reference   Layer 1 of this project's pipeline (app/layer1_triage/semantic_entropy.py), after [1], [2]
Needs       the greedy answer and the samples, grouped by meaning

What it asks
    The semantic entropy our pipeline uses: the greedy answer is counted with the samples, and the
    result is scaled to 0-1. Lets Objective 1 compare the pipeline's operating point with method 01.

Formula
    SE_pipe = min( 1,  [ − Σ_c (n_c / n) · ln(n_c / n) ]  /  ln n )

Symbols
    n      m + 1: the samples plus the greedy answer
    n_c    answers in group c, the greedy answer included
    ln n   the largest possible entropy of n answers, so the result runs from 0 to 1

Worked example
    greedy "Canberra" + 7 "Canberra" + 3 "Sydney"  ->  8/11 = 0.727 and 3/11 = 0.273
    entropy = −(0.727 · ln 0.727 + 0.273 · ln 0.273) = 0.232 + 0.354 = 0.586
    SE_pipe = 0.586 / ln 11 = 0.586 / 2.398 = 0.244

Reading the score
    0 = all answers agree, 1 = all differ. The study also reports it with the first 4 samples
    (`se_pipeline_k4` in score.py), the pipeline's real budget. None without meaning data.

Code: se_pipeline.
"""

from __future__ import annotations

import math
from collections import Counter

from uncertainty.methods.common import entropy, has_meaning
from uncertainty.methods.view import View


def se_pipeline(v: View) -> float | None:
    if v.m < 1 or not has_meaning(v):
        return None
    ids = v.clusters(with_greedy=True)
    counts = Counter(ids)
    return min(1.0, entropy([c / v.n for c in counts.values()]) / math.log(v.n))
