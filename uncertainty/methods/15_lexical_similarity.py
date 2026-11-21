"""
Method 15 of 17: Lexical similarity (1 − mean ROUGE-L between the samples).

Family      word overlap (surface)
Reference   Lin, Trivedi & Sun, 2024 [3]
Needs       the sample texts only (no model, no meaning data)

What it asks
    Do the samples use the same words, in the same order?

Formula
    LCS(a, b)  = longest common subsequence: the most words a and b share in the same order
    P = LCS / |b|,   R = LCS / |a|,   F1(a, b) = 2 · P · R / (P + R)
    LexSim     = 1 − mean over all sample pairs (a, b) of F1(a, b)
    (each sample cut to its first 64 words, so long worked answers stay cheap)

Symbols
    |a|, |b|   number of words in samples a and b
    P, R       precision and recall of the shared words
    F1         one overlap score from 0 (nothing shared) to 1 (identical)

Worked example
    10 samples -> 45 pairs. Same-word pairs: Canberra-Canberra 21 + Sydney-Sydney 3 = 24, each F1 = 1;
    the other 21 pairs share nothing (F1 = 0).
    mean F1 = 24 / 45 = 0.533  ->  LexSim = 1 − 0.533 = 0.467

    The trap: "the capital is Canberra" vs "Canberra is the capital" share only "the capital" in
    order, so F1 = 0.5 although the meaning is identical.

Reading the score
    0 = identical wording; higher = different wording. It misses paraphrases. None with fewer than
    2 samples. ROUGE-L comes from evaluation.truthfulqa_judge (read-only import).

Code: lexical_sim.
"""

from __future__ import annotations

from itertools import combinations

from evaluation.truthfulqa_judge import rouge_l_f1
from uncertainty.methods.view import View


def lexical_sim(v: View) -> float | None:
    if v.m < 2:
        return None
    texts = [" ".join(s["text"].split()[:64]) for s in v.samples]
    scores = [rouge_l_f1(a, b) for a, b in combinations(texts, 2)]
    return float(1 - sum(scores) / len(scores))
