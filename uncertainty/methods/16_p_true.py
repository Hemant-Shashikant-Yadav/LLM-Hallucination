"""
Method 16 of 17: P(True), the model judges its own answer.

Family      self-evaluation
Reference   Kadavath et al., 2022 [5]
Needs       one extra call: the question, the samples as "brainstormed ideas", and the greedy answer,
            with the instruction "Is the possible answer: (A) True (B) False". Asked in generate/runner.py.

What it asks
    Shown its own samples and answer, does the model say "True"?

Formula
    P(True)     = P("A") / ( P("A") + P("B") )      read from the first token of the reply
    uncertainty = 1 − P(True)

Symbols
    P("A"), P("B")   the probabilities of the first reply token being "A" (True) or "B" (False)

Worked example
    P("A") = 0.72 and P("B") = 0.18  ->  P(True) = 0.72 / 0.90 = 0.80  ->  uncertainty = 0.20

Reading the score
    0 = the model is sure its answer is true; 1 = sure it is false. P(True) itself is a
    probability, so it is also checked for calibration (ECE, Brier). None without the extra call.

Code: p_true_unc (the score), p_true_confidence (for calibration).
"""

from __future__ import annotations

from uncertainty.methods.view import View


def p_true_confidence(v: View) -> float | None:
    return v.p_true


def p_true_unc(v: View) -> float | None:
    return None if v.p_true is None else 1.0 - v.p_true
