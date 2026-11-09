"""
Self-evaluation: the model judges its own answer.

- P(True) (Kadavath et al., 2022): probability the model gives "A (True)" when shown the
  question, its samples as brainstormed ideas, and its greedy answer.
- Verbalised confidence (Tian et al., 2023; Xiong et al., 2024): the 0-100 number it states.

Both are confidences; the uncertainty score is 1 − confidence. The confidences themselves
are used for calibration (ECE, Brier).
"""

from __future__ import annotations

from uncertainty.methods.view import View


def p_true_confidence(v: View) -> float | None:
    return v.p_true


def verbal_confidence(v: View) -> float | None:
    return None if v.verbal is None else v.verbal / 100.0


def p_true_unc(v: View) -> float | None:
    return None if v.p_true is None else 1.0 - v.p_true


def verbal_unc(v: View) -> float | None:
    return None if v.verbal is None else 1.0 - v.verbal / 100.0
