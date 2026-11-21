"""
Method 17 of 17: Verbalised (stated) confidence.

Family      self-evaluation
Reference   Tian et al., 2023 [6]; Xiong et al., 2024 [7]
Needs       one extra call: "How confident are you that the proposed answer is correct? Reply with a
            number from 0 to 100." Asked in generate/runner.py; values are clipped to 0-100.

What it asks
    How sure does the model say it is?

Formula
    confidence  = number / 100
    uncertainty = 1 − confidence

Symbols
    number   the 0-100 confidence the model states

Worked example
    the model says 85  ->  confidence = 0.85  ->  uncertainty = 0.15

Reading the score
    0 = "100% sure"; 1 = "0% sure". Easy to read, but small models often say 90 even when wrong,
    so it ranks errors poorly. The confidence is also checked for calibration (ECE, Brier).
    None without the extra call.

Code: verbal_unc (the score), verbal_confidence (for calibration).
"""

from __future__ import annotations

from uncertainty.methods.view import View


def verbal_confidence(v: View) -> float | None:
    return None if v.verbal is None else v.verbal / 100.0


def verbal_unc(v: View) -> float | None:
    return None if v.verbal is None else 1.0 - v.verbal / 100.0
