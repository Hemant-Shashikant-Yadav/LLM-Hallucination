# uncertainty/methods/17_verbalised_confidence.py — Verbalised (stated) confidence

Method 17 of 17. Part of [the 17 methods](README.md); the same text is the file's docstring.

| | |
|---|---|
| Family | self-evaluation |
| Reference | Tian et al., 2023 [6]; Xiong et al., 2024 [7] |
| Needs | one extra call: "How confident are you that the proposed answer is correct? Reply with a number from 0 to 100." Asked in generate/runner.py; values are clipped to 0-100. |
| Code | verbal_unc (the score), verbal_confidence (for calibration). |

## What it asks

How sure does the model say it is?

## Formula

```
confidence  = number / 100
uncertainty = 1 − confidence
```

## Symbols

| Symbol | Meaning |
|---|---|
| `number` | the 0-100 confidence the model states |

## Worked example

```
the model says 85  ->  confidence = 0.85  ->  uncertainty = 0.15
```

## Reading the score

0 = "100% sure"; 1 = "0% sure". Easy to read, but small models often say 90 even when wrong, so it ranks errors poorly. The confidence is also checked for calibration (ECE, Brier). None without the extra call.
