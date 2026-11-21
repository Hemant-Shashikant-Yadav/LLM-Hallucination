# uncertainty/methods/09_selfcheck_nli.py — SelfCheckGPT-NLI

Method 09 of 17. Part of [the 17 methods](README.md); the same text is the file's docstring.

| | |
|---|---|
| Family | meaning of the samples |
| Reference | Manakul, Liusie & Gales, 2023 [8] |
| Needs | free-text answers and the NLI matrix (DeBERTa-v3); not used for maths, options or yes/no |
| Code | selfcheck_nli. |

## What it asks

On average, how strongly does each sample contradict the greedy answer?

## Formula

```
SelfCheck = (1/m) · Σ_j  P(contradiction : premise = sample j, hypothesis = greedy answer)
```

## Symbols

| Symbol | Meaning |
|---|---|
| `P(contradiction : …)` | the NLI model's probability that sample j contradicts the greedy answer |
| `m` | number of samples |

## Worked example

```
7 "Canberra" samples contradict the greedy "Canberra" with 0.02; 3 "Sydney" samples with 0.95
SelfCheck = (7 × 0.02 + 3 × 0.95) / 10 = (0.14 + 2.85) / 10 = 0.299
```

## Reading the score

Close to 0 = the samples support the answer; close to 1 = they contradict it. None for worked answers (no NLI matrix) or without samples.
