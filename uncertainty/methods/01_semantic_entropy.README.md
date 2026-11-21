# uncertainty/methods/01_semantic_entropy.py — Semantic entropy (discrete)

Method 01 of 17. The primary method of Objective 1. Part of [the 17 methods](README.md); the same text is the file's docstring.

| | |
|---|---|
| Family | meaning of the samples |
| Reference | Farquhar et al., 2024 (Nature) [1]; Kuhn et al., 2023 [2] |
| Needs | the samples, grouped by meaning (read answers, or two-way NLI entailment for free text) |
| Code | se_discrete, se_normalized. Import with `from uncertainty.methods import se_discrete`. |

## What it asks

If we ask the same question m times, how spread out are the meanings of the answers?

## Formula

```
p(c)    = n_c / m
SE      = − Σ_c  p(c) · ln p(c)
SE_norm = SE / ln m              (SE scaled to 0-1; used for the "sure" thresholds and the
                                  blind spot, and by Objective 2; not one of the 17)
```

## Symbols

| Symbol | Meaning |
|---|---|
| `m` | number of valid samples (10 in the study) |
| `c` | one meaning group (cluster) of samples |
| `n_c` | number of samples in group c |
| `p(c)` | share of the samples in group c |
| `Σ_c` | add up over every group |
| `ln` | natural logarithm; ln of a share is negative, so the minus sign makes SE positive |

## Worked example (the running example of presentation/Formulas_in_Simple_Words.md)

```
10 samples: 7 "Canberra", 3 "Sydney"  ->  p = 0.7 and 0.3
SE      = −(0.7 · ln 0.7 + 0.3 · ln 0.3) = −(0.7 · (−0.357) + 0.3 · (−1.204)) = 0.250 + 0.361 = 0.611
SE_norm = 0.611 / ln 10 = 0.611 / 2.303 = 0.265

all 10 agree -> 0;  5/5 -> 0.693;  5/3/2 -> 1.030;  all 10 differ -> ln 10 = 2.303 (the maximum)
```

## Reading the score

0 = every sample means the same (sure). Higher = the meanings spread (unsure, more likely a hallucination). None with fewer than 2 samples or no meaning data.
