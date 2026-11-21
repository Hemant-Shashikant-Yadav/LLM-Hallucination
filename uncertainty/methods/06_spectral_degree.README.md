# uncertainty/methods/06_spectral_degree.py — Spectral Degree (average dissimilarity of the samples)

Method 06 of 17. Part of [the 17 methods](README.md); the same text is the file's docstring.

| | |
|---|---|
| Family | meaning of the samples (graph methods 05-07) |
| Reference | Lin, Trivedi & Sun, 2024 [3] |
| Needs | the samples' pairwise similarity (W, as in method 05) |
| Code | degree. |

## What it asks

On average, how unlike each other are the samples?

## Formula

```
D_jj   = Σ_k W_jk                      degree of sample j
Degree = 1 − mean_j(D_jj) / m
```

## Symbols

| Symbol | Meaning |
|---|---|
| `W_jk` | similarity of samples j and k (0-1; W_jj = 1) |
| `D_jj` | how strongly sample j is joined to all samples, itself included (at most m) |
| `m` | number of samples |

## Worked example

```
similarity 0.95 within a group, 0.02 between groups
a "Canberra" sample: 1 + 6 × 0.95 + 3 × 0.02 = 6.76
a "Sydney"   sample: 1 + 2 × 0.95 + 7 × 0.02 = 3.04
average degree = (7 × 6.76 + 3 × 3.04) / 10 = 5.644
Degree = 1 − 5.644 / 10 = 0.436

all agree -> 0.05;  5/5 -> 0.51;  5/3/2 -> 0.62;  all 10 differ -> 0.88
```

## Reading the score

0 = all samples alike (sure); close to 1 = all different (unsure). None with fewer than 2 samples or no meaning data.
