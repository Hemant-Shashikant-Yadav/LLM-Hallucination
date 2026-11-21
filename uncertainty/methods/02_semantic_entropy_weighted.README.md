# uncertainty/methods/02_semantic_entropy_weighted.py — Semantic entropy (likelihood-weighted)

Method 02 of 17. Part of [the 17 methods](README.md); the same text is the file's docstring.

| | |
|---|---|
| Family | meaning of the samples |
| Reference | Kuhn et al., 2023 [2]; Farquhar et al., 2024 [1] |
| Needs | the samples grouped by meaning, and each sample's token log-probabilities |
| Code | se_weighted. |

## What it asks

Like method 01, but answers the model itself found more likely count more.

## Formula

```
w(s)  = exp( (1/L_s) · Σ_i ln p(t_i) )          length-normalised probability of sample s
P(c)  = Σ_{s in c} w(s)  /  Σ_{all s} w(s)
SE_w  = − Σ_c  P(c) · ln P(c)
```

## Symbols

| Symbol | Meaning |
|---|---|
| `s` | one sample; L_s its number of tokens; t_i its i-th token |
| `p(t_i)` | probability the model gave to token t_i (Ollama reports ln p, the log-probability) |
| `w(s)` | the weight of sample s: its average token probability |
| `P(c)` | share of the total weight that falls in meaning group c |

## Worked example

```
each "Canberra" sample has average log-prob −0.1 -> w = e^−0.1 = 0.905;  7 of them -> 6.334
each "Sydney"   sample has average log-prob −0.7 -> w = e^−0.7 = 0.497;  3 of them -> 1.490
shares: 6.334 / 7.824 = 0.810 and 1.490 / 7.824 = 0.190
SE_w = −(0.810 · ln 0.810 + 0.190 · ln 0.190) = 0.171 + 0.316 = 0.487
```

## Reading the score

Same scale as method 01. Here it is lower than 0.611 because the model itself found "Sydney" unlikely. None without log-probs, with fewer than 2 samples or no meaning data.
