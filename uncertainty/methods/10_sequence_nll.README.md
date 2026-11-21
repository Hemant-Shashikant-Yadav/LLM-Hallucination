# uncertainty/methods/10_sequence_nll.py — Sequence NLL (negative log-likelihood of the greedy answer)

Method 10 of 17. Part of [the 17 methods](README.md); the same text is the file's docstring.

| | |
|---|---|
| Family | the model's own word probabilities (token-level) |
| Reference | Malinin & Gales, 2021 [4] |
| Needs | the token log-probabilities of the greedy answer (from the Ollama server) |
| Code | nll. |

## What it asks

How unlikely was the whole greedy answer, for the model itself?

## Formula

```
NLL = − Σ_i  ln p(t_i)            over every token t_1 … t_L of the greedy answer
```

## Symbols

| Symbol | Meaning |
|---|---|
| `t_i` | the i-th token (piece of a word) of the greedy answer |
| `p(t_i)` | the probability the model gave to that token; Ollama reports ln p(t_i) |
| `L` | number of tokens |

## Worked example

```
"Canberra." = 3 tokens with probabilities 0.90, 0.95, 0.99
NLL = −(ln 0.90 + ln 0.95 + ln 0.99) = 0.105 + 0.051 + 0.010 = 0.167
```

## Reading the score

0 = the model was fully sure of every token; higher = less likely words. Long answers always score higher, just for having more tokens (method 11 corrects this). It measures doubt about the wording, not the meaning. None without log-probs.
