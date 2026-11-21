# uncertainty/methods/12_token_entropy_mean.py — Mean token entropy

Method 12 of 17. Part of [the 17 methods](README.md); the same text is the file's docstring.

| | |
|---|---|
| Family | the model's own word probabilities (token-level) |
| Reference | Malinin & Gales, 2021 [4] |
| Needs | the top alternative tokens at each position of the greedy answer (Ollama's top log-probs) |
| Code | token_entropy_mean (per-position entropies: common.token_entropies). |

## What it asks

At each word, how torn was the model between its top choices? On average over the answer.

## Formula

```
q_v  = p(v) / Σ_u p(u)                 the top alternatives at one position, scaled to sum to 1
H_i  = − Σ_v  q_v · ln q_v             entropy at position i
Mean token entropy = (1/L) · Σ_i H_i
```

## Symbols

| Symbol | Meaning |
|---|---|
| `v, u` | the alternative tokens Ollama reports at one position (the top 5) |
| `q_v` | their probabilities, renormalised to add up to 1 |
| `H_i` | the entropy at position i: 0 when one token has all the probability |
| `L` | number of tokens |

## Worked example ("Canberra." = 3 positions)

```
position 1: "Can" 0.90, "Syd" 0.06, "Mel" 0.04  ->  H = 0.392
position 2: "berra" 0.95, "bera" 0.05             ->  H = 0.199
position 3: "." 0.99, "!" 0.01                    ->  H = 0.056
mean = (0.392 + 0.199 + 0.056) / 3 = 0.216
```

## Reading the score

0 = no doubt at any word; higher = more torn. Because only the top few alternatives are seen, it slightly under-estimates the full entropy. None without top log-probs.
