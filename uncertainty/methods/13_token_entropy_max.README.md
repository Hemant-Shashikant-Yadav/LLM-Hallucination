# uncertainty/methods/13_token_entropy_max.py — Max token entropy

Method 13 of 17. Part of [the 17 methods](README.md); the same text is the file's docstring.

| | |
|---|---|
| Family | the model's own word probabilities (token-level) |
| Reference | Malinin & Gales, 2021 [4] |
| Needs | the top alternative tokens at each position of the greedy answer |
| Code | token_entropy_max (per-position entropies: common.token_entropies). |

## What it asks

Where was the model most torn between its choices? The single most doubtful word.

## Formula

```
H_i  = − Σ_v  q_v · ln q_v            entropy at position i (as in method 12)
Max token entropy = max_i H_i
```

## Symbols

| Symbol | Meaning |
|---|---|
| `q_v` | the top alternatives at one position, renormalised to sum to 1 |
| `H_i` | the entropy at position i |
| `max_i` | the largest value over all positions |

## Worked example ("Canberra." = 3 positions)

```
H = 0.392 ("Can" vs "Syd" vs "Mel"), 0.199, 0.056  ->  max = 0.392
```

## Reading the score

0 = no doubt at any word; higher = at least one very doubtful word (here the first one, where "Sydney" was a real option). None without top log-probs.
