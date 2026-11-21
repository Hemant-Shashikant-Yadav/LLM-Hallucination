# uncertainty/methods/08_disagreement_with_greedy.py — Disagreement with the greedy (main) answer

Method 08 of 17. Part of [the 17 methods](README.md); the same text is the file's docstring.

| | |
|---|---|
| Family | meaning of the samples |
| Reference | self-consistency, Wang et al., 2023 [11] |
| Needs | the greedy answer and the samples, grouped by meaning |
| Code | disagree_greedy. |

## What it asks

What share of the samples do not mean what the greedy answer means?

## Formula

```
Disagree = 1 − (number of samples equivalent to the greedy answer) / m
```

## Symbols

| Symbol | Meaning |
|---|---|
| `greedy answer` | the model's single most likely answer (temperature 0): the one we grade |
| `equivalent` | same meaning: same read answer, or two-way NLI entailment for free text |
| `m` | number of samples |

## Worked example

```
greedy "Canberra"; 7 of the 10 samples also say "Canberra"
Disagree = 1 − 7 / 10 = 0.30
```

## Reading the score

0 = every sample backs the main answer; 1 = none does. Close to what the pipeline's majority vote sees. None without meaning data.
