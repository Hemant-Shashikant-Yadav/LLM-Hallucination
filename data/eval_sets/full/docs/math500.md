# MATH-500

`data/eval_sets/full/math500.jsonl` · 500 records · schema: [`math500.schema.json`](../schema/math500.schema.json)

> The standard 500-problem subset of the MATH test set.

## At a glance

| | |
|---|---|
| Used for | pipeline evaluation (thesis Objectives 3-4) |
| Records | 500 |
| Source | HuggingFaceH4/MATH-500 ([link](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)) |
| Split | test |
| Pinned revision | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| Licence | MIT (from MATH) |
| SHA-256 | `6692a2f6c5b6baf141262d3442edf2480657580ff901b7e46977a34fcc07af72` (matches the file) |

## What the dataset is about

MATH-500 is a subset of 500 problems from the MATH test set, held out by Lightman et al. (OpenAI) in 'Let's Verify Step by Step' and now a common smaller test of mathematical reasoning. It keeps MATH's 7 subjects and 5 difficulty levels.

## Why it is in this project

Used by the pipeline's hard-problem evaluation (`evaluation/hard_eval.py`, set `math500`).

## How this file was made

1. Written by `python -m evaluation.export_eval_sets` from `HuggingFaceH4/MATH-500` (test split, pinned revision).
2. `key` = the dataset's answer field.

## What each record contains

Each line of the file is one JSON object with these fields (the full types are in the schema):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | string | Problem id: 'm500-' + the problem's path in the original MATH test set. | `m500-test/precalculus/807.json` |
| `subject` | string (`Algebra`, `Counting & Probability`, `Geometry`, `Intermediate Algebra`, `Number Theory`, `Prealgebra`, `Precalculus`) | MATH subject of the problem. | `Precalculus` |
| `level` | integer (`1`, `2`, `3`, `4`, `5`), 1 to 5 | Difficulty level, 1 (easiest) to 5. | `2` |
| `question` | string | The problem statement in LaTeX. | `Convert the point $(0,3)$ in rectangular coordinates to pol…` |
| `kind` | string (always `expr`) | Answer kind: always 'expr' (a maths expression read by rules). | `expr` |
| `key` | string | Final answer as given by MATH-500. | `\left( 3, \frac{\pi}{2} \right)` |

## Statistics

| | |
|---|---:|
| Records | 500 |
| Question, characters (min / median / max) | 20 / 148 / 1,733 |

**Subject**

| Subject | Count | Share |
|---|---:|---:|
| Algebra | 124 | 25% |
| Intermediate Algebra | 97 | 19% |
| Prealgebra | 82 | 16% |
| Number Theory | 62 | 12% |
| Precalculus | 56 | 11% |
| Geometry | 41 | 8% |
| Counting & Probability | 38 | 8% |

**Level**

| Level | Count | Share |
|---|---:|---:|
| 1 | 43 | 9% |
| 2 | 90 | 18% |
| 3 | 105 | 21% |
| 4 | 128 | 26% |
| 5 | 134 | 27% |

## Example record

A real record from the file:

```json
{
  "id": "m500-test/algebra/1098.json",
  "subject": "Algebra",
  "level": 3,
  "question": "Evaluate $(1+2i)6-3i$.",
  "kind": "expr",
  "key": "6+9i"
}
```

## Caveats

- Problems come from the MATH test set, so they can also appear in `math_test.jsonl`.

## Reference

H. Lightman et al. Let's Verify Step by Step. arXiv:2305.20050, 2023 (subset of: D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, J. Steinhardt. Measuring Mathematical Problem Solving With the MATH Dataset. NeurIPS 2021 (Datasets and Benchmarks).)
