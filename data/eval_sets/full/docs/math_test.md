# MATH test split (full)

`data/eval_sets/full/math_test.jsonl` · 5,000 records · schema: [`math_test.schema.json`](../schema/math_test.schema.json)

> All 5,000 MATH test problems with their full solutions and final answers.

## At a glance

| | |
|---|---|
| Used for | pipeline evaluation (thesis Objectives 3-4) |
| Records | 5,000 |
| Source | EleutherAI/hendrycks_math (all 7 subjects) ([link](https://huggingface.co/datasets/EleutherAI/hendrycks_math)) |
| Split | test |
| Pinned revision | `21a5633873b6a120296cce3e2df9d5550074f4a3` |
| Licence | MIT |
| Note | key = last boxed answer of the reference solution |
| SHA-256 | `14cd352ece0bf258e550f4225651845a9a88600b879e0a35d81496e20ba4a198` (matches the file) |

## What the dataset is about

MATH has 12,500 competition mathematics problems (7,500 train, 5,000 test) from contests such as AMC 10, AMC 12 and AIME, in 7 subjects and 5 difficulty levels. Each has a full step-by-step solution in LaTeX whose final answer is in `\boxed{}`.

## Why it is in this project

Used by the pipeline's hard-problem evaluation (`evaluation/hard_eval.py`, set `math`) and as the source of `data/uq_sets/math.jsonl`.

## How this file was made

1. Written by `python -m evaluation.export_eval_sets` from `EleutherAI/hendrycks_math` (all 7 subjects, test split, pinned revision).
2. `key` = the last boxed answer of the reference solution (`app.answer_utils.final_expression`).

## What each record contains

Each line of the file is one JSON object with these fields (the full types are in the schema):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | string | Problem id: 'math-' + subject + '-' + the problem's index within that subject. | `math-algebra-0` |
| `subject` | string (`algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`, `number_theory`, `prealgebra`, `precalculus`) | MATH subject of the problem. | `algebra` |
| `level` | string (`Level 1`, `Level 2`, `Level 3`, `Level 4`, `Level 5`) | Difficulty level, from Level 1 (easiest) to Level 5. | `Level 3` |
| `question` | string | The problem statement in LaTeX. | `How many vertical asymptotes does the graph of $y=\frac{2}{…` |
| `kind` | string (always `expr`) | Answer kind: always 'expr' (a maths expression read by rules). | `expr` |
| `key` | string | Final answer: the last \boxed{} expression of the reference solution. | `2` |
| `solution` | string | Full reference solution in LaTeX. | `The denominator of the rational function factors into $x^2+…` |

## Statistics

| | |
|---|---:|
| Records | 5,000 |
| Question, characters (min / median / max) | 16 / 143 / 2,828 |
| Solution, characters (min / median / max) | 26 / 394 / 5,313 |

**Subject**

| Subject | Count | Share |
|---|---:|---:|
| algebra | 1187 | 24% |
| intermediate_algebra | 903 | 18% |
| prealgebra | 871 | 17% |
| precalculus | 546 | 11% |
| number_theory | 540 | 11% |
| geometry | 479 | 10% |
| counting_and_probability | 474 | 9% |

**Level**

| Level | Count | Share |
|---|---:|---:|
| Level 1 | 437 | 9% |
| Level 2 | 894 | 18% |
| Level 3 | 1131 | 23% |
| Level 4 | 1214 | 24% |
| Level 5 | 1324 | 26% |

## Example record

A real record from the file:

```json
{
  "id": "math-algebra-4",
  "subject": "algebra",
  "level": "Level 1",
  "question": "If $2^8=4^x$, what is the value of $x$?",
  "kind": "expr",
  "key": "4",
  "solution": "Rewrite $4$ as $2^2$ to find $4^x=2^{2x}$.  Since $2^8=2^{2x}$, we have $2x=8$ which implies $x=\\boxed{4}$."
}
```

## Caveats

- 419 problems include Asymptote diagrams (`[asy]`) that a text model cannot see; the Objective 1-2 set drops them.

## Reference

D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, J. Steinhardt. Measuring Mathematical Problem Solving With the MATH Dataset. NeurIPS 2021 (Datasets and Benchmarks).
