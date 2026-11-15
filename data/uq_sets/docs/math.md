# MATH: competition maths

`data/uq_sets/math.jsonl` · 250 questions · schema: [`math.schema.json`](../schema/math.schema.json)

> Competition maths problems in 7 subjects and 5 difficulty levels, with LaTeX answers.

## At a glance

| | |
|---|---|
| Domain | Competition maths |
| Answer kind | `expr`: one maths expression, read and graded by rules |
| Questions | 250 (extended tier); 120 in the core tier |
| Dev / test | 75 / 175 |
| With evidence / with hint | 0 / 183 |
| Gold uncertainty types | none |
| Source | data/eval_sets/full/math_test.jsonl (EleutherAI/hendrycks_math) ([link](https://huggingface.co/datasets/EleutherAI/hendrycks_math)) |
| Pinned revision | `21a5633873b6a120296cce3e2df9d5550074f4a3` |
| Licence | MIT |
| Built | 2026-11-09, by `python -m uncertainty build-sets --only math` |
| SHA-256 | `df7561dc651ad11a8d2f9c798d1a3f2cc18c389a7770993367b8f4979ec83a96` (matches the file) |

## What the dataset is about

MATH has 12,500 competition mathematics problems (7,500 train, 5,000 test) from contests such as AMC 10, AMC 12 and AIME, in 7 subjects and 5 difficulty levels. Each has a full step-by-step solution in LaTeX whose final answer is in `\boxed{}`.

## Why it is in this study

Harder multi-step reasoning with symbolic answers (fractions, roots, expressions). The levels let the study see how uncertainty changes with difficulty.

## How this file was made

Built by `uncertainty/sets/build.py` (`build_math`). Every random choice is seeded (seed 7), so a rebuild gives the same file.

1. Read the project's frozen file `data/eval_sets/full/math_test.jsonl` (all 5,000 test problems; `key` = the last boxed answer).
2. Drop problems with Asymptote diagrams (`[asy]`, which a text model cannot see), problems longer than 600 characters and problems without a key.
3. Draw 250 spread evenly across the 5 levels (seeded); the core tier is stratified by level too.
4. Hint: the first half of the solution's sentences, without any sentence holding `\boxed`; null when fewer than 2 such sentences exist.
5. Mark the core tier (120 questions) and the fixed dev (30%) / test (70%) split, both seeded per set.

## What each record contains

Each line of the file is one JSON object with these fields (the full types are in the schema):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | string | Unique question id in this set: 'math-' + the source dataset's own id. | `math-prealgebra-434` |
| `set` | string (always `math`) | Name of the question set: always 'math'. | `math` |
| `kind` | string (always `expr`) | Answer kind 'expr': one maths expression, read and graded by rules. | `expr` |
| `question` | string | The question as sent to the model. | `Compute $\left(\sqrt{97969}\right)^2$.` |
| `answers` | array of strings, 1 to 1 items | The final answer as a LaTeX expression (one item): the last \boxed{} answer of the reference solution. | `["97969"]` |
| `split` | string (`dev`, `test`) | Fixed, seeded split: dev (30%) only tunes thresholds and trains the Objective 2 models; every reported number uses test (70%). | `test` |
| `core` | boolean | true if the question is in the core tier, the subset every model answers; the file itself is the extended tier. | `false` |
| `judge_mode` | string (`match`) | How the LLM judge grades a free-text answer: match (agrees with a reference answer), truthfulqa (TruthfulQA judge prompt), abstain (correct only if it says the question cannot be answered), premise (correct only if it rejects the false premise). Unused for rule-graded kinds. | `match` |
| `incorrect_answers` | array (always empty) | Known false answers from the dataset (TruthfulQA only; empty elsewhere). | `[]` |
| `gold_type` | null | Always null: the dataset gives no uncertainty type for these questions. | null |
| `evidence` | null | Always null: this set has no evidence passage. | null |
| `hint` | string or null | First half of the reference solution's sentences, without the boxed answer; null when the solution has fewer than 2 such sentences. Used only by the Objective 2 reasoning intervention. | `For any nonnegative number $n$, the value of $\sqrt{n}$ is …` |
| `meta` | object | Extra fields kept from the source dataset (for strata and analysis). | (see below) |
| `meta.level` | string (`Level 1`, `Level 2`, `Level 3`, `Level 4`, `Level 5`) | Difficulty level, from Level 1 (easiest) to Level 5. | `Level 1` |
| `meta.subject` | string | MATH subject of the problem. | `prealgebra` |

## Statistics

| | All (extended tier) | Core tier |
|---|---:|---:|
| Questions | 250 | 120 |
| Dev | 75 | 34 |
| Test | 175 | 86 |
| With evidence | 0 | 0 |
| With hint | 183 | 85 |

| Length (min / median / max) | |
|---|---:|
| Question, characters | 18 / 131 / 589 |
| Accepted answers per question | 1 / 1 / 1 |
| Hint, characters | 6 / 173 / 1,133 |

**Level**

| Level | Count | Share |
|---|---:|---:|
| Level 1 | 50 | 20% |
| Level 2 | 50 | 20% |
| Level 3 | 50 | 20% |
| Level 4 | 50 | 20% |
| Level 5 | 50 | 20% |

**Subject**

| Subject | Count | Share |
|---|---:|---:|
| algebra | 63 | 25% |
| intermediate_algebra | 50 | 20% |
| prealgebra | 49 | 20% |
| precalculus | 31 | 12% |
| number_theory | 24 | 10% |
| geometry | 17 | 7% |
| counting_and_probability | 16 | 6% |

## Example record

A real record from the file:

```json
{
  "id": "math-prealgebra-604",
  "set": "math",
  "kind": "expr",
  "question": "What is $(7^{-1})^{-1}$?",
  "answers": [
    "7"
  ],
  "split": "dev",
  "core": false,
  "judge_mode": "match",
  "incorrect_answers": [],
  "gold_type": null,
  "evidence": null,
  "hint": null,
  "meta": {
    "level": "Level 1",
    "subject": "prealgebra"
  }
}
```

## How the study uses it

- **Generation:** every model answers each question once greedily and 10 more times by sampling; the uncertainty methods compare those answers.
- **Grading:** The final expression is read from the model's worked answer and compared with `answers[0]` as maths (equivalent forms count) by rules (no judge).
- **Splits:** dev questions tune the thresholds and train the Objective 2 detector and classifier; test questions give every reported number.
- **Objective 2 re-asks:** for the focus models, wrong or unsure answers are asked again (5 samples) with `hint` (reasoning intervention); whether the answer gets fixed tells which kind of uncertainty it was.

## Caveats

- Answers are compared as maths expressions by the pipeline's answer reader, which handles common equivalent forms; rare forms may still be misread.

## Reference

D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, J. Steinhardt. Measuring Mathematical Problem Solving With the MATH Dataset. NeurIPS 2021 (Datasets and Benchmarks).
