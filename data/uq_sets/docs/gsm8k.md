# GSM8K: arithmetic word problems

`data/uq_sets/gsm8k.jsonl` · 250 questions · schema: [`gsm8k.schema.json`](../schema/gsm8k.schema.json)

> Grade-school maths word problems with one numeric answer.

## At a glance

| | |
|---|---|
| Domain | Arithmetic word problems |
| Answer kind | `number`: one number, read and graded by rules |
| Questions | 250 (extended tier); 120 in the core tier |
| Dev / test | 75 / 175 |
| With evidence / with hint | 0 / 250 |
| Gold uncertainty types | none |
| Source | openai/gsm8k (main, test) ([link](https://huggingface.co/datasets/openai/gsm8k)) |
| Pinned revision | `740312add88f781978c0658806c59bc2815b9866` |
| Licence | MIT |
| Built | 2026-11-09, by `python -m uncertainty build-sets --only gsm8k` |
| SHA-256 | `20979f47830d25994d299d5034a67ef1af2b015813977c5252207c272f3407da` (matches the file) |

## What the dataset is about

GSM8K has 8,500 grade-school maths word problems (7,473 train, 1,319 test) written by human problem writers. Each needs between 2 and 8 steps of basic arithmetic. The reference solution is written in plain language and ends with '#### <answer>'.

## Why it is in this study

Arithmetic reasoning with a single numeric answer: answers are read by rules, and errors are mostly reasoning slips rather than missing knowledge.

## How this file was made

Built by `uncertainty/sets/build.py` (`build_gsm8k`). Every random choice is seeded (seed 7), so a rebuild gives the same file.

1. Load `openai/gsm8k` (`main`, test split) at the pinned revision with the `datasets` library.
2. Draw 250 indices with a seeded generator.
3. Answer: the number after '####', normalised by the pipeline's `normalize_number`.
4. Hint: the first half of the solution's lines, with the calculator notes (`<<...>>`) removed; it never includes the last step.
5. Mark the core tier (120 questions) and the fixed dev (30%) / test (70%) split, both seeded per set.

## What each record contains

Each line of the file is one JSON object with these fields (the full types are in the schema):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | string | Unique question id in this set: 'gsm_' + the source dataset's own id. | `gsm_0` |
| `set` | string (always `gsm8k`) | Name of the question set: always 'gsm8k'. | `gsm8k` |
| `kind` | string (always `number`) | Answer kind 'number': one number, read and graded by rules. | `number` |
| `question` | string | The question as sent to the model. | `Janet’s ducks lay 16 eggs per day. She eats three for break…` |
| `answers` | array of strings, 1 to 1 items | The final numeric answer as a string (one item), normalised (no commas or units). | `["18"]` |
| `split` | string (`dev`, `test`) | Fixed, seeded split: dev (30%) only tunes thresholds and trains the Objective 2 models; every reported number uses test (70%). | `test` |
| `core` | boolean | true if the question is in the core tier, the subset every model answers; the file itself is the extended tier. | `true` |
| `judge_mode` | string (`match`) | How the LLM judge grades a free-text answer: match (agrees with a reference answer), truthfulqa (TruthfulQA judge prompt), abstain (correct only if it says the question cannot be answered), premise (correct only if it rejects the false premise). Unused for rule-graded kinds. | `match` |
| `incorrect_answers` | array (always empty) | Known false answers from the dataset (TruthfulQA only; empty elsewhere). | `[]` |
| `gold_type` | null | Always null: the dataset gives no uncertainty type for these questions. | null |
| `evidence` | null | Always null: this set has no evidence passage. | null |
| `hint` | string | First half of the reference solution's lines, calculator notes removed. Used only by the Objective 2 reasoning intervention. | `Janet sells 16 - 3 - 4 = 9 duck eggs a day.` |
| `meta` | object (empty) | No extra fields in this set (always an empty object). | `{}` |

## Statistics

| | All (extended tier) | Core tier |
|---|---:|---:|
| Questions | 250 | 120 |
| Dev | 75 | 31 |
| Test | 175 | 89 |
| With evidence | 0 | 0 |
| With hint | 250 | 120 |

| Length (min / median / max) | |
|---|---:|
| Question, characters | 84 / 230 / 597 |
| Accepted answers per question | 1 / 1 / 1 |
| Hint, characters | 24 / 88.5 / 401 |


## Example record

A real record from the file:

```json
{
  "id": "gsm_113",
  "set": "gsm8k",
  "kind": "number",
  "question": "Ryan plants 2 flowers a day in his garden. After 15 days, how many flowers does he have if 5 did not grow?",
  "answers": [
    "25"
  ],
  "split": "test",
  "core": false,
  "judge_mode": "match",
  "incorrect_answers": [],
  "gold_type": null,
  "evidence": null,
  "hint": "Ryan plants 2*15=30 flowers in total.",
  "meta": {}
}
```

## How the study uses it

- **Generation:** every model answers each question once greedily and 10 more times by sampling; the uncertainty methods compare those answers.
- **Grading:** The final number is read from the model's worked answer and compared with `answers[0]` by rules (no judge).
- **Splits:** dev questions tune the thresholds and train the Objective 2 detector and classifier; test questions give every reported number.
- **Objective 2 re-asks:** for the focus models, wrong or unsure answers are asked again (5 samples) with `hint` (reasoning intervention); whether the answer gets fixed tells which kind of uncertainty it was.

## Caveats

- A few hints already contain an intermediate result that leaves only short work.

## Reference

K. Cobbe, V. Kosaraju, M. Bavarian, et al. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168, 2021.
