# GSM8K test split (full)

`data/eval_sets/full/gsm8k_test.jsonl` · 1,319 records · schema: [`gsm8k_test.schema.json`](../schema/gsm8k_test.schema.json)

> All 1,319 GSM8K test problems, frozen for the pipeline's maths evaluation.

## At a glance

| | |
|---|---|
| Used for | pipeline evaluation (thesis Objectives 3-4) |
| Records | 1,319 |
| Source | openai/gsm8k ([link](https://huggingface.co/datasets/openai/gsm8k)) |
| Split | test (main configuration) |
| Pinned revision | `740312add88f781978c0658806c59bc2815b9866` |
| Licence | MIT |
| Items used in reported results | items 0-99 (test), 500-539 (dev) |
| SHA-256 | `38ab30905125a2d45124b1bdc8e852d44246104bb027d8ae624cf6d958feaefb` (matches the file) |

## What the dataset is about

GSM8K has 8,500 grade-school maths word problems (7,473 train, 1,319 test) written by human problem writers. Each needs between 2 and 8 steps of basic arithmetic. The reference solution is written in plain language and ends with '#### <answer>'.

## Why it is in this project

The pipeline's main maths benchmark for Objectives 3 and 4 (Layer 3, reasoning verification). The reported pipeline results used items 0 to 99 as the test slice and items 500 to 539 as the dev slice. The Objective 1-2 set `data/uq_sets/gsm8k.jsonl` is drawn from the same test split.

## How this file was made

1. Written by `python -m evaluation.export_eval_sets`, which reads the split with the pipeline's own loader (`evaluation.benchmark_runner.load_gsm8k`) at the pinned revision.
2. Every test problem is kept, in the original order.

## What each record contains

Each line of the file is one JSON object with these fields (the full types are in the schema):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | string | Question id: 'gsm_' + the problem's index in the GSM8K test split. | `gsm_0` |
| `prompt` | string | The word problem, as sent to the model. | `Janet’s ducks lay 16 eggs per day. She eats three for break…` |
| `correct_answers` | array of strings, 1 to 1 items | The final numeric answer (one item). | `["18"]` |
| `numerical_answer` | string | The same final answer as a plain string. | `18` |

## Statistics

| | |
|---|---:|
| Records | 1,319 |
| Prompt, characters (min / median / max) | 73 / 222 / 848 |
| True answers per question (min / median / max) | 1 / 1 / 1 |


## Example record

A real record from the file:

```json
{
  "id": "gsm_1",
  "prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?",
  "correct_answers": [
    "3"
  ],
  "numerical_answer": "3"
}
```

## Caveats

- `correct_answers` and `numerical_answer` hold the same value; both are kept because the pipeline's evaluation code reads them.

## Reference

K. Cobbe, V. Kosaraju, M. Bavarian, et al. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168, 2021.
