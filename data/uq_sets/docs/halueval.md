# HaluEval (QA): multi-hop facts

`data/uq_sets/halueval.jsonl` · 300 questions · schema: [`halueval.schema.json`](../schema/halueval.schema.json)

> Two-step factual questions (built on HotpotQA) with the knowledge needed to answer them.

## At a glance

| | |
|---|---|
| Domain | Multi-hop facts |
| Answer kind | `short`: free-text answer, graded by an LLM judge from another model family |
| Questions | 300 (extended tier); 150 in the core tier |
| Dev / test | 90 / 210 |
| With evidence / with hint | 300 / 0 |
| Gold uncertainty types | none |
| Source | pminervini/HaluEval (qa) ([link](https://huggingface.co/datasets/pminervini/HaluEval)) |
| Pinned revision | `12a856119f03975a94509091e8cada3e6be6ead7` |
| Licence | MIT |
| Built | 2026-11-09, by `python -m uncertainty build-sets --only halueval` |
| SHA-256 | `e6784473b338b82662de40be81aca129b2cb8e3087f1053d7208a96a21638e97` (matches the file) |

## What the dataset is about

HaluEval is a benchmark for recognising hallucinations, with 35,000 samples: 5,000 general user queries with ChatGPT responses and 10,000 each for question answering, knowledge-grounded dialogue and summarisation. The question-answering part is built on HotpotQA: each question has its supporting knowledge, the right answer and a generated hallucinated answer.

## Why it is in this study

Multi-hop factual questions, where two facts must be combined. They sit between plain recall and reasoning, and the knowledge paragraph is a natural evidence passage for the knowledge intervention.

## How this file was made

Built by `uncertainty/sets/build.py` (`build_halueval`). Every random choice is seeded (seed 7), so a rebuild gives the same file.

1. Read the `qa` configuration (Parquet) at the pinned revision.
2. Draw 300 row indices with a seeded generator.
3. Answers: the dataset's `right_answer` (one per question).
4. Evidence: the dataset's `knowledge` field, cut to 1,200 characters.
5. The hallucinated answer is not stored.
6. Mark the core tier (150 questions) and the fixed dev (30%) / test (70%) split, both seeded per set.

## What each record contains

Each line of the file is one JSON object with these fields (the full types are in the schema):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | string | Unique question id in this set: 'halu-' + the source dataset's own id. | `halu-32` |
| `set` | string (always `halueval`) | Name of the question set: always 'halueval'. | `halueval` |
| `kind` | string (always `short`) | Answer kind 'short': free-text answer, graded by an LLM judge from another model family. | `short` |
| `question` | string | The question as sent to the model. | `Are Jane and First for Women both women's magazines?` |
| `answers` | array of strings, 1 to 1 items | The dataset's right answer (one item). | `["yes"]` |
| `split` | string (`dev`, `test`) | Fixed, seeded split: dev (30%) only tunes thresholds and trains the Objective 2 models; every reported number uses test (70%). | `test` |
| `core` | boolean | true if the question is in the core tier, the subset every model answers; the file itself is the extended tier. | `true` |
| `judge_mode` | string (`match`) | How the LLM judge grades a free-text answer: match (agrees with a reference answer), truthfulqa (TruthfulQA judge prompt), abstain (correct only if it says the question cannot be answered), premise (correct only if it rejects the false premise). Unused for rule-graded kinds. | `match` |
| `incorrect_answers` | array (always empty) | Known false answers from the dataset (TruthfulQA only; empty elsewhere). | `[]` |
| `gold_type` | null | Always null: the dataset gives no uncertainty type for these questions. | null |
| `evidence` | string | The dataset's knowledge paragraph(s) for the question (at most 1,200 characters). Used only by the Objective 2 knowledge intervention. | `Jane was an American magazine created to appeal to the wome…` |
| `hint` | null | Always null: this set has no hint. | null |
| `meta` | object (empty) | No extra fields in this set (always an empty object). | `{}` |

## Statistics

| | All (extended tier) | Core tier |
|---|---:|---:|
| Questions | 300 | 150 |
| Dev | 90 | 46 |
| Test | 210 | 104 |
| With evidence | 300 | 150 |
| With hint | 0 | 0 |

| Length (min / median / max) | |
|---|---:|
| Question, characters | 36 / 91 / 450 |
| Accepted answers per question | 1 / 1 / 1 |
| Evidence, characters | 97 / 331 / 882 |

Questions whose answer is a plain yes or no: 19 (6%).


## Example record

A real record from the file:

```json
{
  "id": "halu-887",
  "set": "halueval",
  "kind": "short",
  "question": "Are Sparta and King Charles both from the same country?",
  "answers": [
    "no"
  ],
  "split": "test",
  "core": true,
  "judge_mode": "match",
  "incorrect_answers": [],
  "gold_type": null,
  "evidence": "Sparta is an American rock band from El Paso, Texas, formed in 2001.King Charles (born Charles Costa, 6 May 1985) is a singer-songwriter from West London.",
  "hint": null,
  "meta": {}
}
```

## How the study uses it

- **Generation:** every model answers each question once greedily and 10 more times by sampling; the uncertainty methods compare those answers.
- **Grading:** The model's answer is free text. A judge model from another family grades it with the record's `judge_mode`; a second judge from a third family re-grades a sample as a check.
- **Splits:** dev questions tune the thresholds and train the Objective 2 detector and classifier; test questions give every reported number.
- **Objective 2 re-asks:** for the focus models, wrong or unsure answers are asked again (5 samples) with `evidence` (knowledge intervention); whether the answer gets fixed tells which kind of uncertainty it was.

## Caveats

- There is one reference answer per question, so correct answers phrased differently rely on the judge.
- Some questions are comparisons with a yes/no or one-of-two answer.

## Reference

J. Li, X. Cheng, W. X. Zhao, J.-Y. Nie, J.-R. Wen. HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models. EMNLP 2023.
