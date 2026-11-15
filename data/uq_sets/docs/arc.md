# ARC-Challenge: science multiple choice

`data/uq_sets/arc.jsonl` · 200 questions · schema: [`arc.schema.json`](../schema/arc.schema.json)

> Grade-school science questions with four options; the hard (Challenge) part of ARC.

## At a glance

| | |
|---|---|
| Domain | Science MCQ |
| Answer kind | `choice`: one option letter, read and graded by rules |
| Questions | 200 (extended tier); 100 in the core tier |
| Dev / test | 60 / 140 |
| With evidence / with hint | 0 / 0 |
| Gold uncertainty types | none |
| Source | allenai/ai2_arc (ARC-Challenge, test) ([link](https://huggingface.co/datasets/allenai/ai2_arc)) |
| Pinned revision | `210d026faf9955653af8916fad021475a3f00453` |
| Licence | CC BY-SA 4.0 |
| Built | 2026-11-09, by `python -m uncertainty build-sets --only arc` |
| SHA-256 | `784f87cf66ebc92d1333516b58c85e0464b1d5bb68a61cc6694b183df932db0c` (matches the file) |

## What the dataset is about

ARC (AI2 Reasoning Challenge) has 7,787 grade-school science multiple-choice questions, split into an Easy set and a Challenge set. The Challenge set holds the questions that simple retrieval and word co-occurrence methods got wrong. This file uses the Challenge test split.

## Why it is in this study

Science reasoning with a fixed set of options: the answer is an option letter read by rules, and it is easy to see when the samples split between options.

## How this file was made

Built by `uncertainty/sets/build.py` (`build_arc`). Every random choice is seeded (seed 7), so a rebuild gives the same file.

1. Read `ARC-Challenge/test` (Parquet) at the pinned revision.
2. Keep questions with exactly 4 options and draw 200 (seeded); the core tier is stratified by answer letter.
3. The options are added to the question as (a) to (d), whatever their original labels (A to D or 1 to 4), and the answer is the matching letter.
4. Mark the core tier (100 questions) and the fixed dev (30%) / test (70%) split, both seeded per set.

## What each record contains

Each line of the file is one JSON object with these fields (the full types are in the schema):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | string | Unique question id in this set: 'arc-' + the source dataset's own id. | `arc-ACTAAP_2007_7_36` |
| `set` | string (always `arc`) | Name of the question set: always 'arc'. | `arc` |
| `kind` | string (always `choice`) | Answer kind 'choice': one option letter, read and graded by rules. | `choice` |
| `question` | string | The question followed by its four options, one per line, as (a) to (d). | `Which is a renewable natural resource that can be replenish…` |
| `answers` | array of strings, 1 to 1 items | The correct option letter, 'a' to 'd' (one item). | `["c"]` |
| `split` | string (`dev`, `test`) | Fixed, seeded split: dev (30%) only tunes thresholds and trains the Objective 2 models; every reported number uses test (70%). | `test` |
| `core` | boolean | true if the question is in the core tier, the subset every model answers; the file itself is the extended tier. | `false` |
| `judge_mode` | string (`match`) | How the LLM judge grades a free-text answer: match (agrees with a reference answer), truthfulqa (TruthfulQA judge prompt), abstain (correct only if it says the question cannot be answered), premise (correct only if it rejects the false premise). Unused for rule-graded kinds. | `match` |
| `incorrect_answers` | array (always empty) | Known false answers from the dataset (TruthfulQA only; empty elsewhere). | `[]` |
| `gold_type` | null | Always null: the dataset gives no uncertainty type for these questions. | null |
| `evidence` | null | Always null: this set has no evidence passage. | null |
| `hint` | null | Always null: this set has no hint. | null |
| `meta` | object (empty) | No extra fields in this set (always an empty object). | `{}` |

## Statistics

| | All (extended tier) | Core tier |
|---|---:|---:|
| Questions | 200 | 100 |
| Dev | 60 | 27 |
| Test | 140 | 73 |
| With evidence | 0 | 0 |
| With hint | 0 | 0 |

| Length (min / median / max) | |
|---|---:|
| Question, characters | 70 / 237.5 / 925 |
| Accepted answers per question | 1 / 1 / 1 |

**Correct answer**

| Answer | Count | Share |
|---|---:|---:|
| c | 53 | 26% |
| a | 49 | 24% |
| b | 49 | 24% |
| d | 49 | 24% |

## Example record

A real record from the file:

```json
{
  "id": "arc-MEA_2011_8_19",
  "set": "arc",
  "kind": "choice",
  "question": "Which of the following is made of atoms?\n(a) gravity\n(b) light\n(c) heat\n(d) air",
  "answers": [
    "d"
  ],
  "split": "dev",
  "core": false,
  "judge_mode": "match",
  "incorrect_answers": [],
  "gold_type": null,
  "evidence": null,
  "hint": null,
  "meta": {}
}
```

## How the study uses it

- **Generation:** every model answers each question once greedily and 10 more times by sampling; the uncertainty methods compare those answers.
- **Grading:** The chosen option letter is read from the model's answer and compared with `answers[0]` by rules (no judge).
- **Splits:** dev questions tune the thresholds and train the Objective 2 detector and classifier; test questions give every reported number.

## Caveats

- No evidence or hint, so ARC is not used for the Objective 2 re-ask interventions.

## Reference

P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, O. Tafjord. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457, 2018.
