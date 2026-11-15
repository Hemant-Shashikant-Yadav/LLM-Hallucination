# FreshQA: changing facts and false premises

`data/uq_sets/freshqa.jsonl` · 200 questions · schema: [`freshqa.schema.json`](../schema/freshqa.schema.json)

> Questions whose answers change over time, or that rest on a false premise, dated 2025-09-30.

## At a glance

| | |
|---|---|
| Domain | Changing facts, false premises |
| Answer kind | `short`: free-text answer, graded by an LLM judge from another model family |
| Questions | 200 (extended tier); 100 in the core tier |
| Dev / test | 60 / 140 |
| With evidence / with hint | 0 / 0 |
| Gold uncertainty types | false_premise: 80, temporal: 80 |
| Source | bojanbabic/freshqa_09302025 (FreshQA snapshot of 2025-09-30, test) ([link](https://github.com/freshllms/freshqa)) |
| Pinned revision | `ae293206abcacde7a86ee2251a1909e8b0e7edb8` |
| Licence | Apache-2.0 (FreshLLMs) |
| Built | 2026-11-09, by `python -m uncertainty build-sets --only freshqa` |
| SHA-256 | `46403e552926b2f8e329bfe7f06868224c97186d1515b5fe51241d7aa3d5915c` (matches the file) |

## What the dataset is about

FreshQA has about 600 questions whose answers change over time (fast-changing and slow-changing facts), never change, or rest on a false premise. The authors keep the answers up to date, so every copy of the dataset is a dated snapshot.

## Why it is in this study

Temporal uncertainty (the model's knowledge is out of date) and false premises. A model can be fully consistent and still wrong here, which is the blind spot of sampling-based methods.

## How this file was made

Built by `uncertainty/sets/build.py` (`build_freshqa`). Every random choice is seeded (seed 7), so a rebuild gives the same file.

1. Read the snapshot of 2025-09-30 (`bojanbabic/freshqa_09302025` on Hugging Face, test split, pinned revision).
2. Answers: the non-empty `answer_0` to `answer_9` columns.
3. Gold type: 'false_premise' for false-premise questions, 'temporal' for fast- or slow-changing facts, null for never-changing facts.
4. Draw 200 spread evenly across the (false premise, fact type) groups (seeded); a group with fewer questions gives all it has.
5. False-premise items use `judge_mode` 'premise'.
6. Mark the core tier (100 questions) and the fixed dev (30%) / test (70%) split, both seeded per set.

## What each record contains

Each line of the file is one JSON object with these fields (the full types are in the schema):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | string | Unique question id in this set: 'fresh-' + the source dataset's own id. | `fresh-0` |
| `set` | string (always `freshqa`) | Name of the question set: always 'freshqa'. | `freshqa` |
| `kind` | string (always `short`) | Answer kind 'short': free-text answer, graded by an LLM judge from another model family. | `short` |
| `question` | string | The question as sent to the model. | `What is the name of the first animal to land on the moon?` |
| `answers` | array of strings, 1 to 10 items | Reference answers as of the 2025-09-30 snapshot; for false-premise questions they correct the premise. | `["No animal has ever landed on the moon yet.", "Since human…` |
| `split` | string (`dev`, `test`) | Fixed, seeded split: dev (30%) only tunes thresholds and trains the Objective 2 models; every reported number uses test (70%). | `dev` |
| `core` | boolean | true if the question is in the core tier, the subset every model answers; the file itself is the extended tier. | `false` |
| `judge_mode` | string (`match`, `premise`) | How the LLM judge grades a free-text answer: match (agrees with a reference answer), truthfulqa (TruthfulQA judge prompt), abstain (correct only if it says the question cannot be answered), premise (correct only if it rejects the false premise). Unused for rule-graded kinds. | `premise` |
| `incorrect_answers` | array (always empty) | Known false answers from the dataset (TruthfulQA only; empty elsewhere). | `[]` |
| `gold_type` | string or null (`false_premise`, `temporal`, null) | Uncertainty type the dataset itself guarantees (ambiguous, unanswerable, false_premise or temporal); null when it gives none. | `false_premise` |
| `evidence` | null | Always null: this set has no evidence passage. | null |
| `hint` | null | Always null: this set has no hint. | null |
| `meta` | object | Extra fields kept from the source dataset (for strata and analysis). | (see below) |
| `meta.fact_type` | string (`fast-changing`, `never-changing`, `slow-changing`) | How often the answer changes. | `slow-changing` |
| `meta.false_premise` | boolean | true if the question rests on a false premise. | `true` |
| `meta.effective_year` | string | Since when the current answer holds, as given by FreshQA (for example 'before 2022' or '2024'). | `before 2022` |
| `meta.num_hops` | string (`multi-hop`, `one-hop`) | Whether the question needs one fact or several. | `one-hop` |
| `meta.snapshot` | string (always `2025-09-30`) | Date of the FreshQA snapshot the answers come from. | `2025-09-30` |

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
| Question, characters | 19 / 59 / 187 |
| Accepted answers per question | 1 / 1 / 10 |

**Fact type × false premise**

| Group | Count | Share |
|---|---:|---:|
| fast-changing, valid premise | 40 | 20% |
| never-changing, valid premise | 40 | 20% |
| slow-changing, valid premise | 40 | 20% |
| never-changing, false premise | 39 | 20% |
| slow-changing, false premise | 39 | 20% |
| fast-changing, false premise | 2 | 1% |

**Gold type**

| Gold type | Count | Share |
|---|---:|---:|
| false_premise | 80 | 40% |
| temporal | 80 | 40% |
| null (never-changing) | 40 | 20% |

**Hops**

| Hops | Count | Share |
|---|---:|---:|
| one-hop | 148 | 74% |
| multi-hop | 52 | 26% |

## Example record

A real record from the file:

```json
{
  "id": "fresh-30",
  "set": "freshqa",
  "kind": "short",
  "question": "When did Yann LeCun join Google?",
  "answers": [
    "Yann has never ever joined Google."
  ],
  "split": "dev",
  "core": false,
  "judge_mode": "premise",
  "incorrect_answers": [],
  "gold_type": "false_premise",
  "evidence": null,
  "hint": null,
  "meta": {
    "fact_type": "slow-changing",
    "false_premise": true,
    "effective_year": "before 2022",
    "num_hops": "one-hop",
    "snapshot": "2025-09-30"
  }
}
```

## How the study uses it

- **Generation:** every model answers each question once greedily and 10 more times by sampling; the uncertainty methods compare those answers.
- **Grading:** The model's answer is free text. A judge model from another family grades it with the record's `judge_mode`; a second judge from a third family re-grades a sample as a check.
- **Splits:** dev questions tune the thresholds and train the Objective 2 detector and classifier; test questions give every reported number.
- **Gold types:** `gold_type` gives the uncertainty type straight from the dataset, so these questions get their Objective 2 type label from the metadata.

## Caveats

- Answers are true as of 2025-09-30, which may be after a model's training cut-off; that is part of what the set measures. The set is used mainly for the type labels.

## Reference

T. Vu et al. FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation. arXiv:2310.03214, 2023.
