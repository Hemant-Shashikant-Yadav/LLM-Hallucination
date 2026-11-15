# AmbigQA: ambiguous questions

`data/uq_sets/ambigqa.jsonl` · 200 questions · schema: [`ambigqa.schema.json`](../schema/ambigqa.schema.json)

> Open-domain questions with more than one reasonable reading, each reading with its own answers.

## At a glance

| | |
|---|---|
| Domain | Ambiguous questions |
| Answer kind | `short`: free-text answer, graded by an LLM judge from another model family |
| Questions | 200 (extended tier); 100 in the core tier |
| Dev / test | 60 / 140 |
| With evidence / with hint | 0 / 0 |
| Gold uncertainty types | ambiguous: 200 |
| Source | sewon/ambig_qa (light, validation; multipleQAs only) ([link](https://huggingface.co/datasets/sewon/ambig_qa)) |
| Pinned revision | `e969d0132f4dd28c2939d55be34f1788c00ccfe7` |
| Licence | CC BY-SA 3.0 |
| Built | 2026-11-09, by `python -m uncertainty build-sets --only ambigqa` |
| SHA-256 | `029d3335468c6ae78441ef075add2926b4fa743b6e5d803b912bd17086ed65dd` (matches the file) |

## What the dataset is about

AmbigQA has 14,042 open-domain questions from Natural Questions (NQ-open), annotated for ambiguity. When a question has several plausible readings, annotators rewrote each reading as its own unambiguous question with its own answers.

## Why it is in this study

Uncertainty that comes from the question itself: the model may be unsure because the question has more than one meaning, not because it lacks knowledge. Every record has the gold type 'ambiguous', which trains and tests the Objective 2 classifier.

## How this file was made

Built by `uncertainty/sets/build.py` (`build_ambigqa`). Every random choice is seeded (seed 7), so a rebuild gives the same file.

1. Read the `light` configuration's validation split at the pinned revision.
2. Keep only questions annotated as `multipleQAs`, with at least 2 readings and at least one answer.
3. Draw 200 (seeded).
4. `answers` = every answer of every reading, deduplicated, at most 20; `meta.readings` = up to 8 readings, each with its own answers.
5. Mark the core tier (100 questions) and the fixed dev (30%) / test (70%) split, both seeded per set.

## What each record contains

Each line of the file is one JSON object with these fields (the full types are in the schema):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | string | Unique question id in this set: 'ambig-' + the source dataset's own id. | `ambig--1004998905086430798` |
| `set` | string (always `ambigqa`) | Name of the question set: always 'ambigqa'. | `ambigqa` |
| `kind` | string (always `short`) | Answer kind 'short': free-text answer, graded by an LLM judge from another model family. | `short` |
| `question` | string | The question as sent to the model. | `When was the bellagio in las vegas built?` |
| `answers` | array of strings, 1 to 20 items | Every answer of every reading, deduplicated, at most 20; an answer to any reading counts as correct. | `["May, 1996", "October 15, 1998", "May 1996"]` |
| `split` | string (`dev`, `test`) | Fixed, seeded split: dev (30%) only tunes thresholds and trains the Objective 2 models; every reported number uses test (70%). | `dev` |
| `core` | boolean | true if the question is in the core tier, the subset every model answers; the file itself is the extended tier. | `false` |
| `judge_mode` | string (`match`) | How the LLM judge grades a free-text answer: match (agrees with a reference answer), truthfulqa (TruthfulQA judge prompt), abstain (correct only if it says the question cannot be answered), premise (correct only if it rejects the false premise). Unused for rule-graded kinds. | `match` |
| `incorrect_answers` | array (always empty) | Known false answers from the dataset (TruthfulQA only; empty elsewhere). | `[]` |
| `gold_type` | string (`ambiguous`) | Uncertainty type the dataset itself guarantees (ambiguous, unanswerable, false_premise or temporal); null when it gives none. | `ambiguous` |
| `evidence` | null | Always null: this set has no evidence passage. | null |
| `hint` | null | Always null: this set has no hint. | null |
| `meta` | object | Extra fields kept from the source dataset (for strata and analysis). | (see below) |
| `meta.readings` | array of objects, 2 to 8 items | The question's readings (up to 8), each rewritten as an unambiguous question with its own answers. | `{"question": "When did construction begin on the Bellagio i…` |

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
| Question, characters | 33 / 45 / 73 |
| Accepted answers per question | 1 / 4 / 20 |

Readings per question: min / median / max = 2 / 4 / 8.

**Gold uncertainty type**

| Gold type | Count | Share |
|---|---:|---:|
| ambiguous | 200 | 100% |

## Example record

A real record from the file:

```json
{
  "id": "ambig--3421486869502537513",
  "set": "ambigqa",
  "kind": "short",
  "question": "When does the new royal liverpool hospital open?",
  "answers": [
    "1978",
    "2022"
  ],
  "split": "test",
  "core": true,
  "judge_mode": "match",
  "incorrect_answers": [],
  "gold_type": "ambiguous",
  "evidence": null,
  "hint": null,
  "meta": {
    "readings": [
      {
        "question": "When does the new royal liverpool hospital open before redevelopment?",
        "answers": [
          "1978"
        ]
      },
      {
        "question": "When does the new royal liverpool hospital open after redevelopment?",
        "answers": [
          "2022"
        ]
      }
    ]
  }
}
```

## How the study uses it

- **Generation:** every model answers each question once greedily and 10 more times by sampling; the uncertainty methods compare those answers.
- **Grading:** The model's answer is free text. A judge model from another family grades it with the record's `judge_mode`; a second judge from a third family re-grades a sample as a check.
- **Splits:** dev questions tune the thresholds and train the Objective 2 detector and classifier; test questions give every reported number.
- **Gold types:** `gold_type` gives the uncertainty type straight from the dataset, so these questions get their Objective 2 type label from the metadata.

## Caveats

- An answer to any reading counts as correct, so accuracy measures whether the model answers some valid reading; whether it notices the ambiguity is what the type labels capture.

## Reference

S. Min, J. Michael, H. Hajishirzi, L. Zettlemoyer. AmbigQA: Answering Ambiguous Open-domain Questions. EMNLP 2020.
