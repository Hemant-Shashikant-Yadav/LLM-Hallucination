# SelfAware: unanswerable questions

`data/uq_sets/selfaware.jsonl` · 300 questions · schema: [`selfaware.schema.json`](../schema/selfaware.schema.json)

> Questions with no true answer, mixed with answerable controls.

## At a glance

| | |
|---|---|
| Domain | Unanswerable questions |
| Answer kind | `short`: free-text answer, graded by an LLM judge from another model family |
| Questions | 300 (extended tier); 150 in the core tier |
| Dev / test | 90 / 210 |
| With evidence / with hint | 0 / 0 |
| Gold uncertainty types | unanswerable: 200 |
| Source | github.com/yinzhangyue/SelfAware data/SelfAware.json ([link](https://github.com/yinzhangyue/SelfAware)) |
| Pinned revision | `6b15164a3e11a2be714afde0615879dcde5c01f8` |
| Licence | CC BY-SA 4.0 |
| Built | 2026-11-09, by `python -m uncertainty build-sets --only selfaware` |
| SHA-256 | `4f70be6e262320f9956ecb542e85d6f4a21cf187888ba27703e7c1b9f7b8a06d` (matches the file) |

## What the dataset is about

SelfAware has 1,032 unanswerable questions (no scientific consensus, imagination, completely subjective, too many variables, philosophical), collected from online forums and checked by annotators, and 2,337 answerable questions taken from SQuAD, HotpotQA and TriviaQA.

## Why it is in this study

Questions that cannot be answered: a good model says so, and a confident answer is a hallucination. Unanswerable items carry the gold type 'unanswerable'; the answerable controls stop the study from rewarding a model that refuses everything.

## How this file was made

Built by `uncertainty/sets/build.py` (`build_selfaware`). Every random choice is seeded (seed 7), so a rebuild gives the same file.

1. Download `data/SelfAware.json` from the authors' GitHub repository at the pinned commit.
2. Draw 200 unanswerable and 100 answerable questions (seeded); the core tier keeps the same mix.
3. Answerable questions come only from TriviaQA and HotpotQA, because SQuAD questions assume a passage the model does not see.
4. Unanswerable items have an empty answer list and `judge_mode` 'abstain'.
5. Mark the core tier (150 questions) and the fixed dev (30%) / test (70%) split, both seeded per set.

## What each record contains

Each line of the file is one JSON object with these fields (the full types are in the schema):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | string | Unique question id in this set: 'sa-' + the source dataset's own id. | `sa-33` |
| `set` | string (always `selfaware`) | Name of the question set: always 'selfaware'. | `selfaware` |
| `kind` | string (always `short`) | Answer kind 'short': free-text answer, graded by an LLM judge from another model family. | `short` |
| `question` | string | The question as sent to the model. | `What type of soup is a consommé or broth garnished with lon…` |
| `answers` | array of strings, 0 to 358 items | Accepted answers for answerable questions; an empty list for unanswerable ones. | `["julienne disambiguation", "jullienne", "julienne"]` |
| `split` | string (`dev`, `test`) | Fixed, seeded split: dev (30%) only tunes thresholds and trains the Objective 2 models; every reported number uses test (70%). | `dev` |
| `core` | boolean | true if the question is in the core tier, the subset every model answers; the file itself is the extended tier. | `true` |
| `judge_mode` | string (`abstain`, `match`) | How the LLM judge grades a free-text answer: match (agrees with a reference answer), truthfulqa (TruthfulQA judge prompt), abstain (correct only if it says the question cannot be answered), premise (correct only if it rejects the false premise). Unused for rule-graded kinds. | `match` |
| `incorrect_answers` | array (always empty) | Known false answers from the dataset (TruthfulQA only; empty elsewhere). | `[]` |
| `gold_type` | string or null (`unanswerable`, null) | Uncertainty type the dataset itself guarantees (ambiguous, unanswerable, false_premise or temporal); null when it gives none. | `unanswerable` |
| `evidence` | null | Always null: this set has no evidence passage. | null |
| `hint` | null | Always null: this set has no hint. | null |
| `meta` | object | Extra fields kept from the source dataset (for strata and analysis). | (see below) |
| `meta.answerable` | boolean | true for an answerable control question, false for an unanswerable one. | `true` |

## Statistics

| | All (extended tier) | Core tier |
|---|---:|---:|
| Questions | 300 | 150 |
| Dev | 90 | 46 |
| Test | 210 | 104 |
| With evidence | 0 | 0 |
| With hint | 0 | 0 |

| Length (min / median / max) | |
|---|---:|
| Question, characters | 13 / 63 / 259 |
| Accepted answers per question | 0 / 0 / 358 |

**Answerable?**

| Kind | Count | Share |
|---|---:|---:|
| unanswerable | 200 | 67% |
| answerable | 100 | 33% |

**Gold uncertainty type**

| Gold type | Count | Share |
|---|---:|---:|
| unanswerable | 200 | 67% |
| null | 100 | 33% |

## Example record

A real record from the file:

```json
{
  "id": "sa-535",
  "set": "selfaware",
  "kind": "short",
  "question": "An example of a Assault rifle is what?",
  "answers": [
    "stg 44"
  ],
  "split": "test",
  "core": true,
  "judge_mode": "match",
  "incorrect_answers": [],
  "gold_type": null,
  "evidence": null,
  "hint": null,
  "meta": {
    "answerable": true
  }
}
```

## How the study uses it

- **Generation:** every model answers each question once greedily and 10 more times by sampling; the uncertainty methods compare those answers.
- **Grading:** The model's answer is free text. A judge model from another family grades it with the record's `judge_mode`; a second judge from a third family re-grades a sample as a check.
- **Splits:** dev questions tune the thresholds and train the Objective 2 detector and classifier; test questions give every reported number.
- **Gold types:** `gold_type` gives the uncertainty type straight from the dataset, so these questions get their Objective 2 type label from the metadata.

## Caveats

- A few 'unanswerable' questions have a widely accepted answer in practice; the abstain judge accepts an answer that explains there is no single true answer.

## Reference

Z. Yin, Q. Sun, Q. Guo, J. Wu, X. Qiu, X. Huang. Do Large Language Models Know What They Don't Know? Findings of ACL 2023.
