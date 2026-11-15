# TriviaQA: popular facts

`data/uq_sets/triviaqa.jsonl` · 300 questions · schema: [`triviaqa.schema.json`](../schema/triviaqa.schema.json)

> Trivia questions about well-known facts, each with an evidence passage from Wikipedia.

## At a glance

| | |
|---|---|
| Domain | Popular facts |
| Answer kind | `short`: free-text answer, graded by an LLM judge from another model family |
| Questions | 300 (extended tier); 150 in the core tier |
| Dev / test | 90 / 210 |
| With evidence / with hint | 300 / 0 |
| Gold uncertainty types | none |
| Source | mandarjoshi/trivia_qa (rc.wikipedia, validation) ([link](https://huggingface.co/datasets/mandarjoshi/trivia_qa)) |
| Pinned revision | `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f` |
| Licence | Apache-2.0 |
| Built | 2026-11-09, by `python -m uncertainty build-sets --only triviaqa` |
| SHA-256 | `eb5421bc20b59f15e82dd3c3c4695fd52c74043f56f302a603258eb7605b860f` (matches the file) |

## What the dataset is about

TriviaQA collects trivia and quiz-league questions written by trivia enthusiasts, each paired with evidence documents gathered independently (distant supervision). The full dataset has about 95,000 question-answer pairs and over 650,000 question-answer-evidence triples. Every answer comes with a list of aliases: other accepted names and spellings.

## Why it is in this study

Popular facts that most models have seen during training. It is the easy end of the knowledge spectrum: errors here are knowledge gaps on well-known topics, where semantic entropy is expected to work well.

## How this file was made

Built by `uncertainty/sets/build.py` (`build_triviaqa`). Every random choice is seeded (seed 7), so a rebuild gives the same file.

1. Read the `rc.wikipedia` validation split (Parquet) at the pinned revision.
2. Draw a seeded random sample of 600 question ids (twice the target).
3. Keep the questions whose Wikipedia context has a sentence naming one of the answer aliases; that sentence with one sentence either side (at most 600 characters) becomes the evidence.
4. Shuffle with a seeded generator and keep the first 300.
5. Answers: the main answer value followed by its aliases, duplicates removed, at most 12.
6. Mark the core tier (150 questions) and the fixed dev (30%) / test (70%) split, both seeded per set.

## What each record contains

Each line of the file is one JSON object with these fields (the full types are in the schema):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | string | Unique question id in this set: 'tqa-' + the source dataset's own id. | `tqa-qz_3752` |
| `set` | string (always `triviaqa`) | Name of the question set: always 'triviaqa'. | `triviaqa` |
| `kind` | string (always `short`) | Answer kind 'short': free-text answer, graded by an LLM judge from another model family. | `short` |
| `question` | string | The question as sent to the model. | `Which former West Indian fast bowler was known as 'Whisperi…` |
| `answers` | array of strings, 1 to 12 items | Accepted answers: the main answer first, then its aliases (other names and spellings); at most 12. | `["Michael Holding", "Mike Holding"]` |
| `split` | string (`dev`, `test`) | Fixed, seeded split: dev (30%) only tunes thresholds and trains the Objective 2 models; every reported number uses test (70%). | `test` |
| `core` | boolean | true if the question is in the core tier, the subset every model answers; the file itself is the extended tier. | `true` |
| `judge_mode` | string (`match`) | How the LLM judge grades a free-text answer: match (agrees with a reference answer), truthfulqa (TruthfulQA judge prompt), abstain (correct only if it says the question cannot be answered), premise (correct only if it rejects the false premise). Unused for rule-graded kinds. | `match` |
| `incorrect_answers` | array (always empty) | Known false answers from the dataset (TruthfulQA only; empty elsewhere). | `[]` |
| `gold_type` | null | Always null: the dataset gives no uncertainty type for these questions. | null |
| `evidence` | string | Oracle passage: the Wikipedia sentence naming the answer, with one sentence either side (at most 600 characters). Used only by the Objective 2 knowledge intervention. | `From the mid-1970s to the early 1990s, the West Indies team…` |
| `hint` | null | Always null: this set has no hint. | null |
| `meta` | object | Extra fields kept from the source dataset (for strata and analysis). | (see below) |
| `meta.answer_type` | string (`FreeForm`, `Numerical`, `WikipediaEntity`) | Answer type given by TriviaQA. | `WikipediaEntity` |

## Statistics

| | All (extended tier) | Core tier |
|---|---:|---:|
| Questions | 300 | 150 |
| Dev | 90 | 47 |
| Test | 210 | 103 |
| With evidence | 300 | 150 |
| With hint | 0 | 0 |

| Length (min / median / max) | |
|---|---:|
| Question, characters | 27 / 74 / 243 |
| Accepted answers per question | 1 / 11 / 12 |
| Evidence, characters | 30 / 339 / 600 |

**Answer type**

| Answer type | Count | Share |
|---|---:|---:|
| WikipediaEntity | 289 | 96% |
| Numerical | 9 | 3% |
| FreeForm | 2 | 1% |

## Example record

A real record from the file:

```json
{
  "id": "tqa-dpql_4988",
  "set": "triviaqa",
  "kind": "short",
  "question": "What was Margot’s surname in “The Good Life” TV series?",
  "answers": [
    "Leadbetter",
    "Leadbetter (disambiguation)"
  ],
  "split": "test",
  "core": false,
  "judge_mode": "match",
  "incorrect_answers": [],
  "gold_type": null,
  "evidence": "*Felicity Kendal — Barbara Good *Penelope Keith — Margo Leadbetter (née Sturgess) *Paul Eddington — Jeremy \"Jerry\" Leadbetter",
  "hint": null,
  "meta": {
    "answer_type": "WikipediaEntity"
  }
}
```

## How the study uses it

- **Generation:** every model answers each question once greedily and 10 more times by sampling; the uncertainty methods compare those answers.
- **Grading:** The model's answer is free text. A judge model from another family grades it with the record's `judge_mode`; a second judge from a third family re-grades a sample as a check.
- **Splits:** dev questions tune the thresholds and train the Objective 2 detector and classifier; test questions give every reported number.
- **Objective 2 re-asks:** for the focus models, wrong or unsure answers are asked again (5 samples) with `evidence` (knowledge intervention); whether the answer gets fixed tells which kind of uncertainty it was.

## Caveats

- The evidence passage was found by string matching, so it names the answer, but the alias may occasionally appear in another sense.
- Free-text answers are graded by the LLM judge; a model-free alias match is recorded beside each verdict as a check.

## Reference

M. Joshi, E. Choi, D. S. Weld, L. Zettlemoyer. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. ACL 2017.
