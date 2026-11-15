# PopQA: long-tail facts

`data/uq_sets/popqa.jsonl` · 300 questions · schema: [`popqa.schema.json`](../schema/popqa.schema.json)

> Template questions about Wikidata entities, from very popular to very rare, with each subject's popularity.

## At a glance

| | |
|---|---|
| Domain | Long-tail facts |
| Answer kind | `short`: free-text answer, graded by an LLM judge from another model family |
| Questions | 300 (extended tier); 150 in the core tier |
| Dev / test | 90 / 210 |
| With evidence / with hint | 300 / 0 |
| Gold uncertainty types | none |
| Source | akariasai/PopQA (test.tsv); evidence: English Wikipedia page summaries ([link](https://huggingface.co/datasets/akariasai/PopQA)) |
| Pinned revision | `098765c79ea10a2cb19c828324e33281b8336ec0` |
| Licence | MIT (PopQA); CC BY-SA 4.0 (Wikipedia text) |
| Built | 2026-11-09, by `python -m uncertainty build-sets --only popqa` |
| SHA-256 | `40def0273a61a761e5c361ff0f5dab6e4e7ee4edb781d0afe3bd48c3d242d56a` (matches the file) |

## What the dataset is about

PopQA has about 14,000 entity-centric questions generated from Wikidata triples (subject, relation, object) with templates, for example 'Who is the father of X?', over 16 relation types. Each subject comes with its Wikipedia popularity (monthly page views), so the dataset shows how accuracy falls for rarer, long-tail facts.

## Why it is in this study

Long-tail facts the model has often never seen: the clearest case of knowledge uncertainty. The popularity value lets the study check whether entropy rises as facts get rarer.

## How this file was made

Built by `uncertainty/sets/build.py` (`build_popqa`). Every random choice is seeded (seed 7), so a rebuild gives the same file.

1. Read `test.tsv` at the pinned revision.
2. Put every question into one of 5 popularity bins (quintiles of log10 of `s_pop`).
3. Draw 300 questions spread evenly across the bins (seeded).
4. Answers: the dataset's `possible_answers` list.
5. Evidence: the plain-text introduction of the subject's English Wikipedia article, fetched with the Wikipedia API with redirects followed, at most 800 characters.
6. The core tier is stratified by popularity bin, so it stays balanced too.
7. Mark the core tier (150 questions) and the fixed dev (30%) / test (70%) split, both seeded per set.

## What each record contains

Each line of the file is one JSON object with these fields (the full types are in the schema):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | string | Unique question id in this set: 'popqa-' + the source dataset's own id. | `popqa-4577731` |
| `set` | string (always `popqa`) | Name of the question set: always 'popqa'. | `popqa` |
| `kind` | string (always `short`) | Answer kind 'short': free-text answer, graded by an LLM judge from another model family. | `short` |
| `question` | string | The question as sent to the model. | `Who is the father of Isaac B. Desha?` |
| `answers` | array of strings, 1 to 190 items | Accepted answers: the dataset's possible_answers (the object's name and its aliases). | `["Joseph Desha"]` |
| `split` | string (`dev`, `test`) | Fixed, seeded split: dev (30%) only tunes thresholds and trains the Objective 2 models; every reported number uses test (70%). | `test` |
| `core` | boolean | true if the question is in the core tier, the subset every model answers; the file itself is the extended tier. | `true` |
| `judge_mode` | string (`match`) | How the LLM judge grades a free-text answer: match (agrees with a reference answer), truthfulqa (TruthfulQA judge prompt), abstain (correct only if it says the question cannot be answered), premise (correct only if it rejects the false premise). Unused for rule-graded kinds. | `match` |
| `incorrect_answers` | array (always empty) | Known false answers from the dataset (TruthfulQA only; empty elsewhere). | `[]` |
| `gold_type` | null | Always null: the dataset gives no uncertainty type for these questions. | null |
| `evidence` | string | Introduction of the subject's English Wikipedia article (at most 800 characters); it may not state the answer. Used only by the Objective 2 knowledge intervention. | `Isaac Bledsoe Desha (January 1, 1802 – August 13, 1828) was…` |
| `hint` | null | Always null: this set has no hint. | null |
| `meta` | object | Extra fields kept from the source dataset (for strata and analysis). | (see below) |
| `meta.s_pop` | integer, 16 to 554097 | Popularity of the subject: monthly Wikipedia page views, from PopQA. | `98` |
| `meta.pop_bin` | integer, 0 to 4 | Popularity bin: quintile of log10(s_pop), 0 = rarest, 4 = most popular. Used to stratify the sample. | `0` |
| `meta.prop` | string (15 fixed values) | The Wikidata relation the question asks about. | `father` |
| `meta.subject` | string | The subject entity of the question. | `Isaac B. Desha` |
| `meta.wiki_title` | string | English Wikipedia title of the subject (used to fetch the evidence). | `Isaac B. Desha` |

## Statistics

| | All (extended tier) | Core tier |
|---|---:|---:|
| Questions | 300 | 150 |
| Dev | 90 | 48 |
| Test | 210 | 102 |
| With evidence | 300 | 150 |
| With hint | 0 | 0 |

| Length (min / median / max) | |
|---|---:|
| Question, characters | 17 / 34 / 79 |
| Accepted answers per question | 1 / 2 / 190 |
| Evidence, characters | 18 / 537 / 800 |

Subject popularity (`s_pop`, monthly page views): min / median / max = 16 / 948.5 / 554,097.

**Popularity bin (0 = rarest)**

| Bin | Count | Share |
|---|---:|---:|
| 0 | 60 | 20% |
| 1 | 60 | 20% |
| 2 | 60 | 20% |
| 3 | 60 | 20% |
| 4 | 60 | 20% |

**Relation asked about**

| Relation | Count | Share |
|---|---:|---:|
| genre | 42 | 14% |
| director | 41 | 14% |
| screenwriter | 34 | 11% |
| author | 33 | 11% |
| producer | 29 | 10% |
| composer | 20 | 7% |
| place of birth | 18 | 6% |
| father | 17 | 6% |
| country | 15 | 5% |
| capital of | 11 | 4% |
| sport | 11 | 4% |
| capital | 9 | 3% |
| occupation | 9 | 3% |
| religion | 8 | 3% |
| mother | 3 | 1% |

## Example record

A real record from the file:

```json
{
  "id": "popqa-105962",
  "set": "popqa",
  "kind": "short",
  "question": "What is Mamuju the capital of?",
  "answers": [
    "West Sulawesi"
  ],
  "split": "test",
  "core": true,
  "judge_mode": "match",
  "incorrect_answers": [],
  "gold_type": null,
  "evidence": "Mamuju is the capital of the Indonesian province of West Sulawesi. The town was formerly part of South Sulawesi province.",
  "hint": null,
  "meta": {
    "s_pop": 655,
    "pop_bin": 2,
    "prop": "capital of",
    "subject": "Mamuju",
    "wiki_title": "Mamuju (city)"
  }
}
```

## How the study uses it

- **Generation:** every model answers each question once greedily and 10 more times by sampling; the uncertainty methods compare those answers.
- **Grading:** The model's answer is free text. A judge model from another family grades it with the record's `judge_mode`; a second judge from a third family re-grades a sample as a check.
- **Splits:** dev questions tune the thresholds and train the Objective 2 detector and classifier; test questions give every reported number.
- **Objective 2 re-asks:** for the focus models, wrong or unsure answers are asked again (5 samples) with `evidence` (knowledge intervention); whether the answer gets fixed tells which kind of uncertainty it was.

## Caveats

- The questions are made from templates, so a few are ambiguous (several works can share one title).
- The evidence is the subject's article introduction and does not always contain the answer.
- Wikipedia text is licensed CC BY-SA 4.0.

## Reference

A. Mallen, A. Asai, V. Zhong, R. Das, D. Khashabi, H. Hajishirzi. When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories. ACL 2023.
