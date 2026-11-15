# StrategyQA: multi-step commonsense

`data/uq_sets/strategyqa.jsonl` · 250 questions · schema: [`strategyqa.schema.json`](../schema/strategyqa.schema.json)

> Yes/no questions that need several hidden reasoning steps, with facts and a step list.

## At a glance

| | |
|---|---|
| Domain | Multi-step commonsense |
| Answer kind | `yesno`: yes or no, read and graded by rules |
| Questions | 250 (extended tier); 120 in the core tier |
| Dev / test | 75 / 175 |
| With evidence / with hint | 250 / 248 |
| Gold uncertainty types | none |
| Source | ChilleD/StrategyQA (test) + official facts/decompositions from voidful/StrategyQA strategyqa_train.json ([link](https://huggingface.co/datasets/ChilleD/StrategyQA)) |
| Pinned revision | `705562638fe1d8ca6bb98c66fc8f94d45fda8c83 + 2279eaf9f2580aef77ed6fa0efd7846c381ab5a0` |
| Licence | MIT |
| Built | 2026-11-09, by `python -m uncertainty build-sets --only strategyqa` |
| SHA-256 | `a54617655f3711482c2a866776248dbdd792e45de7941c60efd91d26c79c14a4` (matches the file) |

## What the dataset is about

StrategyQA has 2,780 yes/no questions that need implicit multi-step reasoning, such as 'Did Aristotle use a laptop?'. Each question comes with a decomposition into reasoning steps and supporting facts from Wikipedia.

## Why it is in this study

Commonsense reasoning that combines several facts, so it mixes knowledge and reasoning. With only two possible answers, it also tests how entropy behaves when samples can split only two ways.

## How this file was made

Built by `uncertainty/sets/build.py` (`build_strategyqa`). Every random choice is seeded (seed 7), so a rebuild gives the same file.

1. Questions and answers come from `ChilleD/StrategyQA` (test split, pinned revision).
2. Each is matched by `qid` with the official `strategyqa_train.json` (from `voidful/StrategyQA`) to get its facts and decomposition.
3. Draw 250 so that yes and no stay balanced (seeded); the core tier keeps the balance too.
4. Evidence: the official facts, joined.
5. Hint: 'Break the question into steps: (1) ... (2) ...' from the decomposition; null when it has fewer than 2 steps.
6. Mark the core tier (120 questions) and the fixed dev (30%) / test (70%) split, both seeded per set.

## What each record contains

Each line of the file is one JSON object with these fields (the full types are in the schema):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | string | Unique question id in this set: 'sqa-' + the source dataset's own id. | `sqa-d20362599347f39a08e5` |
| `set` | string (always `strategyqa`) | Name of the question set: always 'strategyqa'. | `strategyqa` |
| `kind` | string (always `yesno`) | Answer kind 'yesno': yes or no, read and graded by rules. | `yesno` |
| `question` | string | The question as sent to the model. | `Would the Ku Klux Klan welcome Opal Tometi into their group?` |
| `answers` | array of strings, 1 to 1 items | 'yes' or 'no' (one item). | `["no"]` |
| `split` | string (`dev`, `test`) | Fixed, seeded split: dev (30%) only tunes thresholds and trains the Objective 2 models; every reported number uses test (70%). | `test` |
| `core` | boolean | true if the question is in the core tier, the subset every model answers; the file itself is the extended tier. | `false` |
| `judge_mode` | string (`match`) | How the LLM judge grades a free-text answer: match (agrees with a reference answer), truthfulqa (TruthfulQA judge prompt), abstain (correct only if it says the question cannot be answered), premise (correct only if it rejects the false premise). Unused for rule-graded kinds. | `match` |
| `incorrect_answers` | array (always empty) | Known false answers from the dataset (TruthfulQA only; empty elsewhere). | `[]` |
| `gold_type` | null | Always null: the dataset gives no uncertainty type for these questions. | null |
| `evidence` | string | The official supporting facts for the question, joined. Used only by the Objective 2 knowledge intervention. | `The Ku Klux Klan is an American white supremacist hate grou…` |
| `hint` | string or null | 'Break the question into steps: (1) ... (2) ...' from the official decomposition; null when it has fewer than 2 steps. Used only by the Objective 2 reasoning intervention. | `Break the question into steps: (1) Which groups of people a…` |
| `meta` | object | Extra fields kept from the source dataset (for strata and analysis). | (see below) |
| `meta.term` | string | The Wikipedia term the question was written from. | `Ku Klux Klan` |

## Statistics

| | All (extended tier) | Core tier |
|---|---:|---:|
| Questions | 250 | 120 |
| Dev | 75 | 40 |
| Test | 175 | 80 |
| With evidence | 250 | 120 |
| With hint | 248 | 118 |

| Length (min / median / max) | |
|---|---:|
| Question, characters | 25 / 55 / 102 |
| Accepted answers per question | 1 / 1 / 1 |
| Evidence, characters | 60 / 158 / 573 |
| Hint, characters | 70 / 159 / 337 |

**Correct answer**

| Answer | Count | Share |
|---|---:|---:|
| no | 125 | 50% |
| yes | 125 | 50% |

## Example record

A real record from the file:

```json
{
  "id": "sqa-b8de21dae500bfa66b7b",
  "set": "strategyqa",
  "kind": "yesno",
  "question": "Does water have viscosity?",
  "answers": [
    "no"
  ],
  "split": "test",
  "core": true,
  "judge_mode": "match",
  "incorrect_answers": [],
  "gold_type": null,
  "evidence": "Viscosity is resistance of fluid to deformation. Water is not resistant to deformation.",
  "hint": "Break the question into steps: (1) What is viscosity? (2) Is water #1?",
  "meta": {
    "term": "Viscosity"
  }
}
```

## How the study uses it

- **Generation:** every model answers each question once greedily and 10 more times by sampling; the uncertainty methods compare those answers.
- **Grading:** The yes/no verdict is read from the model's answer and compared with `answers[0]` by rules; only an answer that never says yes or no goes to the judge.
- **Splits:** dev questions tune the thresholds and train the Objective 2 detector and classifier; test questions give every reported number.
- **Objective 2 re-asks:** for the focus models, wrong or unsure answers are asked again (5 samples) with `evidence` (knowledge intervention) and `hint` (reasoning intervention); whether the answer gets fixed tells which kind of uncertainty it was.

## Caveats

- The facts often state the answer almost directly, so the evidence is close to an oracle.

## Reference

M. Geva, D. Khashabi, E. Segal, T. Khot, D. Roth, J. Berant. Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies. TACL 2021.
