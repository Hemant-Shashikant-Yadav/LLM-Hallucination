# data/: the datasets of this project

Every dataset the project uses is frozen here as a JSON Lines file (one JSON object per line), with
its source, pinned revision, licence and SHA-256 recorded in a `MANIFEST.json` beside it. Each dataset
also has:

- **a schema** (`schema/<name>.schema.json`): a [JSON Schema](https://json-schema.org/) (draft 2020-12)
  giving every field's data type, allowed values and meaning, with a real example record;
- **a guide** (`docs/<name>.md`): what the dataset is about, where it comes from, how the file was made,
  what each field holds, statistics, an example record, how the project uses it, and its reference.

The schemas and guides are **documentation only**: no code reads them, and the data files are unchanged.
Every record of every file was checked against its schema when the schemas were written.

```
data/
├── uq_sets/            the 11 question sets of the uncertainty study (thesis Objectives 1-2)
│   ├── <name>.jsonl    the data
│   ├── MANIFEST.json   sources, revisions, licences, counts, SHA-256
│   ├── schema/         one JSON Schema per set
│   └── docs/           one guide per set
├── eval_sets/full/     full test sets of the hallucination-mitigation pipeline (Objectives 3-4)
│   ├── <name>.jsonl, MANIFEST.json, schema/, docs/   (same layout)
└── chromadb/           the pipeline's local ChromaDB store, searched by Layer 2 next to the web
                        (a database, not a question set)
```

## Uncertainty study: `uq_sets/` (Objectives 1-2)

Built by `python -m uncertainty build-sets` (`uncertainty/sets/build.py`); read by `uncertainty/sets/load.py`.
Each file is the extended tier; `core` marks the subset every model answers, and `split` the fixed dev
(30%) / test (70%) assignment.

| Dataset | What it tests | Answer kind | Questions (core) | Guide | Schema |
|---|---|---|---:|---|---|
| TriviaQA | popular facts | short | 300 (150) | [triviaqa.md](uq_sets/docs/triviaqa.md) | [schema](uq_sets/schema/triviaqa.schema.json) |
| PopQA | long-tail facts | short | 300 (150) | [popqa.md](uq_sets/docs/popqa.md) | [schema](uq_sets/schema/popqa.schema.json) |
| HaluEval (QA) | multi-hop facts | short | 300 (150) | [halueval.md](uq_sets/docs/halueval.md) | [schema](uq_sets/schema/halueval.schema.json) |
| TruthfulQA | misconceptions | short | 817 (150) | [truthfulqa.md](uq_sets/docs/truthfulqa.md) | [schema](uq_sets/schema/truthfulqa.schema.json) |
| GSM8K | arithmetic word problems | number | 250 (120) | [gsm8k.md](uq_sets/docs/gsm8k.md) | [schema](uq_sets/schema/gsm8k.schema.json) |
| MATH | competition maths | expr | 250 (120) | [math.md](uq_sets/docs/math.md) | [schema](uq_sets/schema/math.schema.json) |
| StrategyQA | multi-step commonsense | yesno | 250 (120) | [strategyqa.md](uq_sets/docs/strategyqa.md) | [schema](uq_sets/schema/strategyqa.schema.json) |
| ARC-Challenge | science multiple choice | choice | 200 (100) | [arc.md](uq_sets/docs/arc.md) | [schema](uq_sets/schema/arc.schema.json) |
| AmbigQA | ambiguous questions | short | 200 (100) | [ambigqa.md](uq_sets/docs/ambigqa.md) | [schema](uq_sets/schema/ambigqa.schema.json) |
| SelfAware | unanswerable questions | short | 300 (150) | [selfaware.md](uq_sets/docs/selfaware.md) | [schema](uq_sets/schema/selfaware.schema.json) |
| FreshQA | changing facts, false premises | short | 200 (100) | [freshqa.md](uq_sets/docs/freshqa.md) | [schema](uq_sets/schema/freshqa.schema.json) |
| **Total** | | | **3,367 (1,410)** | | |

All 11 files share one record layout (the `Item` dataclass in `uncertainty/sets/load.py`):

| Field | Type | Meaning |
|---|---|---|
| `id` | string | unique question id |
| `set` | string | set name |
| `kind` | string | answer kind: `short` (free text, LLM judge), `number`, `expr`, `choice`, `yesno` (read by rules) |
| `question` | string | the question as sent to the model |
| `answers` | array of strings | accepted answers (empty for unanswerable questions) |
| `split` | string | `dev` or `test` |
| `core` | boolean | in the core tier |
| `judge_mode` | string | how a free-text answer is graded: `match`, `truthfulqa`, `abstain`, `premise` |
| `incorrect_answers` | array of strings | known false answers (TruthfulQA only) |
| `gold_type` | string or null | uncertainty type given by the dataset: `ambiguous`, `unanswerable`, `false_premise`, `temporal` |
| `evidence` | string or null | oracle evidence for the knowledge re-ask (Objective 2) |
| `hint` | string or null | partial method for the reasoning re-ask (Objective 2) |
| `meta` | object | extra source fields; different per set (see each schema) |

## Pipeline evaluation: `eval_sets/full/` (Objectives 3-4)

Written by `python -m evaluation.export_eval_sets`. Whole test splits, kept for reviewers and reruns;
the reported pipeline results used only the slices named in each guide.

| Dataset | What it tests | Records | Guide | Schema |
|---|---|---:|---|---|
| GSM8K test | arithmetic word problems | 1,319 | [gsm8k_test.md](eval_sets/full/docs/gsm8k_test.md) | [schema](eval_sets/full/schema/gsm8k_test.schema.json) |
| TruthfulQA | misconceptions | 817 | [truthfulqa.md](eval_sets/full/docs/truthfulqa.md) | [schema](eval_sets/full/schema/truthfulqa.schema.json) |
| MATH test | competition maths | 5,000 | [math_test.md](eval_sets/full/docs/math_test.md) | [schema](eval_sets/full/schema/math_test.schema.json) |
| MATH-500 | competition maths (standard subset) | 500 | [math500.md](eval_sets/full/docs/math500.md) | [schema](eval_sets/full/schema/math500.schema.json) |
| JEEBench | JEE Advanced physics, chemistry, maths | 515 | [jeebench.md](eval_sets/full/docs/jeebench.md) | [schema](eval_sets/full/schema/jeebench.schema.json) |

## Checking a file against its schema (optional)

```bash
python -c "import json; from jsonschema import Draft202012Validator as V; v = V(json.load(open('data/uq_sets/schema/gsm8k.schema.json', encoding='utf-8'))); bad = [e for l in open('data/uq_sets/gsm8k.jsonl', encoding='utf-8') for e in v.iter_errors(json.loads(l))]; print('valid' if not bad else bad[0].message)"
```
