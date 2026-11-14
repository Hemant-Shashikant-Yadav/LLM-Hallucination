# uncertainty/ — Objectives 1 and 2

Thesis objectives 1 and 2, separate from the hallucination-mitigation pipeline (`app/`):

1. **Measure an AI model's baseline uncertainty using an existing algorithm.** Semantic
   entropy (Farquhar et al., 2024) and 16 comparison methods, on 11 question sets and up to
   12 raw models.
2. **Identify and classify uncertainty in AI-generated responses.** A detector (how likely
   is this answer a hallucination?) and a classifier (knowledge, reasoning, aleatoric or
   temporal uncertainty, and what would fix it), plus a tool that reports both for any
   question.

**Running it on the Mac:** [HANDOFF.md](HANDOFF.md) is a ready-to-use prompt covering setup,
the run, troubleshooting, the write-up and a results deck of at most 10 slides.

**A quicker run on Google Colab:** [colab/README.md](colab/README.md) and its notebook run all 12 models
on all 11 datasets with fewer questions per dataset (20 by default, about 10 hours on a free T4 GPU).

The design and its reasons are in [PLAN.md](PLAN.md). Nothing here changes the pipeline,
and the pipeline reads nothing from here. Read-only imports: `app.answer_utils` (answer
reading) and `evaluation.truthfulqa_judge` (the TruthfulQA judge prompt and ROUGE-L).

## Run it

**On the Mac (the real study):**

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

```bash
caffeinate -i .venv/bin/python -m uncertainty run-all
```

`run-all` reads the Mac's memory and picks the models, pulls any that are missing, checks
the server's log-probs, runs the unit tests, runs a pilot, then every model in turn. It
resumes where it stopped: run the same command again after a crash, a reboot or a fix.
Progress: `uncertainty/results/run.log`; status: `uncertainty/results/reports/SUMMARY.md`.

Useful options: `--max-hours N` (trims itself to fit), `--models a b c` (override the
roster), `--pipeline-model TAG` (if the pipeline moved to another model), `--until STAGE`.

**On any machine, without a model (dry run with the simulated model):**

```bash
python -m uncertainty run-all --backend fake --limit 24
```

Writes to `uncertainty/results_fake/` (not committed). `--limit` keeps n questions per set,
spread evenly through it. The simulated model plants uncertainty types, so its numbers only
show that the code works, never anything about real models.

Gates: worked answers must be readable in at least 90% of samples and at most 10% of samples
may be cut off; judge failures above 2% are recorded as a warning, above 20% (a broken judge)
fail the gate.

**Tests:**

```bash
python -m unittest discover -s uncertainty/tests -t .
```

## After the run: human labels (the one manual step, no model needed)

```bash
python -m uncertainty annotate export
```

Fill in `results/annotation/labels_to_fill.csv` with the guide beside it
(`annotation_guide.md`), one copy per annotator, then:

```bash
python -m uncertainty annotate import --files labels_A.csv labels_B.csv
```

```bash
python -m uncertainty report all
```

The judge audit works the same way: `annotate export-audit`, fill the `correct` column,
`annotate import-audit --files judge_audit_filled.csv`.

## One question at a time

```bash
python -m uncertainty analyze "Who was the first person to walk on the Moon?" --model qwen2.5:3b
```

Prints the answer, the hallucination probability and level, the uncertainty type (and
subtype), the signals behind them, and the recommended action. `--mode lite` uses the
pipeline's budget (4 samples, no extra calls).

## Stages of `run-all`

| Stage | What happens | Stops the run when |
|---|---|---|
| check | memory → roster; pull models; Ollama version; log-prob check; unit tests (~5 min) | Ollama unreachable, another job's model loaded, tests fail |
| pilot | 5 dev questions per set on `qwen2.5:3b` through every later step; speed → projected hours | answers unreadable, too many cut off, judge broken |
| focus | `qwen2.5:3b`, `llama3.1:8b`, `gemma3:4b` (or `--pipeline-model`): extended tier, NLI, labels, scores, interventions | a per-model gate fails: that model is **skipped** and reported, the run goes on; it is tried again the next time you run `run-all` |
| first_reports | Objective 1 and 2 reports from the focus models | — |
| others | every other model, core tier, smallest first | (as focus) |
| final_reports | both reports and `SUMMARY.md` | — |

## Folder layout

| Path | Contents |
|---|---|
| `settings.py`, `roster.py` | settings (`UQ_` prefix, optional `uq.env`); models, judges, focus models |
| `sets/` | `build.py` freezes the question sets into `data/uq_sets/` (pinned sources, manifest); `load.py` reads them |
| `generate/` | prompts; Ollama and simulated backends; the runner (greedy + K samples + self-evaluation + probes) |
| `methods/` | Objective 1 methods: semantic entropy, token-level, spectral, consistency, self-evaluation; NLI |
| `label/` | correctness (rules + cross-family judge), interventions, human annotation |
| `score.py`, `metrics.py`, `thresholds.py` | scores per question; AUROC & co.; dev-tuned thresholds |
| `framework/` | Objective 2: taxonomy, features, detector, classifier, `analyze` tool |
| `report/` | Objective 1 and 2 reports (tables, LaTeX, figures) |
| `orchestrate/` | `run-all`, gates, machine checks |
| `tests/` | unit tests and an end-to-end dry run |
| `results/` | the Mac run's output (committed); `results_fake/` for dry runs (ignored) |

Inside `results/`: one folder per model (`gen/*.jsonl.gz` generations, `nli/*.npz`,
`labels.jsonl`, `scores.csv`, `thresholds.json`, `interventions.jsonl`), `models/` (the
trained framework), `reports/obj1`, `reports/obj2`, `reports/SUMMARY.md`, `run_state.json`.

Each source file has a note beside it (`file.py` → `file.README.md`).
