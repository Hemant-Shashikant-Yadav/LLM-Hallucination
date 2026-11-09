# Handoff: run the uncertainty study on the Mac, then write it up

**How to use this file:** open Claude Code in this repository on the Mac and say
*"Read `uncertainty/HANDOFF.md` and do everything in it."* It is written as the full prompt:
context, rules, every step from setup to the results deck, and what to report back.

---

## 1. Context in short

- **Thesis:** reduce LLM hallucinations. Four objectives: (1) measure a model's baseline
  uncertainty with an existing algorithm; (2) a framework that identifies and classifies
  uncertainty in AI answers; (3) evidence-based verification and structured reasoning;
  (4) compare on faithfulness, hallucination and cost.
- **Objectives 3–4** are the pipeline in `app/` and `evaluation/` (its own handoff:
  [../HANDOFF.md](../HANDOFF.md)). **Objectives 1–2** are this package, `uncertainty/`,
  built separately on purpose: it never changes the pipeline and the pipeline reads nothing
  from it.
- **Built on the Windows PC (2026-11-09, `main` at `77cac4a`), not yet run on a real model.**
  45 unit tests pass; a 12-model dry run with a simulated model (`--backend fake`) produced
  every report; a simulated crash resumed correctly. The simulated numbers mean nothing.
- **What the run produces:**
  - *Objective 1:* semantic entropy (Farquhar et al., 2024) plus 16 comparison methods on
    11 question sets (3,367 questions; 1,410 core) and up to 12 models. AUROC with CIs,
    DeLong tests, calibration, the blind spot (confident errors), AUROC vs samples, size and
    family effects, cost.
  - *Objective 2:* a calibrated hallucination probability, and a type classifier
    (knowledge / reasoning / aleatoric / temporal) checked against the pipeline's routing
    rule, plus `python -m uncertainty analyze "question"`.
- **One real number already:** the pipeline's stored Layer 1 entropy predicts wrong answers
  well on GSM8K (AUROC 0.86) but near chance on TruthfulQA (0.55): the blind spot this study
  measures across models.
- **Read for detail:** [README.md](README.md) (commands, outputs), [PLAN.md](PLAN.md) (design,
  every "Built:" deviation), and the note beside each file (`file.py` → `file.README.md`).

## 2. Rules

1. Do not edit `app/`, `evaluation/`, `tests/`, `doc/`, `.env` or `requirements.txt`. Any fix
   goes inside `uncertainty/`, with a unit test, and is described in PLAN.md as "Built:".
2. One Ollama model loaded at a time; never run this next to a pipeline benchmark.
3. Judges stay cross-family (the roster does this); never let a model judge itself.
4. Thresholds and model choices come from the dev split only; report the test split.
5. Never report numbers from `--backend fake` or `results_fake/`; never invent results or
   human labels. Every number in the write-up and the deck comes from `uncertainty/results/`.
6. Commits: code fixes, results data and write-up in separate commits; push to `main` at the
   end (step 8). Keep each commit's message clear about what it contains.

## 3. Steps

### Step 0. Set up (once)

```bash
sysctl -n machdep.cpu.brand_string hw.memsize   # memory decides the roster: 16 GB → 9 models, 24 GB+ → 12
git pull origin main
python3 --version                                # 3.11 or newer
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install python-pptx                # for the deck only; not added to requirements.txt
ollama --version                                 # a recent Ollama (token log-probs are needed)
```

Start the server in its own terminal and leave it running:

```bash
OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

`ollama ps` must show no model loaded by another job. If the pipeline was moved to a new
model on this Mac (../HANDOFF.md step 3), note its tag for `--pipeline-model` below.

### Step 1. Run everything (about a week, unattended)

```bash
caffeinate -i .venv/bin/python -m uncertainty run-all
```

Add `--pipeline-model <tag>` if the pipeline uses another model; add `--max-hours N` if the
run must finish within N hours (it trims itself and says how). Run it in the background and
follow `uncertainty/results/run.log` and `uncertainty/results/reports/SUMMARY.md`; check every
few hours rather than polling. Stages: check (pulls models, tests the server's log-probs, runs
the unit tests, ~5 min) → pilot (5 questions per set; prints the projected hours) → focus
models → first reports → other models → final reports. Reports are rebuilt after every model.

### Step 2. If it stops

Run the same command again after any fix: finished work is kept and only the rest runs.

| Message | What to do |
|---|---|
| another model is loaded | wait for that job, or `ollama stop <model>`, then re-run |
| unit tests failed | read `results/unit_tests.log`; fix inside `uncertainty/` |
| pilot: answers unreadable / too many cut off | open `results/_pilot/<model>/gen/*.partial.jsonl` or `.jsonl.gz`, see what the model writes, fix the prompt or reader in `uncertainty/`, add a test |
| pilot: judge broken | `ollama pull` the judge named in the message |
| could not pull a model | the run skips it (see SUMMARY.md); pull by hand and re-run to include it |
| a model skipped by a gate | read SUMMARY.md "Degradations"; fix the cause; re-run (it is retried) |
| log-prob mode `topk1` or `none` | not an error: noted in SUMMARY.md; token-level methods adapt or switch off |
| Ollama crashed / Mac rebooted | restart `ollama serve`, re-run |

### Step 3. Check the results are believable

- SUMMARY.md: every roster model `done`; read every degradation and explain it in the write-up.
- `reports/obj1/judge_agreement.csv`: second-judge agreement and κ (flag κ below 0.4).
- `reports/obj1/obj1.md`: `qwen2.5:3b` GSM8K accuracy should be near the pipeline's raw 84%;
  semantic entropy AUROC above 0.5 on most sets.
- `reports/obj2/obj2.md`: label counts per type (temporal and aleatoric come from metadata,
  knowledge and reasoning from interventions); whether the classifier beats the pipeline rule.

### Step 4. Human labels (needs the user; do not do it for them)

```bash
.venv/bin/python -m uncertainty annotate export
.venv/bin/python -m uncertainty annotate export-audit
```

Ask the user to fill `results/annotation/labels_to_fill.csv` (guide beside it; ~2–3 h) and
`judge_audit_to_fill.csv` (100 rows). When they return the files:

```bash
.venv/bin/python -m uncertainty annotate import --files <labels_A.csv> [<labels_B.csv>]
.venv/bin/python -m uncertainty annotate import-audit --files <judge_audit_filled.csv>
.venv/bin/python -m uncertainty report all
```

The write-up and the deck can be made before this, marked "human validation pending", and
rebuilt after.

### Step 5. Write-up

- `uncertainty/results/RESULTS.md` (2–3 pages): setup (machine, Ollama version, models,
  sets, log-prob mode, trims); Objective 1 headline table and findings (best methods, blind
  spot, size and family effects, samples needed vs the pipeline's K = 4); Objective 2 headline
  (detector vs semantic entropy, classifier vs pipeline rule with CI, actionability,
  generalisation, human agreement); limitations; what it means for Objectives 3–4.
- `uncertainty/README.md`: add a short "Results" section linking RESULTS.md, the reports and
  the deck.

### Step 6. Results deck: at most 10 slides, with diagrams

Save as `uncertainty/results/reports/Uncertainty_Study_Results.pptx`, built by a script kept
in the repo (`uncertainty/report/deck.py`, so it can be rebuilt after human labels). Use the
pptx skill if this Claude Code has one, otherwise python-pptx. Optional style reference: the
colours of `../Hallucination_Mitigation_Presentation_Beautified.pptx`.

| # | Slide | Content |
|---|---|---|
| 1 | Title | Study name, date, one-sentence takeaway |
| 2 | Objectives | Diagram: the four objectives, which this study covers, how it feeds 3–4 |
| 3 | How the study works | Flow diagram: questions → models → greedy + 10 samples → 17 methods / labels → reports |
| 4 | Data and models | 11 sets by uncertainty type; roster as size ladders per family |
| 5 | Objective 1: who predicts mistakes best | `obj1/figures/fig_auroc_heatmap.png` + 2–3 bullet findings |
| 6 | Objective 1: the blind spot and model size | blind-spot numbers + `fig_size_effects.png` |
| 7 | Objective 1: how many samples, at what cost | `fig_auroc_vs_k.png`, `fig_auroc_vs_cost_*.png`; verdict on the pipeline's K = 4 |
| 8 | Objective 2: the framework | Taxonomy diagram (type → signature → action) + detector vs semantic entropy |
| 9 | Objective 2: does classification work | `obj2/figures/fig_confusion.png` or `fig_per_class_f1.png`; vs pipeline rule; actionability |
| 10 | Conclusions | 3 findings, limitations, next steps for Objectives 3–4 |

Draw diagrams with native shapes (boxes and arrows), at most ~40 words of text per slide,
numbers only from `reports/*/summary.json` and the CSVs. Then render the deck (e.g.
`soffice --headless --convert-to pdf`) and look at every slide for overflow or overlap before
calling it done.

### Step 7. Commit and push

```bash
du -sh uncertainty/results                       # if over ~1 GB, see PLAN.md decision 4
```

Commit in this order, each with a clear message: (a) any code fixes in `uncertainty/`;
(b) `uncertainty/results/` (generations, labels, scores, models, reports); (c) RESULTS.md,
README update, the deck and `deck.py`. Then `git push origin main`.

### Step 8. Report back to the user

A short message: models run and hours taken; Objective 1 headline (AUROC of semantic entropy
per model, the best method, the blind spot); Objective 2 headline (detector, classifier vs
pipeline rule with CI); anything skipped or degraded; links to RESULTS.md, the deck,
SUMMARY.md; and whether human labels are still pending.

## 4. Where everything is

| Path | What |
|---|---|
| `uncertainty/README.md`, `PLAN.md`, `*.README.md` | how to run; design and deviations; per-file notes |
| `data/uq_sets/` + `MANIFEST.json` | the frozen question sets (pinned sources, SHA-256) |
| `uncertainty/results/<model>/` | `gen/*.jsonl.gz`, `nli/`, `labels.jsonl`, `scores.csv`, `thresholds.json`, `interventions.jsonl` |
| `uncertainty/results/reports/SUMMARY.md` | run status, headline numbers, every degradation |
| `uncertainty/results/reports/obj1/` | `obj1.md`, CSV tables, `method_table_all.tex`, `figures/` |
| `uncertainty/results/reports/obj2/` | `obj2.md`, `summary.json`, rules, examples, `figures/` |
| `uncertainty/results/models/` | the trained framework used by `analyze` |
| `uncertainty/results/run.log`, `run_state.json` | progress log; stage and model status |

Budget: ~1.4 M generated tokens per model (core), ~3.2 M (extended, the three focus models),
about 20 M in all: roughly a week. The pilot replaces this with a measured projection.
