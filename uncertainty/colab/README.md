# Running the uncertainty study on Google Colab

A quicker version of the full Mac study, for analysis: **all 12 models on all 11 datasets**, with fewer
questions per dataset. It runs the same code (`python -m uncertainty run-all`) with a few flags, so the
reports, tables and figures are the same kind as the full run's.

## How to run it

1. Open [uncertainty_colab.ipynb](uncertainty_colab.ipynb) in Colab: *File → Open notebook → GitHub*, paste
   `Hemant-Shashikant-Yadav/LLM-Hallucination`, and pick `uncertainty/colab/uncertainty_colab.ipynb`.
2. *Runtime → Change runtime type → GPU* (T4 works; L4 or A100 is faster).
3. Run the cells from top to bottom. Step 6 runs the study for hours. Keep the tab open.

**If Colab disconnects** (free sessions end after some hours), reconnect and run every cell again. Results
are written to your Google Drive (`MyDrive/LLM-Hallucination/uq_results`), so finished work is kept and the
run continues where it stopped. The models are downloaded again (about 10 minutes).

## How many questions

`PER_SET` (cell 1) is the number of questions per dataset, picked evenly through each dataset, so every kind
of question (and FreshQA's false premises as well as its changing facts) stays in.

| `PER_SET` | Questions per model | Free T4 | L4 | A100 |
|---|---|---|---|---|
| **20 (default)** | 220 | ~10 h | ~5–6 h | ~3 h |
| 30 | 330 | ~15 h | ~8 h | ~4–5 h |

The times are rough estimates. The pilot stage measures the real speed after about 15 minutes; the projection
is in `run.log` and in `reports/SUMMARY.md` (column *Projected h*).

**Why 20 is the default.** It is the fastest setting that still gives every model about 150 test questions
(70% of 220; the other 30% tune the thresholds), so each model's overall AUROC has a confidence interval of
roughly ±0.08. Per-dataset numbers rest on about 14 test questions each, so treat them as indicative. For
firmer per-dataset numbers use 30 or more, or the full Mac run.

## What is different from the Mac run

| | Mac (full study) | Colab (this notebook) |
|---|---|---|
| Questions | 3,367 for the 3 focus models, 1,410 for the others | `PER_SET` × 11 for every model (same questions for all) |
| Models | picked from the Mac's memory (9 or 12) | the 12 listed in the notebook (`--models`) |
| Judges | by memory class | `--memory-gb 16`: llama3.1:8b, or qwen2.5:7b for the llama models; a third family checks a sample |
| Second-judge sample | 300 answers per model | 50 per model |
| Everything else | samples, methods, metrics, Objective 2 re-asks and reports | the same |

Why `--memory-gb 16` and `--models`: on Linux the run reads system RAM to pick models, and Colab has only about
13 GB of it (the GPU memory is separate). With no model fitting, the run stops and says so; naming the models
avoids that. The 14B models fit on a 16 GB T4.

## After the run

- Read `reports/SUMMARY.md` (shown in cell 7), `reports/obj1/obj1.md` and `reports/obj2/obj2.md`.
- To bring the results into the repository, copy the Drive folder `uq_results` into `uncertainty/results/`
  (or download the zip from cell 8). The deck's result slides can then be rebuilt from the measured numbers
  (see `uncertainty/presentation/build_phase1_deck.py`: put them in `EXP` and set `EXPECTED = False`).

## Troubleshooting

| What you see | What to do |
|---|---|
| `No roster model fits ... GB of memory` | run cell 1 again: the study cell must pass `--models` |
| `could not pull <model>` in the summary | run the study cell again (it retries); check the disk with cell 2 |
| Ollama `NOT RUNNING` in cell 5 | read `/content/ollama.log`; run cell 5 again |
| `Unit tests failed` | open `uq_results/unit_tests.log`; usually a missing package: add it to the `pip install` line in cell 4 |
| The session ended | reconnect and run all cells again; the run resumes |
