# evaluation/

Scripts sit at the top level (run as `python -m evaluation.<script>`); everything they
produce or read lives in `results/` and `question_sets/`. The test datasets themselves are
frozen in `data/eval_sets/full/` (see its `MANIFEST.json`).

## Scripts

| Script | What it does |
|---|---|
| `benchmark_runner.py` | GSM8K / TruthfulQA, raw vs framework, paired; writes one folder per run |
| `hard_eval.py` | Hard sets, difficulty ladder, MATH-500 / MATH / JEEBench: `collect`, `score`, `compare` |
| `smoke_general.py` | 16 basic web and maths questions against the live API |
| `mcq_10th.py` | 15 10th-standard competitive-exam MCQs |
| `model_compare.py` | Raw accuracy of candidate pipeline models (to choose the model) |
| `judge_agreement.py` | Re-grades TruthfulQA answers with a second judge (judge independence) |
| `paper_results.py` | Recomputes every headline number from `results/` → `results/paper_results.json` |
| `export_eval_sets.py` | Rebuilds `data/eval_sets/full/` at pinned dataset versions |
| `hard_sets.py`, `ladder_sets.py` | Question sets written / sampled for this project |
| `metrics.py`, `truthfulqa_judge.py` | Scoring helpers (statistics, LLM judge) |
| `presentation_plots.py`, `build_presentation.py`, `thesis_plots.py`, `plot_generator.py` | Figures and slides (Phase 1 era) |

## results/

| Folder / file | Contents |
|---|---|
| `test_qwen3b_v6/` | **Final GSM8K test** (items 0–99), qwen2.5:3b, corrected scoring: 84% → 88% |
| `test_qwen3b_v5/` | First qwen2.5:3b test run: GSM8K and **TruthfulQA test** (items 0–99) |
| `hard_sets_and_ladder/` | Hard sets (`hard_*`) and difficulty ladder (`ladder_*`): answers (`.jsonl`), scores (`_scored.json`), yes/no judge cache |
| `dev/` | Design runs on dev items 500–539: `pilot`, `pilot_v1`, `llama2_self*` (Phase 1), `qwen3b_v1`–`v4` (qwen design choices) |
| `phase1_llama2/` | Phase 1: llama2:7b, raw vs framework (TruthfulQA +18 points) |
| `phase1_llama2_all_configs/` | Phase 1 with the extra reference-verifier config |
| `judge_check/` | Phase 1 TruthfulQA re-graded by a second judge |
| `run_logs/` | Console logs of runs, including all smoke tests |
| `plots/`, `legacy/` | Old figures and Phase 1-era CSVs |
| `mcq_10th.json` | 10th-standard MCQ answers and scores |
| `paper_results.json` | All headline numbers (from `paper_results.py`) |

Each benchmark run folder holds `results_<dataset>.csv` (one row per question and config:
prompt, answer, gold, correctness, latency, tokens, route), `*_audit.jsonl` (full pipeline
trace), `summary_<dataset>.csv` (accuracy, CI, McNemar) and `routing_*.csv`.

## question_sets/

`ladder_items.json`: the 80 sampled difficulty-ladder questions (fixed seed).
