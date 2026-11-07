# 3. Evaluation: data, scoring, reproduction

## Test sets

All frozen in `data/eval_sets/full/` with pinned Hugging Face versions and SHA-256
checksums (`MANIFEST.json`; rebuild with `python -m evaluation.export_eval_sets`).

| Set | Size | Source | Used so far |
|---|---|---|---|
| GSM8K test | 1,319 | `openai/gsm8k` (MIT) | items 0–99 (test), 500–539 (dev) |
| TruthfulQA | 817 | `truthfulqa/truthful_qa` (Apache-2.0) | items 0–99 (test), 500–539 (dev) |
| MATH test | 5,000 | `EleutherAI/hendrycks_math` (MIT) | 40 via the ladder |
| MATH-500 | 500 | `HuggingFaceH4/MATH-500` | not yet |
| JEEBench (JEE Advanced) | 515 | `daman1209arora/jeebench` (MIT) | not yet |
| Hard sets (12th maths, JEE-style MCQ, real-life, reasoning, hard maths) | 74 | written for this project, keys checked with sympy or brute force | all |
| Difficulty ladder (MATH L1–5, ARC-Easy/Challenge, StrategyQA, MMLU logic) | 80 | sampled, seed 7 | all |
| 10th-standard MCQs (counting, circles) | 15 | written for this project | all |
| Smoke test (basic web and maths) | 16 | written for this project | all |

Dev items (500–539) were used to choose designs and thresholds; test items only to report.

## Scoring

| Answer type | How it is scored |
|---|---|
| GSM8K | Final number (`#### / \boxed{} / Final answer:` marker, else the number the closing sentence states) equals the key |
| Maths value / expression | `answers_equivalent`: numbers within 0.5% relative; expressions equal after simplification; MATH's LaTeX keys parsed; tuples and words compared as text |
| Multiple choice | `extract_choice`: the option picked (a value matching one option wins over a mis-copied letter) |
| JEE multi-correct | exactly the key's set of letters |
| Yes/no | stated yes/no, else the judge model reads the answer (cached) |
| TruthfulQA | LLM judge (`llama3.1:8b`): truthful and informative; ROUGE-L also reported |

The judge differs from the pipeline model and grades raw and pipeline answers identically.
Re-grading the Phase 1 TruthfulQA answers with a second judge kept the same gain
(82% agreement, Cohen's κ = 0.55).

**Statistics:** paired comparison on the same draft; McNemar exact test on fixed vs broken
questions; 95% bootstrap confidence intervals.

**Scorer changes are audited:** after any change to answer reading, every stored run is
re-scored and every verdict flip listed. The GSM8K correction of 6 Nov 2026 flipped 14
verdicts across all runs, all from wrong to right (see `04_results.md`).

## Reproduce

```bash
# 1. Server (pipeline model from .env)
uvicorn app.main:app --port 8000

# 2. GSM8K + TruthfulQA, raw vs framework (always name the model and a new folder)
python -m evaluation.benchmark_runner --n 100 --datasets gsm8k truthfulqa \
    --configs baseline full_framework --generator qwen2.5:3b \
    --judge-model llama3.1:8b --out-dir evaluation/results/<new_run>

# 3. Hard sets, ladder, MATH-500, MATH, JEEBench
python -m evaluation.hard_eval collect --set hard --tag <tag>    # also: ladder, math500, math, jeebench
python -m evaluation.hard_eval score --set hard --tag <tag>

# 4. Basic questions and MCQs
python -m evaluation.smoke_general
python -m evaluation.mcq_10th

# 5. All headline numbers from the stored files (no model calls)
python -m evaluation.paper_results    # → evaluation/results/paper_results.json

# 6. Offline unit tests (72)
python -m unittest discover -s tests -t .
```

Run one model and one benchmark at a time, and check `ollama ps` shows only the pipeline
model before a run. Result folders are described in `evaluation/README.md`.
