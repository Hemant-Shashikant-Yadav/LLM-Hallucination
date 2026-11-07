# Handoff: read this first (written 2026-11-06, from the Windows PC)

Thesis project: a defence-in-depth pipeline that reduces LLM hallucination. Layer 1 (triage) →
Layer 2 (web evidence) or Layer 3 (verified voting). Goal: **beat the raw model**, paired on the
same draft. Work moves to a Mac with more memory, for bigger models and bigger test sets.

## Standing rules (from the user)
1. **One model in every pipeline role** (`.env`: `DID_OLLAMA_MODEL` = `DID_OLLAMA_JUDGE_MODEL`).
   The model may change. The **evaluation judge must be a different model** (now `llama3.1:8b`:
   `benchmark_runner --judge-model`, `evaluation/hard_eval.py` `_JUDGE_MODEL`).
2. Must answer **any basic web or maths question** (smoke list: `doc/IMPROVEMENT_ROADMAP.md` §4).
3. Commits: pre-existing changes and Claude's changes in separate commits. Commit/push only when asked.
4. One model and one benchmark at a time. Before a run, `ollama ps` must show only the pipeline model.

## Code map
`app/router.py` (pipeline, option/yes-no votes) · `app/answer_utils.py` (answer reading, routing
rules, calculator) · `app/layer1_triage/` (semantic entropy, task router) · `app/layer2_nli_rag/`
(DeBERTa NLI, search, reviser) · `app/layer3_halluclean/program_verifier.py` (vote + sandboxed
programs) · `app/config.py` (all settings, env prefix `DID_`) · `evaluation/` (runners, results).
Detailed backlog and evidence: `doc/IMPROVEMENT_ROADMAP.md`.

## Results so far (qwen2.5:3b, GTX 1650 4 GB). Regenerate: `python -m evaluation.paper_results`
| Test | N | Raw | Framework | Fixed/broken | p |
|---|---|---|---|---|---|
| GSM8K test 0–99 (`results_v6_qwen3b`) | 100 | 84 | 88 | 5/1 | 0.22 |
| TruthfulQA test 0–99, judge truthful (`results_v5_qwen3b`) | 100 | 66% | 63% | 1/4 | 0.38 |
| Hard sets: 12th maths, JEE, real-life, reasoning | 74 | 61 | 65 | 5/1 | 0.22 |
| Difficulty ladder: MATH L1–5, ARC, StrategyQA, logic | 80 | 61 | 66 | 7/2 | 0.18 |
| GSM8K dev 500–539 (design run) | 40 | 28 | 35 | 7/0 | 0.016 |
| Smoke test (16 basic questions) | 16 | | 15 pass | | |
Test sets: frozen copies with pinned dataset versions and checksums in `data/eval_sets/`
(`MANIFEST.json`; rebuild with `python -m evaluation.export_eval_sets`). Full test sets for large runs
(8,151 questions) in `data/eval_sets/full/`: GSM8K 1,319 · TruthfulQA 817 · MATH 5,000 · MATH-500 · JEEBench 515.
Phase 1 reference (llama2:7b): TruthfulQA 65% → 83%, the same +18 under two judges.
Reports: [Results Summary](https://claude.ai/artifact/CpEAGKraVZmCeHi2ge8u8g) ·
[detailed report](https://claude.ai/artifact/26qLfLwSTTtmWRXzPCbu64).

## Lessons (don't relearn)
- Gains come from **self-consistency voting** led by reasoning samples (programs weight 0.5). Programs
  from the same model share its mistakes, so they can't overturn the draft alone.
- Layer 2 full rewrite hurt qwen (TruthfulQA dev 80 → 65); evidence-gated edits are neutral. Open-web
  snippets import myths (4 of 30 edits); curated sources are the open problem.
- **Answer reading** caused the biggest errors: stray digits (old framework 52 vs 65 on hard sets);
  "36 hours … in 4 weeks" read as 4 (inflated GSM8K from +4 to +6). After any reader change,
  re-score every stored run and list verdict flips.
- Samples cut off at the token limit must not vote (maths samples now 1,024 tokens, advanced 1,536).
- Raw drafts differ slightly between runs: always compare framework vs its own drafts.
- DuckDuckGo search drops out intermittently (question search retries). The benchmark runner once
  defaulted to llama2:7b: always pass `--generator`.

## Built but not yet measured
- **Yes/no vote** (`DID_YES_NO_VOTE`, on). Measure on ladder StrategyQA and hard-set reasoning.
- **Wikipedia evidence** (`DID_WIKIPEDIA_EVIDENCE`, off). Measure on TruthfulQA before enabling.
- **Thinking switch** `DID_OLLAMA_THINK=false` for thinking models (qwen3.5, qwen3 hybrids).

## Plan on the Mac
1. **Hardware.** `system_profiler SPHardwareDataType | grep -E "Chip|Memory"`. Ollama can use about
   70% of unified memory. Pick the model size from that (q4 ≈ 0.6 GB per 1B params, plus cache).
2. **Setup.** Install Ollama. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
   In `.claude/launch.json` change `.venv/Scripts/python.exe` to `.venv/bin/python`.
   Pull: `qwen2.5:3b` (baseline reference), the judge `llama3.1:8b`, and 2–3 candidate models.
   Check https://ollama.com/library for current tags; prefer instruct or non-thinking variants
   (seen 2026-11: `qwen3:4b-instruct-2507`, `qwen3.5:4b`, larger sizes in the same families).
   If the pipeline model becomes a Llama, switch the judge to another family.
3. **Choose the model.** `python -m evaluation.model_compare collect --model <tag>` for each
   candidate (add `--no-think` for thinking models), then `... score` (194 questions vs qwen2.5:3b).
   Then set `.env` `DID_OLLAMA_MODEL` and `DID_OLLAMA_JUDGE_MODEL` to it, plus `DID_OLLAMA_THINK=false`
   if needed. Optionally raise `DID_OLLAMA_NUM_CTX` (now 4096).
4. **Recalibrate.** Start the server with `uvicorn app.main:app --port 8000`, then
   `python -m evaluation.benchmark_runner --n 40 --offset 500 --datasets gsm8k truthfulqa --configs baseline full_framework --generator <model> --judge-model llama3.1:8b --out-dir evaluation/dev_<model>`.
   Check: every fast-path draft (H_sem = 0) is correct, else lower `DID_T_SAFE_REASONING`; Layer 3
   fixed > broken; Layer 2 not negative.
5. **Test.** `python -m unittest discover -s tests -t .` (70 pass) · `python -m evaluation.smoke_general`
   · `python -m evaluation.mcq_10th` · `python -m evaluation.hard_eval collect|score --set hard|ladder --tag <model>`
   · full benchmarks for significance: `--n 1319 --datasets gsm8k` and `--n 817 --datasets truthfulqa`.
   Then update the run names in `evaluation/paper_results.py` and regenerate.
   Full MATH/JEE: `python -m evaluation.hard_eval collect --set math500|math|jeebench --tag <model>`, then `score`.
   The pipeline picks one option per MCQ, so JEE multi-correct questions (186) need pipeline support first.
6. **Improve next** (roadmap §3): a verifier for votes where the right answer lost (12c); curated
   Layer 2 sources plus a stricter edit gate (P2-7); false-premise and fabrication checks
   ("What happened on July 20, 1969?" still fails); a bigger ladder (more questions per rung) for the paper.
7. Update the results docs, the roadmap and memory with the new hardware and model.
