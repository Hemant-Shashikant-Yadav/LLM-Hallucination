# 5. History

## Phase 1 (to Oct 2026): llama2:7b

- Three-layer design: a TinyLlama "surrogate probe" for triage (`probe_model.py`, never
  trained), Layer 2 with a full rewrite and Chain-of-Verification, Layer 3 as
  plan → execute → self-judge.
- Result: TruthfulQA 61 → 79% (+18, p = 0.004); GSM8K weak (llama2 drafts mostly wrong).

## Phase 2 (Nov 2026): one small model, qwen2.5:3b

| Date | Change | Why |
|---|---|---|
| 5 Nov | Reliability audit: silent Stage 1 exits fixed; trace IDs, stage logs, degradations; Ollama gate, retries, model checks, truncation warnings | The pipeline was exiting at Stage 1 with no error |
| 5 Nov | Layer 1 → sampled semantic entropy (the probe had no trained weights) | A real uncertainty signal |
| 5 Nov | Switched every role to qwen2.5:3b (fits 4 GB fully) | One-model rule; llama2 too weak and slow |
| 5 Nov | Dev runs v1–v3: program-led vote → sample-led vote; full rewrite → evidence-gated edit | v1 hurt both layers; v3 GSM8K 70 → 87.5% |
| 5 Nov | Test run (100 + 100): GSM8K up, TruthfulQA −3 | First test numbers |
| 5 Nov | Task router, exact calculator, freshness rule, answer normalisation, MCQ support | "Answer any basic question" requirement; smoke 15/16 |
| 6 Nov | Hard sets (74): safe symbolic reader, sympy and check programs, independent-method rule, truncated samples excluded | Old reader scored 52; new 65 vs raw 61 |
| 6 Nov | Judge check: Phase 1 re-graded by a second judge | Same +18; agreement 82% |
| 6 Nov | Difficulty ladder: first run 61 = 61; longer maths samples, option vote, router fixes → 66 vs 61 | Hard maths and logic needed votes |
| 6 Nov | Search retry, fresh-answer prompt with today's date | Stale "current PM" answer |
| 6 Nov | GSM8K re-run: scoring correction (+6 was +4); runner default model fix; spelled-out word-problem routing | Final GSM8K 84 → 88% |
| 6 Nov | Yes/no vote, thinking-mode switch, Wikipedia evidence (off), model comparison script | Unmeasured; prepared for bigger hardware |
| 7 Nov | Frozen test sets (8,151 questions) with pinned versions; MATH/JEE scoring; evaluation folder reorganised; docs rewritten | Reproducibility for the paper |

## Bugs that changed the numbers

| Bug | Effect | Fix |
|---|---|---|
| Stray digits read as answers (`4` from `\frac{\pi}{4}`) | Old framework broke 11 hard-set answers | Safe symbolic parser (`to_sympy`) |
| Cut-off samples voting with their last digit | Wrong votes on hard maths | Truncated samples dropped |
| 512-token sample limit | 1–3 voters on MATH L3–L5 | 1,024 / 1,536 tokens |
| Last number used when no marker ("in 4 weeks") | GSM8K gain overstated (+6 vs +4) | Concluding-sentence reading |
| Benchmark runner defaulted to llama2:7b | A run silently used the wrong model on CPU | Defaults to the configured model |
| "Today" in a word problem or quoted premise | Misrouted to web search | Word problems checked first; quotes ignored |
| DuckDuckGo "No results found" | Stale current-events answers | Retry with back-off |
