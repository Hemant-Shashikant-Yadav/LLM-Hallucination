# app/router.py — the pipeline: draft → route → triage → layer → answer

`DefenseInDepthRouter` builds every component once and runs each query through them. This
is the file to read first to understand the system's flow.

```
question → draft (greedy) → task type (math / factual / time_sensitive)
         → Layer 1 triage (sample + semantic entropy) ──SAFE──→ return draft
                  │ uncertain
                  ├─ non-maths multiple choice ─→ option vote
                  ├─ yes/no question ───────────→ yes/no vote
                  ├─ factual / time-sensitive ──→ Layer 2 (claims + evidence + NLI + edit)
                  └─ maths ─────────────────────→ Layer 3 (samples + programs + vote)
```

| Lines | What it does |
|---|---|
| 1–11 | Docstring: the pipeline steps. |
| 22–65 | Imports from every layer, the Ollama client (with the token meter), telemetry and schemas. |
| 69 | `_task_type`: the query's task type, copied into the audit trail. |
| 80–119 | **Constructor.** Two Ollama clients: `_ollama` (draft, samples, router) and `_judge_ollama` (Layer 2/3 roles). Both point to the **same model** in the current setup (`.env`). Builds Layer 1 (probe engine, used only in probe mode; semantic-entropy triage; task router), Layer 2 (claim parser, search, NLI verifier, reviser) and Layer 3 (planner, executor, judge, program verifier). |
| 125–214 | **`initialize`** (at server start): loads the NLI model first (Layer 1 clustering also needs it); checks the pipeline model is pulled and warms it up (mandatory: raises); checks the judge-role model and falls back to the pipeline model if unusable. In `semantic_entropy` mode it stops there (lines 162–170); the rest loads the legacy TinyLlama probe for `surrogate_probe` mode. |
| 216–230 | Optional ChromaDB start-up; `shutdown` unloads models. |
| 236–256 | **`process_query`**: opens a trace (trace ID, degradation list) and a `pipeline` stage around the real work; logs which path ran. |
| 258–275 | **Step 1, draft**: starts the token meter; one greedy (temperature 0) answer to the question. This draft is also the "raw model" answer in every evaluation. An empty draft is recorded as a degradation. |
| 277–281 | `force_layer` set (per-layer endpoints): skip triage. |
| 283–288 | **Step 2a, task type**: rules first, model for unclear cases (`task_router.py`). For maths, also tries the exact calculator. |
| 290–310 | **Step 2b, Layer 1 triage**: samples 4 more answers, groups them by meaning, computes semantic entropy, decides SAFE / knowledge / reasoning uncertain. If an exact value exists and the draft disagrees, it is forced to Layer 3 regardless. |
| 313–336 | **Fast path**: SAFE → return the draft unchanged (`layer1_only`). |
| 338–344 | Uncertain **non-maths multiple choice** → option vote (line 663). |
| 346–353 | Uncertain **yes/no question** whose draft says yes/no → vote over verdicts. |
| 355–360 | Knowledge uncertainty → **Layer 2**. |
| 362–367 | Otherwise (maths) → **Layer 3**. |
| 369–381 | `_cost_fields`: tokens across all calls (meter), draft tokens and latency, trace ID, degradations, task type, for every `PipelineResult`. |
| 387–437 | `_run_layer1`: legacy probe-based triage (probe mode only), with fallbacks. |
| 443–549 | **Layer 2**: in parallel, extract up to 6 checkable claims from the draft and search the web for the question itself (retried if empty). If neither gives anything → return the draft and record it. Search evidence per claim (optionally add question evidence, off). Run NLI per claim against its evidence, count entailed/contradicted/neutral, then **revise**: time-sensitive questions are answered from fresh results; otherwise only claims the evidence contradicts are edited. Returns `layer1_layer2` with the result. |
| 555–630 | **Layer 3**: if the exact calculator has a value, use it directly (no planning). Otherwise make a short plan; for maths (`program_vote` mode) take Layer 1's answers, add samples adaptively, then `ProgramVerifier.verify` writes programs and runs the vote. The non-maths branch (old plan → execute → judge) is kept for `plan_judge` mode. |
| 632–661 | **`_adaptive_samples`**: draws 6 extra samples (seeded, maths token budget), then 3 more at a time while the most common answer has under 70% agreement, up to 12; stops early if most samples were cut off and dropped. `key` says how to read an answer (number, option, or yes/no). |
| 663–709 | **`_process_choice_vote`** (option and yes/no vote): reads every answer's choice, picks the most common; **a tie keeps the draft**; returns the first sample giving the winner. The audit records the vote counts, e.g. `option votes over 17: (d) 9, (a) 7`. |
| 711–739 | `_finish_layer3`: builds the `layer1_layer3` result (latency, changed-or-not, costs). |
| 745–787 | `_process_forced_layer`: a synthetic triage result for the per-layer endpoints, then runs the requested layer. |

**Key safety rules here:** the draft is always the fallback; ties keep the draft; an exact
calculation beats any model answer; every fallback is recorded as a degradation.
