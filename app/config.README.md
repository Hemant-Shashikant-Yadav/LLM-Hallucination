# app/config.py — every setting in one place

All thresholds, model names, limits and switches are fields of one Pydantic settings class.
Each can be overridden by an environment variable or the project's `.env` file, using the
prefix `DID_` and the field name in capitals: `t_safe` → `DID_T_SAFE`. Import it anywhere as
`from app.config import settings`.

**Current `.env` values** (they override the defaults below): `DID_OLLAMA_MODEL=qwen2.5:3b`,
`DID_OLLAMA_JUDGE_MODEL=qwen2.5:3b` (one model in every layer), `DID_T_SAFE_REASONING=0.2`.

| Lines | What it does |
|---|---|
| 18 | `_PROJECT_ROOT`: repo root, used to find `.env`, ChromaDB and output folders. |
| 21–30 | `_default_device`: picks `cuda`, else Apple `mps`, else `cpu` for the local models (NLI, probe). |
| 41–47 | Settings behaviour: read `.env`, prefix `DID_`, case-insensitive, ignore unknown keys. |
| 52–59 | **Hardware**: `device`, `use_fp16` (half precision on CUDA). |
| 64–72 | `t_safe` (0.35): Layer 1 threshold for factual questions. Semantic entropy below it and the draft in the majority → **fast path** (answer returned unverified). |
| 73–82 | `t_safe_reasoning`: same for maths questions. Default 0.0 (always verify, chosen for llama2); `.env` sets 0.2 for qwen, because every unanimous qwen dev draft was correct. |
| 83–90 | `triage_mode`: `semantic_entropy` (sample and cluster, used) or `surrogate_probe` (TinyLlama probe, needs trained weights). |
| 91–97 | `task_router_mode`: `llm` = rules first, the model decides unclear questions; `heuristic` = rules only. |
| 98–123 | **Layer 1 sampling**: `se_num_samples` (4 extra answers), `se_sample_temperature` (0.7), `se_max_tokens` (512 for non-maths), `se_min_samples` (fewer successful samples than 2 → don't trust the entropy, escalate). |
| 124–143 | Probe settings (TinyLlama model, which layers to probe, weights path). Only for `surrogate_probe` mode. |
| 148–175 | **Layer 2 NLI**: NLI model (DeBERTa-v3), confidence needed for entailment, and the consensus score cut-offs: S ≥ 0.15 accept a claim, S ≤ −0.15 reject. |
| 176–183 | `layer2_revision_mode`: `evidence_gated` (edit only contradicted claims; used) vs `full_rewrite` (Phase 1; hurt qwen 80 → 65). |
| 184–191 | `layer2_question_evidence_for_claims` (off): checking claims against question-level results added false contradictions. |
| 192–219 | **Search**: backend (DuckDuckGo), optional Tavily key, ChromaDB path, `wikipedia_evidence` (off until measured), results per search. |
| 224–236 | **Layer 3**: max plan steps; `halluclean_reasoning_mode` = `program_vote` (used) or `plan_judge`. |
| 237–255 | Program vote: 5 programs per question; `sample_primary` weighting (samples vote, each program adds 0.5). |
| 256–275 | **Adaptive sampling**: at least 6 extra samples, then 3 more at a time while the top answer has under 70% agreement, up to 12. |
| 276–291 | Token budgets for maths samples: 1,024 normal, 1,536 advanced. Cut-off samples are dropped, so short limits silenced hard questions. |
| 292–302 | `question_search_retries` (2): retry the question search when DuckDuckGo returns nothing. |
| 303–318 | `mcq_vote`, `yes_no_vote`: uncertain non-maths multiple-choice and yes/no questions are settled by a vote over the samples instead of Layer 2. |
| 319–325 | `verify_advanced_math`: calculus, matrices, probability, counting never take the fast path. |
| 326–338 | Old plan-and-judge Layer 3 options (executor mode, temperature). |
| 343–354 | **Ollama**: server URL, `ollama_model` (pipeline model), `ollama_judge_model` (its in-pipeline judging role; set equal to the pipeline model). |
| 355–389 | Call limits: timeout 600 s, context window 4,096 tokens, 1 request at a time (matches Ollama's parallelism), 2 retries on transient errors, keep the model loaded 10 minutes. |
| 390–396 | `ollama_think`: send `think=false` to thinking models so samples stay short. |
| 401–404 | API server host, port, debug reload, log level. |
| 409–412 | Default folder for evaluation output. |
| 416 | `settings`: the single instance everyone imports. |
