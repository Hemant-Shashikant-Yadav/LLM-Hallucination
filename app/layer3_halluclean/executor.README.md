# app/layer3_halluclean/executor.py — Layer 3 step 2 (plan-and-judge path)

> **Not on the current maths path.** Maths uses `program_verifier.py` (programs + vote).
> This executor runs only in `DID_HALLUCLEAN_REASONING_MODE=plan_judge`, or for a forced
> Layer 3 request that isn't maths. Kept from the HalluClean design (Phase 1).

Carries out the planner's steps with the model and returns the reasoning text.

| Lines | What it does |
|---|---|
| 1–7 | Docstring: execute each plan step with its own context. |
| 20–39 | Per-step prompts: do only this step, use earlier steps' outputs, state a conclusion, don't invent facts. |
| 41–60 | Single-pass prompts: follow the whole plan in one answer, label `Step N:`, write each calculation, end with `Final answer: <answer>`. |
| 71–73 | Constructor. |
| 75–89 | **`execute_plan`**: `single_pass` mode (default) → one call for the whole plan. |
| 91–141 | `per_step` mode: steps in dependency order; each step gets the outputs of the steps it depends on; a failed step is marked `failed` and recorded as a degradation. Returns the graph and per-step traces. |
| 143–173 | `_execute_single_pass`: plan as a numbered list → one call (1,536 tokens) → output stored on the last step; failure → degradation. |
| 175–196 | `_execute_step`: one step via one call. |
| 198–217 | `_build_dependency_context`: earlier steps' outputs as context text. |
| 219–256 | `_topological_sort`: Kahn's algorithm for a valid order (ties by step number; cyclic leftovers appended). |
