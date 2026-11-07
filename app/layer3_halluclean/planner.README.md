# app/layer3_halluclean/planner.py — Layer 3 step 1: plan the solution

Asks the model to break the problem into a short list of ordered steps (max 6). In the
current maths path the plan is passed to the program writer as a **hint** (it "may be
imperfect"); the plan itself never decides the answer.

| Lines | What it does |
|---|---|
| 1–7 | Docstring: a step-by-step plan with dependencies (HalluClean stage 1). |
| 23–48 | `_PLANNER_SYSTEM`: atomic sequential steps, each with dependencies, at most `max_steps`; reply as JSON `{"steps": [{"step_id", "description", "dependencies"}]}`. |
| 50–54 | `_PLANNER_PROMPT`: the query, JSON only. |
| 65–72 | Constructor: client, max steps (`halluclean_max_steps`, 6). |
| 74–104 | **`generate_plan(query)`**: one JSON call at `halluclean_temperature` (0.1), parsed into an `ExecutionGraph`. Any failure → the fallback plan. |
| 106–141 | `_parse_plan`: reads the JSON (or the first `{...}` in the text), keeps up to `max_steps` steps as `PlanNode`s, status `pending`. No steps → fallback. |
| 143–172 | `_fallback_plan`: a generic 3-step plan (analyse → reason → synthesise), recorded as a degradation. |
