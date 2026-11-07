# app/layer1_triage/task_router.py — what kind of question is this?

Labels each question **math**, **factual** or **time_sensitive**. The label decides how
answers are compared in Layer 1 and which layer verifies them:

| Type | Layer 1 compares answers by | Verification if uncertain |
|---|---|---|
| math | the final number / expression | Layer 3: samples + programs + vote |
| factual | meaning (NLI), the option, or yes/no | Layer 2, or the option / yes-no vote |
| time_sensitive | never trusted (no fast path) | Layer 2 answers from fresh web results |

Clear cases are decided by rules in `answer_utils.classify_task_heuristic` (no model call).
Only unclear ones ask the pipeline model, keeping the one-model rule.

| Lines | What it does |
|---|---|
| 1–12 | Docstring: the three types and what each triggers. |
| 27–36 | `_ROUTER_SYSTEM`: instructions for the model: math = must be computed; time_sensitive = present state of something that changes; factual = everything else, including history and comparisons that need no calculation. Answer as JSON. |
| 38 | `_PAST_YEAR`: a 4-digit year. |
| 44–46 | Constructor: keeps the shared Ollama client. |
| 48–52 | **`classify`**: rules first; a confident rule (or `heuristic` mode) answers directly. |
| 53–65 | Otherwise asks the model (JSON, greedy, 30 tokens). If the call fails, records a degradation and uses the rule's guess. |
| 66–70 | An unexpected label → degradation, rule's guess. |
| 71–75 | **Correction 1**: model says time_sensitive but the question names a past year (≤ 2 years ago) → factual. "What happened on July 20, 1969?" is history. |
| 76–80 | **Correction 2**: model says time_sensitive but the question gives 2+ numbers (not years) → math. "Mars has a 697-day year and a 12-day week…" is computed, not looked up. |
| 81–82 | Otherwise the model's label. |
| 84–92 | Helpers for the two corrections. |
