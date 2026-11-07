# app/telemetry.py — trace IDs, stage timing, degradation records

Small helpers that make every query traceable. Each query gets a **trace ID** stamped on
all its log lines; each pipeline step logs when it starts and ends; and any step that falls
back instead of doing its job records a **degradation**, which is returned in the audit
trail. This exists because the pipeline used to fail silently at Stage 1.

| Lines | What it does |
|---|---|
| 1–8 | Docstring: the three jobs above. |
| 20–22 | `_degradations`: a context variable holding the current query's list of degradations. Context variables are per-request, and `asyncio.gather` copies them into child tasks, so parallel steps record into the same list. |
| 23 | `_trace_id`: context variable with the current query's trace ID. |
| 26–37 | **`request_trace(trace_id=None)`**: context manager opened once per query (in `router.process_query`). Creates a 12-character ID, starts an empty degradation list, and binds the ID to loguru (`logger.contextualize`) so the log format in `main.py` prints it. Restores the previous values on exit. |
| 40–41 | `current_trace_id()`: returns the ID, copied into `PipelineResult.trace_id`. |
| 44–50 | **`record_degradation(stage, reason)`**: logs `DEGRADED | stage: reason` as a warning and appends it to the list. Called by components when, for example, search returns nothing, the NLI model is unavailable, or a sample failed. |
| 53–54 | `degradations()`: a copy of the list, copied into `PipelineResult.degradations`. |
| 57–79 | **`stage(name, **fields)`**: async context manager wrapped around each pipeline step. Logs `STAGE_ENTER` with any fields, yields a dict the step fills with its outcome (e.g. `H_sem`, decision), then logs `STAGE_EXIT` with the duration and that outcome. On an exception it logs `STAGE_FAIL` with name, duration and error, then re-raises: errors are never swallowed. |

Typical log of one query: `STAGE_ENTER | draft` … `STAGE_EXIT | draft | 4120ms | {tokens: 312}`
→ `layer1.task_route` → `layer1.triage` → `layer3.extra_samples` → `layer3.program_vote`.
