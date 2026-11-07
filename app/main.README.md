# app/main.py — HTTP API entry point

FastAPI app that exposes the pipeline over HTTP. It owns one `DefenseInDepthRouter` (the
whole pipeline), starts it when the server starts, and maps each endpoint to it.

Run it with `uvicorn app.main:app --port 8000` (or `python -m app.main`).

| Lines | What it does |
|---|---|
| 1–8 | Module docstring: full pipeline, per-layer test endpoints, health check. |
| 10–22 | Imports: FastAPI, CORS middleware, loguru logger, `settings` (all configuration), request/response schemas, and the router. |
| 28–42 | **Logging setup.** Removes loguru's default handler and installs one format: time, level, **trace ID**, source location, message. `trace_id` defaults to `-` outside a request; inside one, `telemetry.request_trace` fills it, so every log line of a query can be grouped. Level comes from `settings.log_level`. |
| 44 | `_init_error`: the last start-up error, reported by `/health`. |
| 51 | `_router`: the single pipeline instance shared by all requests. |
| 54–83 | **`lifespan`**: FastAPI start/stop hook. On start it builds the router and calls `initialize()` (loads NLI model, checks Ollama models, warms up). If that fails, the error is kept and the server still starts; queries retry initialization (line 144). On shutdown it calls `router.shutdown()`. |
| 90–99 | Creates the FastAPI `app` with title, description, version and the lifespan hook. |
| 102–108 | CORS: allows any origin, so a browser front end on another port can call the API. |
| 116–141 | **`GET /api/v1/health`**: `healthy` only when the router finished initializing; otherwise `degraded` plus `init_error`. Also echoes the key settings (models, thresholds, search backend, device) so a run can confirm which configuration it hit. |
| 144–159 | **`_ready_router`**: used by every query endpoint. 503 if no router exists; if start-up initialization failed, it retries once and returns 503 with the reason if it fails again. This is why a slow Ollama start no longer breaks the server permanently. |
| 162–175 | **`POST /api/v1/query`**: the main entry. Runs the full pipeline (Layer 1 triage, then Layer 2 or Layer 3) and returns `QueryResponse`. Any exception is logged with its stack trace and returned as HTTP 500 with the message. |
| 178–226 | **`/query/layer1`, `/layer2`, `/layer3`**: same as `/query` but set `request.force_layer` so the router skips triage and runs one layer only. Used for ablations and benchmarking. |
| 234–248 | `run_server`: starts Uvicorn with host, port, reload (debug) and log level from settings. Entry point of the `did-serve` command in `pyproject.toml`. |
| 251–252 | Allows `python -m app.main`. |

**Request body** (`QueryRequest`): `prompt` (required), `model` (defaults to `DID_OLLAMA_MODEL`),
`include_audit` (return the full `PipelineResult` trace), `force_layer`.
