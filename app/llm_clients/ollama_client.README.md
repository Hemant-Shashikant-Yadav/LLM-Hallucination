# app/llm_clients/ollama_client.py — the only door to the LLM

Every model call in the pipeline (draft, samples, claim extraction, edits, programs, router)
goes through `OllamaClient`. It adds what the raw Ollama SDK lacks: one-request-at-a-time
queueing, retries on transient failures, a clear error when a model isn't pulled, token
counting per query, and warnings when output is cut off.

| Lines | What it does |
|---|---|
| 23–24 | `ModelUnavailableError`: raised when the model isn't pulled (instead of a vague 404). |
| 27–28 | `_normalize_model`: `qwen2.5` → `qwen2.5:latest`, so name checks match Ollama's list. |
| 34–51 | **Token meter.** A per-query context variable `[prompt_tokens, completion_tokens]`. The router calls `start_token_meter()` at the start of each query; `_record_tokens` adds every call's counts. This is how `PipelineResult.tokens_generated` counts *all* calls, not just the draft. |
| 57–64 | **Concurrency gate.** One semaphore per (event loop, server) allowing `ollama_max_concurrency` (1) calls at once. Requests wait here, not in Ollama's queue, so the HTTP timeout measures generation only. |
| 70 | Network errors worth retrying (connection refused/reset). Timeouts are deliberately not retried: one already cost 600 s. |
| 81–98 | Constructor: server URL, default model, timeout from settings; creates the async SDK client. |
| 104–111 | `default_model` and `use_model` (re-point every component sharing this client to another model). |
| 113–131 | **`validate_models`**: lists pulled models at start-up; raises `ConnectionError` if Ollama is down and `ModelUnavailableError` if the pipeline model is missing, with the exact `ollama pull` command. |
| 133–139 | `ensure_available(model)`: the name check used above. |
| 141–153 | `warmup`: a 1-token generation at start-up loads the model into memory and fails early if it can't run. |
| 159–185 | **`_call`**: every request goes through here. Adds `keep_alive`, and `think` when `ollama_think` is set. Then, inside the gate: HTTP 404 → `ModelUnavailableError`; other 4xx → raise; 5xx (runner crashed, out of memory while loading) or dropped connection → retry up to 2 times with 2 s, 4 s back-off, logging each retry. |
| 187–196 | `_options`: temperature, max tokens (`num_predict`), context window (`num_ctx`, 4,096) and optional seed (samples use fixed seeds so runs are repeatable). |
| 198–221 | **`_to_result`**: wraps the response as `GenerationResult`, warns when output hit the token limit (`done_reason == "length"`, i.e. cut off) or when prompt + output filled the context window (Ollama silently drops the prompt start), and adds the tokens to the meter. |
| 227–272 | **`generate(prompt, …)`**: single completion. Greedy by default (temperature 0, used for the draft); samples pass a temperature and seed. Errors are logged with model name and re-raised. |
| 274–301 | `_generate_streaming`: same, but collects a streamed response (holds the gate for the whole stream). |
| 307–341 | `chat(messages, …)`: multi-turn chat variant. |
| 347–374 | **`generate_json`**: forces JSON output (`format="json"`). Must ask for a JSON *object*; used by the task router and claim extraction. |
| 382–433 | **`GenerationResult`**: text, model, prompt/completion tokens, durations, `done_reason`; `truncated` (cut off at the limit; maths samples that are truncated don't vote), `total_tokens`, `latency_ms`. |
