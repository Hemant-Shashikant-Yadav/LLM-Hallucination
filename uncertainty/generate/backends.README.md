# uncertainty/generate/backends.py — talking to a model

`OllamaBackend` for the real server and the interface `FakeBackend` shares. Results carry
per-token log-probs; a server that sends none gives empty lists.

| Part | What it does |
|---|---|
| `Generation` | Text, tokens, log-probs, top alternatives, stop reason, token counts, seconds; `truncated` when stopped by the token limit; `as_record` rounds log-probs for storage. |
| `OllamaBackend.generate` | Options (temperature, limit, context, seed, top_k), log-probs on request, JSON format, `think`. A per-event-loop gate keeps at most `concurrency` requests in flight; 5xx and dropped connections are retried, timeouts are not. |
| `_to_generation` | Ollama's response → `Generation`. |
| `version`, `available`, `loaded`, `pull`, `unload` | Server checks used by `run-all`; pulls have no timeout (8–9 GB downloads). |
| `make_backend` | `ollama` or `fake`. |
