# app/layer2_nli_rag/search_retriever.py — fetch evidence

Finds text passages to check claims against. Sources: **DuckDuckGo** web search (default,
free), optional **Tavily** (needs a key), optional **Wikipedia** article intros (off until
measured), and an optional local **ChromaDB** store. Every result carries a relevance weight
used when the NLI step weighs evidence.

| Lines | What it does |
|---|---|
| 20–25 | Wikipedia API address and a User-Agent naming the project's GitHub page. Without a contact, Wikipedia's robot policy rejects requests (HTTP 403). |
| 36–49 | Constructor: backend (`search_backend`), results per search (5), empty ChromaDB handles. |
| 55–112 | **`search(query, max_results, retries)`**: DuckDuckGo first, **retrying with back-off** (1.5 s, 3 s) when it returns nothing (it intermittently answers "No results found"; the router retries only the question search); Tavily if configured; Wikipedia if enabled; ChromaDB if set up. Then removes duplicates, sorts by relevance (Wikipedia 0.9 above web 0.7) and keeps the top `max_results`. A failing source never fails the search. |
| 114–148 | **`search_for_propositions(claims)`**: one search per claim, at most 2 at a time (to avoid tripping DuckDuckGo); a failed search gives that claim no evidence. Returns `{claim id: [results]}`. |
| 154–183 | **`_search_wikipedia`**: Wikipedia full-text search for the best titles, then their plain-text intros (first 800 characters) as results. Works well for claims ("knuckle cracking" → *Joint cracking*), poorly for question-style queries. |
| 189–230 | **DuckDuckGo**: runs the `ddgs` library in a worker thread (it is synchronous); each hit → title, URL, snippet, weight 0.7. Errors are logged and give an empty list. |
| 236–275 | Tavily API search (only with `tavily_api_key`); uses Tavily's own scores. |
| 281–306 | ChromaDB set-up: a persistent collection `evidence_store` (cosine distance). Missing library or errors only log a warning. |
| 308–362 | ChromaDB query: nearest documents, relevance = 1 − distance. |
| 368–378 | `_deduplicate`: drop results with the same URL (or same opening text). |
