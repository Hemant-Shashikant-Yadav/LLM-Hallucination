# app/layer2_nli_rag/proposition_parser.py — split a draft into checkable claims

Layer 2 checks facts one at a time. This file asks the pipeline model to break the draft
into **atomic propositions**: short, self-contained claims ("The Eiffel Tower was completed
in 1889."), each of which can be searched for and checked against evidence on its own.

| Lines | What it does |
|---|---|
| 1–7 | Docstring. |
| 23–35 | `_AP_EXTRACTION_SYSTEM`: rules for the model: one verifiable statement per claim, resolve pronouns to names, no opinions, keep numbers/dates/names exact; reply as `{"propositions": [...]}`. |
| 37–45 | `_AP_EXTRACTION_PROMPT`. Asks for a JSON **object**, because Ollama's JSON mode only allows an object at the top level; asking for a bare list made the model invent wrappers and fall back. |
| 56–58 | Constructor: keeps the Ollama client. |
| 60–102 | **`extract_propositions(text, max_propositions)`**: empty draft → no claims. Otherwise one greedy JSON call, parse, keep the first `max_propositions` (the router asks for 6). Any failure → split into sentences instead. |
| 104–158 | `_parse_json_response`: reads the JSON (or the first `[...]` inside the text). Accepts `{"propositions": [...]}`, any other key holding a list, or a flat `{"AP1": "...", …}` mapping. Each string (or `{"text": ...}` item) becomes an `AtomicProposition` with its index and the source draft. Anything unusable → sentence split. |
| 160–189 | `_fallback_sentence_split`: each sentence over 10 characters becomes a claim; recorded as a degradation so the audit shows the model's extraction failed. |
