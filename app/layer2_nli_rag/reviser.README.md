# app/layer2_nli_rag/reviser.py — turn the evidence verdicts into the final answer

The last step of Layer 2. Three modes, chosen by question type and settings:

| Mode | When | Behaviour |
|---|---|---|
| **Fresh answer** | time-sensitive questions | Answer from today's search results (with today's date); the draft is only a fallback, marked "may be out of date". |
| **Evidence-gated edit** (default) | everything else | Change the draft **only** where evidence contradicts a claim; otherwise return it untouched. |
| Full rewrite + Chain-of-Verification | `layer2_revision_mode=full_rewrite` (Phase 1) | Rewrite the whole answer, then self-check it. Helped llama2:7b, hurt qwen2.5:3b (TruthfulQA dev 80 → 65), so it's off. |

| Lines | What it does |
|---|---|
| 1–17 | Docstring (describes the Phase 1 rewrite + CoVe design). |
| 34–69 | `_REVISER_SYSTEM` / `_REVISER_PROMPT`: full-rewrite instructions: start with the answer, keep supported claims, fix contradicted ones, don't state myths as facts, call a premise false only with evidence, be minimal, 1–2 sentences. |
| 71–86 | Chain-of-Verification prompts: plan up to 3 check questions, answer each independently, then a final answer that trusts the checks where they clearly contradict. |
| 88–99 | `_strip_preamble`: removes lead-ins like "The corrected answer is:". |
| 101–109 | Final CoVe prompt. |
| 111–133 | **`_EDIT_SYSTEM` / `_EDIT_PROMPT`** (default mode): fix only the listed contradicted claims (correct if the evidence states the right fact, else remove); keep every other sentence and caveat; add no new facts. |
| 135–153 | **`_FRESH_SYSTEM` / `_FRESH_PROMPT`**: answer from the given results only, today's date included, the most recent dated statement wins, reply `NOT_FOUND` if the results don't answer. The draft is **not shown**: shown "the PM is Rishi Sunak", the 3B model repeated it over five results naming his successor. |
| 155–159 | Verdict labels for prompts. |
| 169–214 | **`revise`**: time-sensitive → fresh answer; `evidence_gated` → targeted edit; else full rewrite (claims with verdicts + question evidence → rewrite → CoVe). A failed call records a degradation and returns the draft. |
| 216–249 | **`_answer_from_evidence`**: no results, a failed call or `NOT_FOUND` → the draft plus "may be out of date" (and a degradation); otherwise the answer from the results. |
| 251–302 | **`_evidence_gated_edit`**: picks claims labelled contradiction that have real evidence. None → **draft returned unchanged** (the common case). Otherwise one edit call with those claims, their evidence and up to 4 question-level results. |
| 304–340 | `_chain_of_verification` (full-rewrite mode only): plan checks (JSON), answer them in parallel without seeing the answer, then the final answer; any failure keeps the rewrite. |
