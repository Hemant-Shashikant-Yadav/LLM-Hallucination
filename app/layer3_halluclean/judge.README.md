# app/layer3_halluclean/judge.py — Layer 3 steps 3–4 (plan-and-judge path)

> **Not on the current maths path** (see `program_verifier.py`). Used only in
> `plan_judge` mode or for forced non-maths Layer 3 requests. Replaced for maths because
> a model judging its own reasoning without outside feedback tends to make answers worse
> (Huang et al., ICLR 2024); programs supply that outside feedback.

The model checks the executor's reasoning, fixes it if it finds errors, and decides
between the fixed answer and the draft.

| Lines | What it does |
|---|---|
| 1–10 | Docstring: stage 3 judge, stage 4 refine. |
| 35–59 | Judge prompts: recompute every calculation, check every quantity is in the question, check it answers what was asked; reply as JSON (`is_consistent`, `confidence`, `violations`). |
| 66–83 | Refiner prompts: fix each reported issue, recompute, end with `Final answer:`. |
| 94–96 | Constructor. |
| 98–146 | **`judge_and_refine`**: judge; refine only if problems were found; then choose the final answer; returns `Layer3Result`. |
| 152–179 | `_judge`: one JSON call; on failure the solution passes unchecked (recorded as a degradation). |
| 181–219 | `_parse_judgment`: reads the JSON (string "true" accepted, confidence clamped to 0–1); unreadable output → passes unchecked (degradation). |
| 225–248 | **`_select_final`**: for maths, keep the draft when its final number equals the checked chain's (don't replace correct drafts); otherwise the checked chain. Errors → the draft. |
| 250–289 | `_refine`: rewrite with the listed issues; failure → the last reasoning step (degradation). |
