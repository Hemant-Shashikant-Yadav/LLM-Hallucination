# uncertainty/generate/fake.py — the simulated model

For dry runs and tests (PLAN §1). Every (model, question) pair gets a hidden type, and every
reply follows from it the way the plan expects a real model to behave. Deterministic.

| Part | What it does |
|---|---|
| `PLANTED`, `PLANTED_TO_TYPE` | certain-right, certain-wrong (confident hallucination), knowledge gap, reasoning slip, ambiguous, unanswerable, false premise, temporal; and the Objective 2 type each should be classified as. |
| `find_item` | Which question a prompt is about (prefix index, longest match). |
| `planted` | Metadata types first; otherwise skill grows with model size, and reasoning slips are likelier on maths. |
| `_distribution`, `_greedy_answer` | Answer distribution of samples by type, changed by evidence (fixes knowledge gaps) or a hint (fixes reasoning slips). |
| `_premise`, `_render`, `_tokens` | Worked answers share their premises when only the conclusion is unstable; confident types get high log-probs. |
| `_p_true`, `_probe`, `_judge` | Informative but noisy P(True); probe answers by type; a judge that checks reference answers. |
