# app/layer3_halluclean/program_verifier.py — Layer 3 for maths: programs + vote

The heart of the maths gains. For an uncertain maths question it has the model write
several short **Python programs** that compute the answer, runs them in a locked-down
subprocess, and holds a **vote** over the reasoning samples (draft + Layer 1 samples +
extra samples) in which each working program adds half a vote. Programs give outside
evidence, but programs from the same model share its misreadings, so they can't overturn
the draft on their own unless two different methods agree.

| Lines | What it does |
|---|---|
| 1–17 | Docstring: plan → write programs → run and check → vote (PAL, PROVE, self-consistency). |
| 49–98 | **Basic programs** (word problems): instructions (one variable per quantity, return exactly what is asked, plain arithmetic) and three worked examples written for this project (not from GSM8K, so no test leakage). |
| 100–139 | **Advanced programs** (12th-standard and JEE maths): exact computation with sympy, `Fraction`, `math.comb`; examples for an integral, a derivative and a probability. |
| 141–176 | **Check programs**: solve by an *independent* method (enumerate cases, numeric approximation, search), so a shared formula mistake is unlikely to repeat. |
| 178–196 | Prompt templates: examples + problem + the planner's plan as a hint; a repair prompt with the error. |
| 198–201 | Words that make a negative answer plausible (debt, temperature…). |
| 206–218 | **Sandbox allow-list**: imports limited to math, fractions, decimal, sympy, itertools, statistics, collections, functools, operator. Blocked names include `open`, `exec`, `eval`, `__import__`, `getattr`, and sympy's string parsers (`sympify`, `parse_expr`, `lambdify`), which call `eval`. |
| 221–234 | `_RUNNER`: the script run in the subprocess: calls `solution()`, prints a number, or `EXPR:` plus a sympy expression (only a real sympy object, never a string). |
| 237–252 | Helpers: which option a value matches; numeric conversion; integer check. |
| 258–263 | Constructor: client, 5 programs, 20 s time limit per program (sympy is slow to import). |
| 265–286 | **`verify`**: an exact calculator value short-circuits (line 442). Otherwise picks program styles (advanced maths alternates exact sympy and independent-check programs; else all basic), writes all programs in parallel (first one greedy, the rest at temperature 0.5 with different seeds), runs them. |
| 288–299 | **Self-debug**: each crashed or unparseable program gets its error back once and is re-run (never for sandbox-blocked ones). |
| 301–322 | **Plausibility checks** for everyday word problems: a fractional answer when the problem only has whole numbers, or a negative answer without a reason for one, is discarded. Not applied to advanced maths, where π/4 or negative values are normal. Each program's result goes into the audit trace. |
| 324–345 | Puts all answers on one scale: multiple choice → which option each answer/program picks; otherwise canonical answers (`0.875` = `7/8`). Records `[Samples] draft=… answers=[…]` in the trace. |
| 347–375 | **Main rule, sample-primary vote**: score = samples for an answer + 0.5 × programs for it; **ties favour the draft**. If the winner has **no sample** behind it, it may overturn the draft only when 3+ programs agree **including both an exact and a check program**; otherwise the draft is kept. |
| 376–396 | Older program-primary rule (Phase 1 setting): only program-confirmed answers count; one lone program can't overturn the draft. |
| 397–410 | No program worked → plain vote over samples (recorded as a degradation); nothing at all → keep the draft. |
| 412–440 | Judgment (agreement share, what changed). Draft wins → returned unchanged. Otherwise returns the first sample that gave the winning answer plus `Final answer: …`, or failing that, the program's code. |
| 442–471 | **`_verify_exact`**: plain arithmetic: the exact calculator result decides; the draft is kept if it agrees, else replaced (works for multiple choice too). |
| 475–493 | `_write_program`: one program in the chosen style; failure → degradation, empty program. |
| 495–507 | `_repair`: sends the error back once. |
| 509–512 | `_extract_code`: code from a ```python block. |
| 514–520 | `_run`: one bad program fails alone, never the query. |
| 522–558 | **`_execute`**: needs `def solution`; static safety check; runs with `python -I` (isolated mode) in a worker thread with the time limit. Reads a number (rounded when within 1e-6 of an integer) or an `EXPR:` expression; rejects non-numbers and infinity. |
| 560–589 | **`_unsafe_reason`**: parses the code and rejects syntax errors, disallowed or blocked imports and names, any `_private` attribute access, `global`/`nonlocal`, and async code. A disallowed import can be repaired (the message lists allowed modules); a blocked name can't. |
