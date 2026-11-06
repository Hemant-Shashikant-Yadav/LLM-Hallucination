"""
Layer 3 (quantitative queries): Program-Verified Self-Consistency.

Replaces the LLM "judge + refine" stages for numeric reasoning, because
intrinsic self-correction without external feedback tends to hurt accuracy
(Huang et al., ICLR 2024). Program execution supplies that external feedback:

  Stage 1  Plan       — HalluClean planner decomposes the problem (unchanged).
  Stage 2  Execute    — the verifier writes P independent Python programs that
                        follow the plan (Program-Aided LM; Gao et al., 2023).
  Stage 3  Judge      — programs are statically checked and executed in a
                        sandboxed subprocess; the sampled natural-language answers
                        (draft + Layer 1 samples) only count when a program
                        confirms them (PROVE; Toh et al., 2024).
  Stage 4  Refine     — majority vote over confirmed answers (Self-Consistency;
                        Wang et al., 2023). The draft is kept when it matches.
"""

from __future__ import annotations

import ast
import asyncio
import math
import re
import subprocess
import sys
from collections import Counter

from loguru import logger

from app.answer_utils import (
    Calculation,
    answers_match,
    canonicalize,
    extract_choice,
    extract_final_answer,
    is_advanced_math,
    normalize_number,
    numbers_match,
    option_value,
    parse_options,
    values_close,
)
from app.config import settings
from app.llm_clients.ollama_client import OllamaClient
from app.models.schemas import JudgmentResult
from app.telemetry import record_degradation

_PROGRAM_SYSTEM = """You solve math word problems by writing a short Python program.

Rules:
1. Define a function `solution()` that returns the final numeric answer.
2. Use one clearly named variable per quantity in the problem, with a comment.
3. Compute every intermediate quantity explicitly. Use only numbers stated in the problem.
4. Return exactly what the question asks for (e.g. the remaining amount, not the total).
5. Use plain Python arithmetic only (the `math` module is allowed). No input(), no printing, no files.
6. Return a number, not a string.
Respond with the Python code only, inside a ```python code block."""

# Few-shot examples in the style of Program-Aided LMs (Gao et al., 2023). Written for
# this project — not taken from the GSM8K test set — so they cannot leak test answers.
_PROGRAM_EXAMPLES = '''PROBLEM: A bakery bakes 48 muffins. It sells 3/4 of them in the morning and then bakes 10 more. How many muffins does it have now?

```python
def solution():
    muffins_baked = 48  # muffins baked at first
    fraction_sold = 3 / 4  # sold in the morning
    muffins_sold = muffins_baked * fraction_sold
    muffins_left = muffins_baked - muffins_sold
    extra_baked = 10  # baked afterwards
    return muffins_left + extra_baked
```

PROBLEM: Tom works 6 hours a day for 5 days and earns $12 per hour. He spends $45 on groceries and saves half of what is left. How much does he save?

```python
def solution():
    hours_per_day = 6
    days = 5
    pay_per_hour = 12  # dollars
    earnings = hours_per_day * days * pay_per_hour
    groceries = 45  # dollars spent
    left = earnings - groceries
    savings = left / 2  # saves half of the rest
    return savings
```

PROBLEM: A store has 120 shirts. 25% of them are red and 1/3 of the remaining shirts are blue. The rest are white. How many white shirts are there?

```python
def solution():
    total_shirts = 120
    red = total_shirts * 25 / 100
    remaining = total_shirts - red
    blue = remaining / 3  # 1/3 of the remaining, not of the total
    white = remaining - blue
    return white
```'''

# Senior-school maths: exact symbolic computation with sympy
_ADVANCED_SYSTEM = """You solve maths problems by writing a short Python program.

Rules:
1. Define a function `solution()` that returns the final answer.
2. Use sympy for calculus, algebra, matrices and vectors; use fractions.Fraction and
   math.comb / math.perm for probability and counting, so the result is exact.
3. Return a number or a sympy expression (for example an exact value such as pi/4,
   or an expression in x for a derivative). Never return a string.
4. Return exactly what the question asks for. No input(), no printing, no files.
Respond with the Python code only, inside a ```python code block."""

_ADVANCED_EXAMPLES = '''PROBLEM: Evaluate the definite integral of x^2 e^x dx from x = 0 to x = 1.

```python
import sympy as sp
def solution():
    x = sp.symbols('x')
    return sp.integrate(x**2 * sp.exp(x), (x, 0, 1))
```

PROBLEM: Find dy/dx if y = x^3 cos x.

```python
import sympy as sp
def solution():
    x = sp.symbols('x')
    return sp.diff(x**3 * sp.cos(x), x)
```

PROBLEM: Three cards are drawn without replacement from 5 red and 3 blue cards. What is the probability that exactly two are red?

```python
from fractions import Fraction
from math import comb
def solution():
    favourable = comb(5, 2) * comb(3, 1)  # choose 2 red and 1 blue
    total = comb(8, 3)
    return Fraction(favourable, total)
```'''

# A second, independent method: errors that the direct formula makes are unlikely to
# be repeated by enumeration or numerical approximation, so agreement is real evidence
_CHECK_SYSTEM = """You check maths answers by solving the problem with an INDEPENDENT
method in a short Python program, without reusing the obvious formula.

Rules:
1. Define a function `solution()` that returns the final numeric answer.
2. Counting and probability with small numbers: enumerate every case with itertools
   and count. Integrals, limits and derivatives at a point: approximate numerically
   (a fine midpoint sum, a value very close to the limit point, a small difference
   quotient). Equations: search for the value that satisfies the condition.
3. Use only what the problem states. No input(), no printing, no files.
4. Return a number, never a string.
Respond with the Python code only, inside a ```python code block."""

_CHECK_EXAMPLES = '''PROBLEM: Three cards are drawn without replacement from 5 red and 3 blue cards. What is the probability that exactly two are red?

```python
from itertools import combinations
from fractions import Fraction
def solution():
    cards = ['R'] * 5 + ['B'] * 3
    hands = list(combinations(range(8), 3))
    good = sum(1 for hand in hands if sum(cards[i] == 'R' for i in hand) == 2)
    return Fraction(good, len(hands))
```

PROBLEM: Evaluate the definite integral of x^2 e^x dx from x = 0 to x = 1.

```python
import math
def solution():
    n = 200000
    h = 1 / n
    return sum(((i + 0.5) * h) ** 2 * math.exp((i + 0.5) * h) for i in range(n)) * h
```'''

_PROGRAM_PROMPT = """{examples}

PROBLEM: {query}

HINT (solution plan, may be imperfect):
{plan}

Write the program."""

_REPAIR_PROMPT = """PROBLEM: {query}

This program fails with the error below. Fix the error and return the corrected
complete program. Keep the same approach unless it is wrong.

```python
{code}
```

ERROR: {error}"""

_NEGATIVE_CUES = re.compile(
    r"\b(negative|below zero|minus|debt|owes?|temperature|degrees)\b",
    re.IGNORECASE,
)

# Sandbox rejections that must not be "repaired" into something that gets past the check
# (A disallowed import is repairable: the error names the allowed modules and the
# repaired code is checked again, so a repair can never widen what runs.)
_UNSAFE_PREFIXES = ("rejected: blocked", "rejected: private")

_ALLOWED_IMPORTS = {
    "math", "fractions", "decimal", "sympy", "itertools", "statistics",
    "collections", "functools", "operator",
}
_BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals", "locals",
    "getattr", "setattr", "delattr", "vars", "dir", "breakpoint", "exit", "quit",
    "help", "memoryview", "classmethod", "staticmethod", "super", "type",
    # sympy entry points that parse or compile strings (sympify / parse_expr use eval)
    "sympify", "parse_expr", "lambdify", "preview", "S", "parsing", "init_session",
}
# A result that is not a plain number is printed as a sympy expression, but only when
# the program built a sympy object: a returned string is never parsed
_RUNNER = """
import math, fractions, decimal
{code}
_r = solution()
if isinstance(_r, (list, tuple)) and len(_r) == 1:
    _r = _r[0]
try:
    print(repr(float(_r)))
except (TypeError, ValueError):
    import sympy as _sp
    if not isinstance(_r, _sp.Basic):
        raise TypeError('solution() must return a number or a sympy expression')
    print('EXPR:' + _sp.sstr(_sp.simplify(_r)))
"""


def _option_for(value: str | None, options: dict[str, str]) -> str | None:
    """The single option whose numeric value matches, if any."""
    hits = [k for k, text in options.items() if values_close(option_value(text), value)]
    return hits[0] if len(hits) == 1 else None


def _as_float(value: str | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None  # a symbolic result such as '2*x*sin(x)'


def _is_integer(value: str) -> bool:
    number = _as_float(value)
    return number is not None and number.is_integer()


class ProgramVerifier:
    """Program-verified self-consistency for quantitative queries."""

    def __init__(self, ollama_client: OllamaClient, num_programs: int | None = None,
                 timeout_s: float = 20.0) -> None:  # sympy import + integration takes seconds
        self._client = ollama_client
        self._num_programs = num_programs or settings.program_vote_num_programs
        self._timeout_s = timeout_s
        logger.info("ProgramVerifier initialized | programs={}", num_programs)

    async def verify(
        self,
        query: str,
        plan_text: str,
        candidates: list[str],
        draft: str,
        exact: Calculation | None = None,
    ) -> tuple[str, list[str], JudgmentResult]:
        if exact is not None:
            return self._verify_exact(draft, exact, parse_options(query))

        advanced = is_advanced_math(query)
        # Advanced maths: alternate exact sympy programs with independent checks
        # (enumeration, numerical approximation), so a shared misreading is less likely
        styles = [
            ("check" if advanced and i % 2 else "advanced" if advanced else "basic")
            for i in range(self._num_programs)
        ]
        programs = list(await asyncio.gather(
            *(self._write_program(query, plan_text, seed=i, style=s) for i, s in enumerate(styles))
        ))
        results = list(await asyncio.gather(*(self._run(code) for code in programs)))

        # Self-debug round (Chen et al., 2024): programs that crashed or failed to
        # parse get the interpreter's error back once. Unsafe or empty programs don't.
        broken = [i for i, (ans, err) in enumerate(results)
                  if ans is None and programs[i] and not err.startswith(_UNSAFE_PREFIXES)]
        if broken:
            repaired = await asyncio.gather(
                *(self._repair(query, programs[i], results[i][1], seed=i) for i in broken)
            )
            rerun = await asyncio.gather(*(self._run(code) for code in repaired))
            for i, code, res in zip(broken, repaired, rerun):
                logger.debug("Program {} repaired: {} → {}", i + 1, results[i][1], res)
                programs[i], results[i] = code, res

        # Plausibility rules are for everyday word problems: probabilities, exact values
        # such as pi/4 and negative determinants are normal in advanced maths
        integer_task = not re.search(r"\d\.\d", query) and not advanced
        negatives_possible = bool(_NEGATIVE_CUES.search(query)) or advanced
        traces: list[str] = []
        program_answers: list[str] = []
        program_styles: list[str] = []
        for i, (code, (ans, err)) in enumerate(zip(programs, results), start=1):
            if ans is not None and integer_task and not _is_integer(ans):
                # Plausibility check: a problem stated in whole numbers that yields a
                # fractional count is almost always a mis-modelled program.
                ans, err = None, f"implausible non-integer answer {ans}"
            elif ans is not None and (_as_float(ans) or 0) < 0 and not negatives_possible:
                # Word-problem quantities (money, items, time) are non-negative; a
                # negative result means a subtraction was modelled backwards.
                ans, err = None, f"implausible negative answer {ans}"
            traces.append(
                f"[Program {i} {styles[i - 1]}] result={ans if ans is not None else err}\n{code}"
            )
            if ans is not None:
                program_answers.append(ans)
                program_styles.append(styles[i - 1])

        # Compare everything on canonical answers so '0.875' == '7/8' and
        # '142857.14' == '142857.142857' when voting
        options = parse_options(query)
        if options:
            # Multiple choice: vote on the option picked. A program counts for the
            # option whose value it computed; one matching no option is discarded.
            draft_answer = extract_choice(draft, options)
            nl_answers = [extract_choice(c, options) for c in candidates]
            kept = [(_option_for(a, options), s) for a, s in zip(program_answers, program_styles)]
            kept = [(k, s) for k, s in kept if k]
            program_answers, program_styles = [k for k, _ in kept], [s for _, s in kept]
        else:
            raw_nl = [extract_final_answer(c) for c in candidates]
            n_prog = len(program_answers)
            canon = canonicalize([extract_final_answer(draft), *program_answers, *raw_nl])
            draft_answer = canon[0]
            program_answers = canon[1:1 + n_prog]  # all non-None: only answers were kept
            nl_answers = canon[1 + n_prog:]
        final_line = (lambda k: f"({k}) {options[k]}") if options else (lambda k: k)
        prog_votes = Counter(program_answers)
        nl_votes = Counter(a for a in nl_answers if a is not None)
        traces.insert(0, f"[Samples] draft={draft_answer} answers={nl_answers}")

        if settings.program_vote_weighting == "sample_primary" and nl_votes:
            # Self-consistency over reasoning samples is the primary vote (Wang et al.,
            # 2023); programs add weight w each. Programs from one model share its
            # misreadings, so on dev their consensus was right only ~half the time
            # it disagreed with the draft — too unreliable to dominate.
            w = settings.program_vote_weight
            scores = {
                a: nl_votes.get(a, 0) + w * prog_votes.get(a, 0)
                for a in set(nl_votes) | set(prog_votes)
            }
            final, top = max(
                scores.items(),
                key=lambda kv: (kv[1], kv[0] == draft_answer, prog_votes.get(kv[0], 0)),
            )
            total = sum(scores.values())
            source = f"sample-primary vote (program weight {w})"
            # Programs alone overturn the draft only as two independent methods: at
            # least 3 programs agreeing, including both an exact (sympy) program and an
            # independent check. Seen live: 5 programs of one style that all misread a
            # cross product as a dot product outvoted a correct draft, while on another
            # problem the right answer came only from programs.
            backing_styles = {s for a, s in zip(program_answers, program_styles) if a == final}
            independent = prog_votes.get(final, 0) >= 3 and {"advanced", "check"} <= backing_styles
            if (final != draft_answer and draft_answer is not None
                    and nl_votes.get(final, 0) == 0 and not independent):
                logger.info("Only one kind of program supports {}; keeping draft {}",
                            final, draft_answer)
                final, top = draft_answer, scores.get(draft_answer, 1)
                source = "draft kept (program answer not confirmed independently)"
        elif prog_votes:
            # Natural-language answers only count when a program confirms them
            scores = {a: prog_votes[a] + nl_votes.get(a, 0) for a in prog_votes}
            # Ties keep the draft (do-no-harm), then favour more program agreement
            final, top = max(
                scores.items(),
                key=lambda kv: (kv[1], kv[0] == draft_answer, prog_votes[kv[0]]),
            )
            total = sum(scores.values())
            source = "program-verified vote"
            # Do-no-harm: one stray program with no agreeing sample is too weak to
            # overturn the draft. Overriding needs 2 programs, or 1 program + 1 sample.
            weak = prog_votes[final] < 2 and nl_votes.get(final, 0) < 1
            if final != draft_answer and weak and draft_answer is not None:
                logger.info(
                    "Program vote for {} too weak (programs={}, samples=0); keeping draft {}",
                    final, prog_votes[final], draft_answer,
                )
                final, top = draft_answer, nl_votes.get(draft_answer, 0)
                total = max(total, top)
                source = "draft kept (weak program support)"
        elif nl_votes:
            record_degradation(
                "layer3.program_vote",
                f"no program produced an answer ({[t.split(chr(10))[0] for t in traces]}); "
                "falling back to unverified self-consistency vote",
            )
            final, top = nl_votes.most_common(1)[0]
            total = sum(nl_votes.values())
            source = "self-consistency vote (no program executed)"
        else:
            return draft, traces, JudgmentResult(
                is_consistent=False, confidence=0.0,
                violations=["No program executed and no numeric candidate found"],
            )

        share = top / total if total else 0.0
        violations = []
        if draft_answer != final:
            violations.append(f"Draft answer {draft_answer} not confirmed; {source} selected {final}")
        judgment = JudgmentResult(
            is_consistent=draft_answer == final,
            confidence=min(max(share, 0.0), 1.0),
            violations=violations,
        )
        logger.info(
            "Program verification | programs_ok={}/{} | votes={} | draft={} | final={} ({})",
            len(program_answers), len(programs), dict(prog_votes), draft_answer, final, source,
        )

        if draft_answer == final:
            return draft, traces, judgment
        for cand, ans in zip(candidates, nl_answers):
            if ans == final and cand.strip():
                return f"{cand.strip()}\n\nFinal answer: {final_line(final)}", traces, judgment
        code = next(
            (c for c, (a, _) in zip(programs, results)
             if answers_match(a, final) or (options and _option_for(a, options) == final)),
            "",
        )
        return (
            f"Program-verified solution:\n```python\n{code}\n```\nFinal answer: {final_line(final)}",
            traces,
            judgment,
        )

    @staticmethod
    def _verify_exact(
        draft: str, exact: Calculation, options: dict[str, str] | None = None,
    ) -> tuple[str, list[str], JudgmentResult]:
        """The query is plain arithmetic: the exact calculator result is decisive."""
        trace = f"[Calculator] {exact.expression} = {exact.display}"
        exact_option = _option_for(exact.value, options) if options else None
        if exact_option:
            draft_answer = extract_choice(draft, options)
            agrees = draft_answer == exact_option
        else:
            draft_answer = extract_final_answer(draft)
            agrees = numbers_match(draft_answer, exact.value)
        if agrees:
            logger.info("Exact calculation confirms draft ({})", exact.display)
            return draft, [trace], JudgmentResult(is_consistent=True, confidence=1.0, violations=[])
        logger.info("Exact calculation overrides draft | draft={} | exact={}", draft_answer, exact.display)
        answer = (
            f"({exact_option}) {options[exact_option]}" if exact_option
            else exact.display.removeprefix("x = ")
        )
        return (
            f"{exact.display}\n\nFinal answer: {answer}",
            [trace],
            JudgmentResult(
                is_consistent=False,
                confidence=1.0,
                violations=[f"Draft answer {draft_answer} differs from exact result {exact.display}"],
            ),
        )

    # -------------------------------------------------------------------------

    async def _write_program(self, query: str, plan_text: str, seed: int,
                             style: str = "basic") -> str:
        system, examples = {
            "basic": (_PROGRAM_SYSTEM, _PROGRAM_EXAMPLES),
            "advanced": (_ADVANCED_SYSTEM, _ADVANCED_EXAMPLES),
            "check": (_CHECK_SYSTEM, _CHECK_EXAMPLES),
        }[style]
        try:
            res = await self._client.generate(
                prompt=_PROGRAM_PROMPT.format(examples=examples, query=query, plan=plan_text),
                system=system,
                temperature=0.0 if seed == 0 else 0.5,
                max_tokens=600 if style == "basic" else 900,
                seed=seed,
            )
        except Exception as exc:
            record_degradation("layer3.program_gen", f"program {seed} generation failed: {exc}")
            return ""
        return self._extract_code(res.text)

    async def _repair(self, query: str, code: str, error: str, seed: int) -> str:
        try:
            res = await self._client.generate(
                prompt=_REPAIR_PROMPT.format(query=query, code=code, error=error),
                system=_PROGRAM_SYSTEM,
                temperature=0.0,
                max_tokens=600,
                seed=seed,
            )
        except Exception as exc:
            record_degradation("layer3.program_repair", f"repair of program {seed} failed: {exc}")
            return code
        return self._extract_code(res.text)

    @staticmethod
    def _extract_code(text: str) -> str:
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        return (match.group(1) if match else text).strip()

    async def _run(self, code: str) -> tuple[str | None, str]:
        """One misbehaving program fails on its own; it never fails the whole query."""
        try:
            return await self._execute(code)
        except Exception as exc:
            record_degradation("layer3.program_exec", f"program result unreadable: {exc}")
            return None, f"error: {type(exc).__name__}: {exc}"

    async def _execute(self, code: str) -> tuple[str | None, str]:
        if "def solution" not in code:
            return None, "no solution() function"
        problem = self._unsafe_reason(code)
        if problem:
            return None, f"rejected: {problem}"
        # subprocess.run in a worker thread rather than asyncio.create_subprocess_exec:
        # the latter raises NotImplementedError on Windows SelectorEventLoop, which
        # uvicorn uses with --reload / workers, and that would silently fail every program.
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-I", "-c", _RUNNER.format(code=code)],
                capture_output=True,
                timeout=self._timeout_s,
            )
        except subprocess.TimeoutExpired:
            return None, "timeout"
        except Exception as exc:
            record_degradation("layer3.program_exec", f"sandbox could not start: {exc}")
            return None, f"exec error: {exc}"
        out, err = proc.stdout, proc.stderr
        if proc.returncode != 0:
            lines = err.decode(errors="ignore").strip().splitlines()
            return None, lines[-1] if lines else "runtime error"
        value = out.decode().strip().splitlines()[-1] if out.strip() else ""
        if value.startswith("EXPR:"):
            return value.removeprefix("EXPR:").strip() or None, "ok"
        try:
            num = float(value)
        except ValueError:
            return None, f"non-numeric output {value!r}"
        if not math.isfinite(num):
            return None, f"non-finite result {value}"  # e.g. a limit mis-computed as oo
        if abs(num - round(num)) < 1e-6:
            num = round(num)
        return normalize_number(str(num)), "ok"

    @staticmethod
    def _unsafe_reason(code: str) -> str | None:
        """Static allow-list check before execution."""
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"syntax error: {exc.msg}"
        allowed = ", ".join(sorted(_ALLOWED_IMPORTS))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if set(a.name.split(".")) & _BLOCKED_NAMES:
                        return f"blocked name in import {a.name}"
                    if a.name.split(".")[0] not in _ALLOWED_IMPORTS:
                        return f"disallowed import {a.name}; use only: {allowed}"
            elif isinstance(node, ast.ImportFrom):
                parts = (node.module or "").split(".")
                if set(parts) & _BLOCKED_NAMES or any(a.name in _BLOCKED_NAMES for a in node.names):
                    return f"blocked name in import {node.module}"
                if parts[0] not in _ALLOWED_IMPORTS:
                    return f"disallowed import {node.module}; use only: {allowed}"
            elif isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
                return f"blocked name {node.id}"
            elif isinstance(node, ast.Attribute) and node.attr in _BLOCKED_NAMES:
                return f"blocked name {node.attr}"
            elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                return "private attribute access"
            elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
                return "disallowed construct"
        return None
