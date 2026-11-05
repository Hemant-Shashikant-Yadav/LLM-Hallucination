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
import re
import subprocess
import sys
from collections import Counter

from loguru import logger

from app.answer_utils import extract_final_number, normalize_number
from app.llm_clients.ollama_client import OllamaClient
from app.models.schemas import JudgmentResult
from app.telemetry import record_degradation

_PROGRAM_SYSTEM = """You solve math word problems by writing a short Python program.

Rules:
1. Define a function `solution()` that returns the final numeric answer.
2. Use one clearly named variable per quantity in the problem, with a comment.
3. Follow the given plan step by step. Use only numbers stated in the problem.
4. Use plain Python arithmetic only (the `math` module is allowed). No input(), no printing, no files.
5. Return a number, not a string.
Respond with the Python code only, inside a ```python code block."""

_PROGRAM_PROMPT = """PROBLEM: {query}

PLAN:
{plan}

Write the program."""

_ALLOWED_IMPORTS = {"math", "fractions", "decimal"}
_BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals", "locals",
    "getattr", "setattr", "delattr", "vars", "dir", "breakpoint", "exit", "quit",
    "help", "memoryview", "classmethod", "staticmethod", "super", "type",
}
_RUNNER = """
import math, fractions, decimal
{code}
_r = solution()
print(repr(float(_r)))
"""


def _is_integer(value: str) -> bool:
    try:
        return float(value).is_integer()
    except ValueError:
        return False


class ProgramVerifier:
    """Program-verified self-consistency for quantitative queries."""

    def __init__(self, ollama_client: OllamaClient, num_programs: int = 3,
                 timeout_s: float = 5.0) -> None:
        self._client = ollama_client
        self._num_programs = num_programs
        self._timeout_s = timeout_s
        logger.info("ProgramVerifier initialized | programs={}", num_programs)

    async def verify(
        self,
        query: str,
        plan_text: str,
        candidates: list[str],
        draft: str,
    ) -> tuple[str, list[str], JudgmentResult]:
        programs = await asyncio.gather(
            *(self._write_program(query, plan_text, seed=i) for i in range(self._num_programs))
        )
        results = await asyncio.gather(*(self._run(code) for code in programs))

        integer_task = not re.search(r"\d\.\d", query)
        traces: list[str] = []
        program_answers: list[str] = []
        for i, (code, (ans, err)) in enumerate(zip(programs, results), start=1):
            if ans is not None and integer_task and not _is_integer(ans):
                # Plausibility check: a problem stated in whole numbers that yields a
                # fractional count is almost always a mis-modelled program.
                ans, err = None, f"implausible non-integer answer {ans}"
            traces.append(f"[Program {i}] result={ans if ans is not None else err}\n{code}")
            if ans is not None:
                program_answers.append(ans)

        nl_answers = [extract_final_number(c) for c in candidates]
        draft_answer = extract_final_number(draft)
        prog_votes = Counter(program_answers)
        nl_votes = Counter(a for a in nl_answers if a is not None)

        if prog_votes:
            # Natural-language answers only count when a program confirms them
            scores = {a: prog_votes[a] + nl_votes.get(a, 0) for a in prog_votes}
            final, top = max(scores.items(), key=lambda kv: (kv[1], prog_votes[kv[0]]))
            total = sum(scores.values())
            source = "program-verified vote"
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
                return f"{cand.strip()}\n\nFinal answer: {final}", traces, judgment
        code = next((c for c, (a, _) in zip(programs, results) if a == final), "")
        return (
            f"Program-verified solution:\n```python\n{code}\n```\nFinal answer: {final}",
            traces,
            judgment,
        )

    # -------------------------------------------------------------------------

    async def _write_program(self, query: str, plan_text: str, seed: int) -> str:
        try:
            res = await self._client.generate(
                prompt=_PROGRAM_PROMPT.format(query=query, plan=plan_text),
                system=_PROGRAM_SYSTEM,
                temperature=0.0 if seed == 0 else 0.5,
                max_tokens=600,
                seed=seed,
            )
        except Exception as exc:
            record_degradation("layer3.program_gen", f"program {seed} generation failed: {exc}")
            return ""
        match = re.search(r"```(?:python)?\s*\n(.*?)```", res.text, re.DOTALL)
        return (match.group(1) if match else res.text).strip()

    async def _run(self, code: str) -> tuple[str | None, str]:
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
        try:
            num = float(value)
        except ValueError:
            return None, f"non-numeric output {value!r}"
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
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] not in _ALLOWED_IMPORTS for a in node.names):
                    return "disallowed import"
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] not in _ALLOWED_IMPORTS:
                    return "disallowed import"
            elif isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
                return f"blocked name {node.id}"
            elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                return "private attribute access"
            elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
                return "disallowed construct"
        return None
