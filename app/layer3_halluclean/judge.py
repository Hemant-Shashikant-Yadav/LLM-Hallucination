"""
Layer 3: HalluClean Judge & Content Refiner (Stages 3 & 4).

Stage 3 — Final Judgment:
    Compares the synthesized conclusion against original constraints
    and checks for logical consistency.

Stage 4 — Content Refinement:
    Rewrites the verified logical path into polished natural language.
"""

from __future__ import annotations

import json
import re
import time

from loguru import logger

from app.answer_utils import extract_final_number, is_numeric_reasoning_query
from app.config import settings
from app.llm_clients.ollama_client import OllamaClient
from app.models.schemas import (
    ExecutionGraph,
    JudgmentResult,
    Layer3Result,
)
from app.telemetry import record_degradation


# =============================================================================
# Stage 3: Judgment Prompts
# =============================================================================

_JUDGE_SYSTEM = """You are a rigorous verifier of step-by-step solutions.

You must check:
1. Recompute EVERY arithmetic operation. Any wrong result is a violation.
2. Check that every quantity used is actually stated in the query.
3. Check that the solution answers exactly what the query asks (units, totals, remaining amounts).
4. Check that the final answer follows from the steps.

Respond with a JSON object:
{
    "is_consistent": true/false,
    "confidence": 0.0-1.0,
    "violations": ["description of violation 1", ...]
}
Report only real errors. If the solution is correct, return "is_consistent": true and an empty list.
"""

_JUDGE_PROMPT = """Verify the following solution.

ORIGINAL QUERY: {query}

SOLUTION:
{reasoning_traces}

Return ONLY the JSON evaluation."""


# =============================================================================
# Stage 4: Refinement Prompts
# =============================================================================

_REFINER_SYSTEM = """You are a careful problem solver who corrects flawed solutions.

Rules:
1. Fix every identified issue. Recompute all arithmetic explicitly.
2. Use only quantities stated in the query.
3. Show the corrected steps concisely, labelled "Step N:".
4. End with exactly one line in the form "Final answer: <answer>".
   For numeric questions, <answer> is a single number with no units.
"""

_REFINER_PROMPT = """ORIGINAL QUERY: {query}

FLAWED SOLUTION:
{reasoning_traces}

{violation_context}

Write the corrected solution, then the final answer line."""


class HalluCleanJudge:
    """
    Stages 3 & 4 of HalluClean: Judgment and Content Refinement.

    Evaluates logical consistency of the execution chain and
    rewrites the result into polished natural language.
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        self._client = ollama_client
        logger.info("HalluCleanJudge initialized")

    async def judge_and_refine(
        self,
        graph: ExecutionGraph,
        reasoning_traces: list[str],
        draft_response: str = "",
    ) -> Layer3Result:
        """
        Run Stages 3 and 4: Judge the reasoning chain and refine the output.

        Args:
            graph: The executed plan graph.
            reasoning_traces: Step-by-step reasoning outputs.

        Returns:
            Layer3Result with judgment, refined response, and full audit.
        """
        start = time.perf_counter()

        # Stage 3: Judgment
        judgment = await self._judge(graph.query, reasoning_traces)

        # Stage 4: Refinement (only when the judge found violations)
        if judgment.is_consistent and not judgment.violations:
            verified = "\n\n".join(reasoning_traces)
        else:
            verified = await self._refine(graph.query, reasoning_traces, judgment)

        refined = self._select_final(graph.query, verified, draft_response)

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = Layer3Result(
            execution_graph=graph,
            reasoning_traces=reasoning_traces,
            judgment=judgment,
            refined_response=refined,
            latency_ms=elapsed_ms,
        )

        logger.info(
            "Judge & refine complete | consistent={} | confidence={:.2f} | "
            "violations={} | {:.1f}ms",
            judgment.is_consistent,
            judgment.confidence,
            len(judgment.violations),
            elapsed_ms,
        )

        return result

    # -------------------------------------------------------------------------
    # Stage 3: Final Judgment
    # -------------------------------------------------------------------------

    async def _judge(
        self,
        query: str,
        reasoning_traces: list[str],
    ) -> JudgmentResult:
        """Evaluate the logical consistency of the reasoning chain."""
        traces_text = "\n\n".join(reasoning_traces)

        try:
            result = await self._client.generate_json(
                prompt=_JUDGE_PROMPT.format(
                    query=query,
                    reasoning_traces=traces_text,
                ),
                system=_JUDGE_SYSTEM,
                temperature=0.0,
                max_tokens=1024,
            )

            return self._parse_judgment(result.text)

        except Exception as exc:
            record_degradation("layer3.judge", f"judge call failed, solution passed unchecked: {exc}")
            return JudgmentResult(
                is_consistent=True,
                violations=[],
                confidence=0.5,
            )

    def _parse_judgment(self, response_text: str) -> JudgmentResult:
        """Parse the judgment JSON from the LLM response."""
        try:
            parsed = json.loads(response_text.strip())
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    record_degradation("layer3.judge", "unparseable judge JSON, solution passed unchecked")
                    return JudgmentResult(
                        is_consistent=True,
                        violations=[],
                        confidence=0.5,
                    )
            else:
                record_degradation("layer3.judge", "no JSON in judge output, solution passed unchecked")
                return JudgmentResult(
                    is_consistent=True,
                    violations=[],
                    confidence=0.5,
                )

        is_consistent = parsed.get("is_consistent", True)
        if isinstance(is_consistent, str):
            is_consistent = is_consistent.strip().lower() == "true"
        try:
            confidence = min(max(float(parsed.get("confidence", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            confidence = 0.5
        violations = parsed.get("violations", []) or []
        if not isinstance(violations, list):
            violations = [violations]
        return JudgmentResult(
            is_consistent=bool(is_consistent),
            confidence=confidence,
            violations=[str(v) for v in violations if str(v).strip()],
        )

    # -------------------------------------------------------------------------
    # Stage 4: Content Refinement
    # -------------------------------------------------------------------------

    @staticmethod
    def _select_final(query: str, verified: str, draft_response: str) -> str:
        """
        Choose the final response.

        For quantitative queries the draft is kept only if its final number
        matches the verified chain (no degradation of correct drafts);
        otherwise the verified chain wins. For other queries the verified
        chain is returned.
        """
        if not verified.strip() or verified.startswith("[Error"):
            return draft_response or verified
        if draft_response and is_numeric_reasoning_query(query):
            verified_ans = extract_final_number(verified)
            draft_ans = extract_final_number(draft_response)
            if verified_ans is not None and verified_ans == draft_ans:
                logger.info("Verified chain agrees with draft ({}), keeping draft", draft_ans)
                return draft_response
            logger.info(
                "Verified chain overrides draft | draft={} | verified={}",
                draft_ans,
                verified_ans,
            )
        return verified

    async def _refine(
        self,
        query: str,
        reasoning_traces: list[str],
        judgment: JudgmentResult,
    ) -> str:
        """Correct the reasoning chain using the judge's violations."""
        traces_text = "\n\n".join(reasoning_traces)

        # Build violation context if any were found
        violation_context = ""
        if judgment.violations:
            violations_text = "\n".join(
                f"- {v}" for v in judgment.violations
            )
            violation_context = (
                f"IDENTIFIED ISSUES (address these in the refined response):\n"
                f"{violations_text}"
            )

        try:
            result = await self._client.generate(
                prompt=_REFINER_PROMPT.format(
                    query=query,
                    reasoning_traces=traces_text,
                    violation_context=violation_context,
                ),
                system=_REFINER_SYSTEM,
                temperature=0.0,
                max_tokens=1536,
            )

            return result.text.strip()

        except Exception as exc:
            record_degradation("layer3.refine", f"refinement failed: {exc}")
            # Fallback: concatenate the last step outputs
            if reasoning_traces:
                return reasoning_traces[-1]
            return "[Refinement failed. Please try again.]"
