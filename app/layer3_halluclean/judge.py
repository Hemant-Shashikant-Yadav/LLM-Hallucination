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

from app.config import settings
from app.llm_clients.ollama_client import OllamaClient
from app.models.schemas import (
    ExecutionGraph,
    JudgmentResult,
    Layer3Result,
)


# =============================================================================
# Stage 3: Judgment Prompts
# =============================================================================

_JUDGE_SYSTEM = """You are a rigorous logical consistency judge. Your task is to evaluate 
whether a multi-step reasoning process produced a logically sound conclusion.

You must check:
1. Does each step follow logically from its predecessors?
2. Are there any contradictions between steps?
3. Does the final conclusion accurately reflect the reasoning chain?
4. Are there any unsupported claims or logical leaps?

Respond with a JSON object:
{{
    "is_consistent": true/false,
    "confidence": 0.0-1.0,
    "violations": ["description of violation 1", ...]
}}
"""

_JUDGE_PROMPT = """Evaluate the logical consistency of the following reasoning chain.

ORIGINAL QUERY: {query}

REASONING CHAIN:
{reasoning_traces}

Judge whether the reasoning is logically sound and identify any violations.
Return ONLY the JSON evaluation."""


# =============================================================================
# Stage 4: Refinement Prompts
# =============================================================================

_REFINER_SYSTEM = """You are a precise content refiner. Your task is to rewrite a 
verified reasoning chain into a clear, polished natural language response.

Rules:
1. Preserve all factual content from the verified reasoning.
2. Remove meta-commentary about the reasoning process itself.
3. Write in a natural, authoritative tone.
4. If any violations were identified, address them in the refined response.
5. Do NOT add any new information beyond what was verified.
"""

_REFINER_PROMPT = """Rewrite the following verified reasoning into a polished response.

ORIGINAL QUERY: {query}

VERIFIED REASONING:
{reasoning_traces}

{violation_context}

Write a clear, natural response that answers the original query."""


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

        # Stage 4: Refinement
        refined = await self._refine(
            graph.query,
            reasoning_traces,
            judgment,
        )

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
            logger.error("Judgment failed: {}", exc)
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
                    return JudgmentResult(
                        is_consistent=True,
                        violations=[],
                        confidence=0.5,
                    )
            else:
                return JudgmentResult(
                    is_consistent=True,
                    violations=[],
                    confidence=0.5,
                )

        return JudgmentResult(
            is_consistent=parsed.get("is_consistent", True),
            confidence=float(parsed.get("confidence", 0.5)),
            violations=parsed.get("violations", []),
        )

    # -------------------------------------------------------------------------
    # Stage 4: Content Refinement
    # -------------------------------------------------------------------------

    async def _refine(
        self,
        query: str,
        reasoning_traces: list[str],
        judgment: JudgmentResult,
    ) -> str:
        """Rewrite the verified reasoning into polished natural language."""
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
                temperature=0.1,
                max_tokens=2048,
            )

            return result.text.strip()

        except Exception as exc:
            logger.error("Refinement failed: {}", exc)
            # Fallback: concatenate the last step outputs
            if reasoning_traces:
                return reasoning_traces[-1]
            return "[Refinement failed. Please try again.]"
