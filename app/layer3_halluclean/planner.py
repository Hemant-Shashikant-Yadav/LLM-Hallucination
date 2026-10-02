"""
Layer 3: HalluClean Symbolic Task-Oriented Planner (Stage 1).

Generates a structured, symbolic execution graph from the user query.
The plan decomposes complex reasoning tasks into an ordered DAG of
discrete steps, each with explicit dependencies.
"""

from __future__ import annotations

import json
import re

from loguru import logger

from app.config import settings
from app.llm_clients.ollama_client import OllamaClient
from app.models.schemas import ExecutionGraph, PlanNode


# System prompt for plan generation
_PLANNER_SYSTEM = """You are a precise reasoning planner. Given a question or task, 
you must decompose it into a structured execution plan.

Rules:
1. Break the task into atomic, sequential reasoning steps.
2. Each step should be a clear, actionable sub-task.
3. Specify dependencies between steps (which steps must complete first).
4. Keep the plan concise — no more than {max_steps} steps.
5. Each step should be independently executable with a clear expected output.

Respond with a JSON object in this format:
{{
    "steps": [
        {{
            "step_id": 1,
            "description": "Clear description of what this step does",
            "dependencies": []
        }},
        {{
            "step_id": 2,
            "description": "Next step description",
            "dependencies": [1]
        }}
    ]
}}
"""

_PLANNER_PROMPT = """Create a structured reasoning plan for the following query.

QUERY: {query}

Return ONLY the JSON plan. No explanation or preamble."""


class HalluCleanPlanner:
    """
    Stage 1 of HalluClean: Task-Oriented Planning.

    Generates a symbolic execution graph that decomposes a complex
    query into ordered, dependency-aware reasoning steps.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        max_steps: int | None = None,
    ) -> None:
        self._client = ollama_client
        self._max_steps = max_steps or settings.halluclean_max_steps
        logger.info("HalluCleanPlanner initialized | max_steps={}", self._max_steps)

    async def generate_plan(self, query: str) -> ExecutionGraph:
        """
        Generate a symbolic execution plan for the given query.

        Args:
            query: The user query requiring structured reasoning.

        Returns:
            ExecutionGraph with ordered PlanNode steps.
        """
        try:
            result = await self._client.generate_json(
                prompt=_PLANNER_PROMPT.format(query=query),
                system=_PLANNER_SYSTEM.format(max_steps=self._max_steps),
                temperature=settings.halluclean_temperature,
                max_tokens=2048,
            )

            graph = self._parse_plan(result.text, query)

            logger.info(
                "Generated execution plan | query={:.50}... | steps={}",
                query,
                len(graph.steps),
            )

            return graph

        except Exception as exc:
            logger.error("Plan generation failed: {}", exc)
            return self._fallback_plan(query)

    def _parse_plan(self, response_text: str, query: str) -> ExecutionGraph:
        """Parse the JSON plan from the LLM response."""
        try:
            parsed = json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON object from the response
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("Could not parse JSON plan from LLM response")
                    return self._fallback_plan(query)
            else:
                return self._fallback_plan(query)

        steps_data = parsed.get("steps", [])
        if not steps_data:
            return self._fallback_plan(query)

        steps = []
        for i, step_data in enumerate(steps_data[: self._max_steps]):
            steps.append(
                PlanNode(
                    step_id=step_data.get("step_id", i + 1),
                    description=step_data.get("description", f"Step {i + 1}"),
                    dependencies=step_data.get("dependencies", []),
                    status="pending",
                )
            )

        return ExecutionGraph(
            query=query,
            steps=steps,
            total_steps=len(steps),
        )

    def _fallback_plan(self, query: str) -> ExecutionGraph:
        """Generate a simple 3-step fallback plan."""
        logger.warning("Using fallback plan for query: {:.50}...", query)

        steps = [
            PlanNode(
                step_id=1,
                description="Analyze the query and identify key components that need verification",
                dependencies=[],
                status="pending",
            ),
            PlanNode(
                step_id=2,
                description="Reason through each component step by step, checking logical consistency",
                dependencies=[1],
                status="pending",
            ),
            PlanNode(
                step_id=3,
                description="Synthesize the verified components into a coherent, factual answer",
                dependencies=[2],
                status="pending",
            ),
        ]

        return ExecutionGraph(
            query=query,
            steps=steps,
            total_steps=len(steps),
        )
