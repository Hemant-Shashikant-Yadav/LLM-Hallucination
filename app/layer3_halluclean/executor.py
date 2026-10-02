"""
Layer 3: HalluClean Plan-Guided Executor (Stage 2).

Executes each step of the symbolic execution graph sequentially.
Each step runs within local context boundaries to prevent cross-step
information leakage, producing traceable reasoning traces.
"""

from __future__ import annotations

from loguru import logger

from app.config import settings
from app.llm_clients.ollama_client import OllamaClient
from app.models.schemas import ExecutionGraph, PlanNode


# System prompt for step execution
_EXECUTOR_SYSTEM = """You are a precise step-by-step reasoner. You are executing ONE step 
of a multi-step reasoning plan. 

Rules:
1. Focus ONLY on this specific step — do not anticipate future steps.
2. Be thorough but concise in your reasoning.
3. If this step depends on previous step outputs, use them as context.
4. Clearly state your conclusion for this step.
5. Do NOT hallucinate or fabricate information. If unsure, state your uncertainty.
"""

_EXECUTOR_PROMPT = """You are executing Step {step_id} of a reasoning plan.

ORIGINAL QUERY: {query}

STEP DESCRIPTION: {description}

{dependencies_context}

Execute this step carefully and provide your reasoning and conclusion."""


class HalluCleanExecutor:
    """
    Stage 2 of HalluClean: Plan-Guided Sequential Reasoning.

    Executes each plan node in dependency order, maintaining
    isolated context boundaries per step.
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        self._client = ollama_client
        logger.info("HalluCleanExecutor initialized")

    async def execute_plan(
        self,
        graph: ExecutionGraph,
    ) -> tuple[ExecutionGraph, list[str]]:
        """
        Execute all steps in the execution graph in dependency order.

        Args:
            graph: The execution graph from the Planner.

        Returns:
            Tuple of (updated graph, reasoning traces).
        """
        reasoning_traces: list[str] = []
        step_outputs: dict[int, str] = {}

        # Topological execution order (respecting dependencies)
        execution_order = self._topological_sort(graph.steps)

        for step_id in execution_order:
            step = next(s for s in graph.steps if s.step_id == step_id)
            step.status = "running"

            try:
                # Build context from completed dependencies
                dep_context = self._build_dependency_context(
                    step.dependencies, step_outputs
                )

                # Execute the step
                output = await self._execute_step(
                    step=step,
                    query=graph.query,
                    dependencies_context=dep_context,
                )

                step.output = output
                step.status = "completed"
                step_outputs[step.step_id] = output
                reasoning_traces.append(
                    f"[Step {step.step_id}] {step.description}\n{output}"
                )

                logger.debug(
                    "Step {} completed | output_len={}",
                    step.step_id,
                    len(output),
                )

            except Exception as exc:
                step.status = "failed"
                step.output = f"[Error: {exc}]"
                reasoning_traces.append(
                    f"[Step {step.step_id}] FAILED: {exc}"
                )
                logger.error("Step {} failed: {}", step.step_id, exc)

        logger.info(
            "Plan execution complete | steps={} | completed={}",
            len(graph.steps),
            sum(1 for s in graph.steps if s.status == "completed"),
        )

        return graph, reasoning_traces

    async def _execute_step(
        self,
        step: PlanNode,
        query: str,
        dependencies_context: str,
    ) -> str:
        """Execute a single plan step via the LLM."""
        prompt = _EXECUTOR_PROMPT.format(
            step_id=step.step_id,
            query=query,
            description=step.description,
            dependencies_context=dependencies_context,
        )

        result = await self._client.generate(
            prompt=prompt,
            system=_EXECUTOR_SYSTEM,
            temperature=settings.halluclean_temperature,
            max_tokens=1024,
        )

        return result.text.strip()

    @staticmethod
    def _build_dependency_context(
        dependencies: list[int],
        step_outputs: dict[int, str],
    ) -> str:
        """Build context string from completed dependency outputs."""
        if not dependencies:
            return "No previous steps to reference."

        context_parts = []
        for dep_id in dependencies:
            if dep_id in step_outputs:
                context_parts.append(
                    f"PREVIOUS STEP {dep_id} OUTPUT:\n{step_outputs[dep_id]}"
                )

        if not context_parts:
            return "No previous step outputs available."

        return "\n\n".join(context_parts)

    @staticmethod
    def _topological_sort(steps: list[PlanNode]) -> list[int]:
        """
        Topological sort of plan steps based on dependencies.

        Returns step IDs in valid execution order.
        """
        # Build adjacency and in-degree maps
        in_degree: dict[int, int] = {}
        adjacency: dict[int, list[int]] = {}

        for step in steps:
            in_degree.setdefault(step.step_id, 0)
            adjacency.setdefault(step.step_id, [])
            for dep in step.dependencies:
                adjacency.setdefault(dep, []).append(step.step_id)
                in_degree[step.step_id] = in_degree.get(step.step_id, 0) + 1

        # Kahn's algorithm
        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        order: list[int] = []

        while queue:
            # Sort for deterministic ordering
            queue.sort()
            current = queue.pop(0)
            order.append(current)

            for neighbor in adjacency.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If we couldn't sort all steps, append remaining in original order
        remaining = [s.step_id for s in steps if s.step_id not in order]
        order.extend(remaining)

        return order
