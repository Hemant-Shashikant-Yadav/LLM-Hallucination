"""
Layer 1: Task router — decides what kind of question this is.

  math            → numeric/symbolic clustering, Layer 3 (programs + self-consistency)
  factual         → NLI clustering, Layer 2 (evidence-gated edit)
  time_sensitive  → never the fast path; Layer 2 answers from fresh web evidence

Clear-cut questions are decided by rules (no LLM call): exact expressions, word
problems, math keywords, "current/latest/today". Only ambiguous questions (e.g.
"What happened on July 20, 1969?", which has numbers but is not math) are classified
by the pipeline model itself, keeping the single-model design.
"""

from __future__ import annotations

import datetime
import json
import re

from loguru import logger

from app.answer_utils import TASK_TYPES, classify_task_heuristic
from app.config import settings
from app.llm_clients.ollama_client import OllamaClient
from app.telemetry import record_degradation

_ROUTER_SYSTEM = """Classify the user's question into exactly one type:
- "math": the answer is a number or expression that must be COMPUTED (arithmetic,
  algebra, unit conversion, percentages, word problems with quantities).
- "time_sensitive": asks about the present state of something that changes over time
  and needs up-to-date information (who holds an office now, latest version, today's
  price, ongoing events).
- "factual": everything else: facts, history (including past dates and events),
  science, definitions, explanations, advice, and comparisons or lookups that need no
  calculation (e.g. "Which is taller, X (8849 m) or Y (8611 m)?").
Respond with JSON only: {"type": "math" | "time_sensitive" | "factual"}"""

_PAST_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


class TaskRouter:
    """Rule-first task classification with an LLM decision for ambiguous queries."""

    def __init__(self, ollama_client: OllamaClient) -> None:
        self._client = ollama_client
        logger.info("TaskRouter initialized | mode={}", settings.task_router_mode)

    async def classify(self, query: str) -> str:
        label, confident = classify_task_heuristic(query)
        if confident or settings.task_router_mode == "heuristic":
            logger.info("Task route: {} (rules)", label)
            return label
        try:
            result = await self._client.generate_json(
                prompt=f"QUESTION: {query}",
                system=_ROUTER_SYSTEM,
                temperature=0.0,
                max_tokens=30,
            )
            llm_label = str(json.loads(result.text).get("type", "")).strip().lower()
        except Exception as exc:
            record_degradation(
                "layer1.task_router", f"LLM classification failed ({exc}); using rule label {label}"
            )
            return label
        if llm_label not in TASK_TYPES:
            record_degradation(
                "layer1.task_router", f"unexpected label {llm_label!r}; using rule label {label}"
            )
            return label
        if llm_label == "time_sensitive" and self._names_past_year(query):
            # "What happened on July 20, 1969?" is history, not current events; the
            # rules already caught genuine present-tense cues ("current", "latest")
            logger.info("Task route: model said time_sensitive, but the query names a past year")
            return "factual"
        if llm_label == "time_sensitive" and self._gives_quantities(query):
            # "If Mars has a 697-day year and a 12-day week, which year …" is computed from
            # the numbers it gives, not looked up (MATH ladder item numb190 was misrouted)
            logger.info("Task route: model said time_sensitive, but the query gives the data")
            return "math"
        logger.info("Task route: {} (model; rules guessed {})", llm_label, label)
        return llm_label

    @staticmethod
    def _gives_quantities(query: str) -> bool:
        numbers = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", query)
        return len([n for n in numbers if not _PAST_YEAR.fullmatch(n)]) >= 2

    @staticmethod
    def _names_past_year(query: str) -> bool:
        this_year = datetime.date.today().year
        return any(int(y) <= this_year - 2 for y in _PAST_YEAR.findall(query))
