"""
Per-request telemetry: trace IDs, stage boundaries, and degradation records.

Every query gets a trace ID that is bound to all log lines emitted on its
behalf (including inside asyncio.gather tasks, which copy the context).
Components call `record_degradation` whenever they fall back instead of
failing, so silent fallbacks show up in both the logs and the audit trail.
"""

from __future__ import annotations

import contextvars
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator

from loguru import logger

_degradations: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "degradations", default=None
)
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


@contextmanager
def request_trace(trace_id: str | None = None) -> Iterator[str]:
    """Open a trace for one query: binds trace_id to logs and starts a degradation list."""
    trace_id = trace_id or uuid.uuid4().hex[:12]
    deg_token = _degradations.set([])
    id_token = _trace_id.set(trace_id)
    try:
        with logger.contextualize(trace_id=trace_id):
            yield trace_id
    finally:
        _degradations.reset(deg_token)
        _trace_id.reset(id_token)


def current_trace_id() -> str:
    return _trace_id.get()


def record_degradation(stage: str, reason: str) -> None:
    """A stage fell back instead of doing its job. Logged and kept for the audit trail."""
    entry = f"{stage}: {reason}"
    logger.warning("DEGRADED | {}", entry)
    items = _degradations.get()
    if items is not None:
        items.append(entry)


def degradations() -> list[str]:
    return list(_degradations.get() or [])


@asynccontextmanager
async def stage(name: str, **fields) -> AsyncIterator[dict]:
    """
    Log entry/exit of a pipeline stage with its duration.

    The yielded dict can be filled with outcome fields that are logged on exit.
    Exceptions are logged with the stage name and re-raised — never swallowed.
    """
    outcome: dict = {}
    logger.info("STAGE_ENTER | {} | {}", name, fields or "")
    start = time.perf_counter()
    try:
        yield outcome
    except BaseException as exc:
        logger.error(
            "STAGE_FAIL | {} | {:.0f}ms | {}: {}",
            name, (time.perf_counter() - start) * 1000, type(exc).__name__, exc,
        )
        raise
    logger.info(
        "STAGE_EXIT | {} | {:.0f}ms | {}",
        name, (time.perf_counter() - start) * 1000, outcome or "",
    )
