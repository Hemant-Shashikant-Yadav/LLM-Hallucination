"""
Model backends. `OllamaBackend` talks to a real Ollama server; `FakeBackend`
(generate/fake.py) simulates one so the whole study can be dry-run without a model.

Both return `Generation` records with the per-token log-probabilities the token-level
uncertainty methods need. A server that sends no log-probs yields empty lists, and the
methods that need them report None instead of failing.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger

from uncertainty.settings import settings


@dataclass
class Generation:
    text: str
    tokens: list[str] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    top_logprobs: list[list[list]] = field(default_factory=list)  # per token: [[token, logprob], ...]
    done_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    @property
    def truncated(self) -> bool:
        """Stopped at the token limit: a cut-off worked answer has no final answer."""
        return self.done_reason == "length"

    def as_record(self, top: bool = False) -> dict:
        d = round_lp
        rec = {
            "text": self.text,
            "tokens": self.tokens,
            "logprobs": [d(x) for x in self.logprobs],
            "done_reason": self.done_reason,
            "eval_count": self.completion_tokens,
            "prompt_count": self.prompt_tokens,
            "seconds": round(self.seconds, 2),
        }
        if top:
            rec["top_logprobs"] = [[[t, d(lp)] for t, lp in alts] for alts in self.top_logprobs]
        return rec


def round_lp(x: float) -> float:
    return round(float(x), settings.logprob_decimals)


class Backend:
    """Interface shared by the real and the simulated model."""

    name = "base"

    async def generate(self, model: str, prompt: str, *, system: str = "", temperature: float = 0.0,
                       max_tokens: int = 256, seed: int | None = None, logprobs: bool = False,
                       top_logprobs: int = 0, fmt: str | None = None, top_k: int | None = None,
                       think: bool | None = None) -> Generation:
        raise NotImplementedError

    async def version(self) -> str:
        raise NotImplementedError

    async def available(self) -> list[str]:
        raise NotImplementedError

    async def loaded(self) -> list[str]:
        raise NotImplementedError

    async def pull(self, model: str) -> None:
        raise NotImplementedError

    async def unload(self, model: str) -> None:
        raise NotImplementedError


def normalize_tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


# Transient failures worth a retry (runner crashed while loading, connection dropped).
# Timeouts are not retried: they already cost the full timeout.
_TRANSIENT = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, base_url: str | None = None, concurrency: int | None = None) -> None:
        from ollama import AsyncClient

        self._base_url = base_url or settings.ollama_base_url
        self._client = AsyncClient(host=self._base_url, timeout=settings.ollama_timeout)
        self._concurrency = concurrency or settings.concurrency
        self._gates: dict[int, asyncio.Semaphore] = {}

    def _gate(self) -> asyncio.Semaphore:
        # One gate per event loop: requests beyond the server's parallel slots wait here,
        # not in the server queue where they would burn their HTTP timeout.
        key = id(asyncio.get_running_loop())
        if key not in self._gates:
            self._gates[key] = asyncio.Semaphore(self._concurrency)
        return self._gates[key]

    async def generate(self, model: str, prompt: str, *, system: str = "", temperature: float = 0.0,
                       max_tokens: int = 256, seed: int | None = None, logprobs: bool = False,
                       top_logprobs: int = 0, fmt: str | None = None, top_k: int | None = None,
                       think: bool | None = None) -> Generation:
        from ollama import ResponseError

        options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens,
                                   "num_ctx": settings.num_ctx}
        if seed is not None:
            options["seed"] = seed
        if top_k is not None:
            options["top_k"] = top_k
        kwargs: dict[str, Any] = {"model": model, "prompt": prompt, "system": system, "options": options,
                                  "stream": False, "keep_alive": settings.ollama_keep_alive}
        if logprobs or top_logprobs:
            kwargs["logprobs"] = True
        if top_logprobs:
            kwargs["top_logprobs"] = top_logprobs
        if fmt:
            kwargs["format"] = fmt
        if think is not None:
            kwargs["think"] = think
        for attempt in range(settings.max_retries + 1):
            try:
                async with self._gate():
                    start = time.perf_counter()
                    resp = await self._client.generate(**kwargs)
                    seconds = time.perf_counter() - start
                return self._to_generation(resp, seconds)
            except ResponseError as exc:
                if exc.status_code < 500 or attempt == settings.max_retries:
                    raise
                reason = f"HTTP {exc.status_code}: {exc.error}"
            except _TRANSIENT as exc:
                if attempt == settings.max_retries:
                    raise
                reason = f"{type(exc).__name__}: {exc}"
            logger.warning("Ollama call on {} failed (attempt {}), retrying | {}", model, attempt + 1, reason)
            await asyncio.sleep(2.0 * (attempt + 1))
        raise AssertionError("unreachable")

    @staticmethod
    def _to_generation(resp: Any, seconds: float) -> Generation:
        tokens, lps, tops = [], [], []
        for lp in resp.logprobs or []:
            tokens.append(lp.token)
            lps.append(lp.logprob)
            tops.append([[t.token, t.logprob] for t in (lp.top_logprobs or [])])
        return Generation(
            text=resp.response or "",
            tokens=tokens,
            logprobs=lps,
            top_logprobs=tops if any(tops) else [],
            done_reason=resp.done_reason or "",
            prompt_tokens=resp.prompt_eval_count or 0,
            completion_tokens=resp.eval_count or 0,
            seconds=seconds,
        )

    async def version(self) -> str:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{self._base_url}/api/version")
            r.raise_for_status()
            return r.json().get("version", "?")

    async def available(self) -> list[str]:
        resp = await self._client.list()
        return [normalize_tag(m.model) for m in resp.models]

    async def loaded(self) -> list[str]:
        resp = await self._client.ps()
        return [normalize_tag(m.model) for m in resp.models]

    async def pull(self, model: str) -> None:
        from ollama import AsyncClient

        logger.info("Pulling {} (this can take a while)", model)
        # Downloads of 8-9 GB outlast the generation timeout
        await AsyncClient(host=self._base_url, timeout=None).pull(model)

    async def unload(self, model: str) -> None:
        await self._client.generate(model=model, prompt="", keep_alive=0)


def make_backend(name: str, items: list | None = None) -> Backend:
    if name == "ollama":
        return OllamaBackend()
    if name == "fake":
        from uncertainty.generate.fake import FakeBackend

        return FakeBackend(items or [])
    raise ValueError(f"unknown backend {name!r}")
