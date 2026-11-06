"""
Async Ollama client wrapper for the Defense-in-Depth Framework.

Provides an asynchronous interface to the local Ollama server for:
- Text generation with logprob extraction
- Multi-turn chat conversations
- Model availability validation
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any

import httpx
from loguru import logger
from ollama import AsyncClient, ResponseError

from app.config import settings


class ModelUnavailableError(RuntimeError):
    """The requested model is not pulled on the Ollama server."""


def _normalize_model(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


# Per-request token accumulator. The router opens a meter at the start of a
# query so that every Ollama call made on its behalf (draft, semantic entropy
# samples, Layer 2 / Layer 3 calls) is counted in the reported token cost.
_token_meter: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar(
    "token_meter", default=None
)


def start_token_meter() -> list[int]:
    """Start a new token meter for the current async context: [prompt, completion]."""
    meter = [0, 0]
    _token_meter.set(meter)
    return meter


def _record_tokens(result: "GenerationResult") -> "GenerationResult":
    meter = _token_meter.get()
    if meter is not None:
        meter[0] += result.prompt_tokens
        meter[1] += result.completion_tokens
    return result


# One gate per (event loop, server): Ollama serves OLLAMA_NUM_PARALLEL requests at a
# time and queues the rest server-side, where queued requests burn their HTTP timeout.
# Waiting here instead means the timeout only measures actual generation.
_gates: dict[tuple[int, str], asyncio.Semaphore] = {}


def _gate(base_url: str) -> asyncio.Semaphore:
    key = (id(asyncio.get_running_loop()), base_url)
    if key not in _gates:
        _gates[key] = asyncio.Semaphore(settings.ollama_max_concurrency)
    return _gates[key]


# Transient failures worth retrying: runner crashed / out of VRAM while loading
# (HTTP 5xx), or the connection dropped. Timeouts are NOT retried — they already
# cost ollama_timeout seconds and a retry would double that.
_TRANSIENT_NETWORK = (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)


class OllamaClient:
    """
    High-level async wrapper around the Ollama Python SDK.

    Handles connection lifecycle, retry logic, and provides
    structured interfaces for generation and chat.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._base_url = base_url or settings.ollama_base_url
        self._default_model = model or settings.ollama_model
        self._timeout = timeout or settings.ollama_timeout
        self._client = AsyncClient(host=self._base_url, timeout=self._timeout)
        self._available_models: list[str] = []
        logger.info(
            "OllamaClient initialized | base_url={} | model={} | num_ctx={} | concurrency={}",
            self._base_url,
            self._default_model,
            settings.ollama_num_ctx,
            settings.ollama_max_concurrency,
        )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    @property
    def default_model(self) -> str:
        return self._default_model

    def use_model(self, model: str) -> None:
        """Re-point this client (and every component sharing it) at another model."""
        logger.warning("OllamaClient default model {} → {}", self._default_model, model)
        self._default_model = model

    async def validate_models(self) -> list[str]:
        """
        Fetch the server's model list and check that this client's model is pulled.

        Raises:
            ConnectionError: Ollama is unreachable.
            ModelUnavailableError: The default model is not pulled.
        """
        try:
            response = await self._client.list()
        except Exception as exc:
            logger.error("Failed to list Ollama models: {}", exc)
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._base_url}: {exc}"
            ) from exc
        self._available_models = [m.model for m in response.models]
        logger.info("Available Ollama models: {}", self._available_models)
        self.ensure_available(self._default_model)
        return self._available_models

    def ensure_available(self, model: str) -> None:
        available = {_normalize_model(m) for m in self._available_models}
        if _normalize_model(model) not in available:
            raise ModelUnavailableError(
                f"Model {model!r} is not pulled on {self._base_url} "
                f"(available: {sorted(available)}). Run: ollama pull {model}"
            )

    async def warmup(self, model: str | None = None) -> None:
        """
        Load a model into memory with a 1-token generation.

        Raises on failure: a model that cannot generate here will fail every
        real query too, and should be caught at startup.
        """
        target = model or self._default_model
        logger.info("Warming up Ollama model: {}", target)
        await self._call(
            "generate", model=target, prompt="Hello", options={"num_predict": 1}
        )
        logger.info("Model {} warmed up successfully", target)

    # -------------------------------------------------------------------------
    # Transport: concurrency gate + retries
    # -------------------------------------------------------------------------

    async def _call(self, method: str, **kwargs: Any) -> Any:
        kwargs.setdefault("keep_alive", settings.ollama_keep_alive)
        if settings.ollama_think is not None:
            # Thinking models (qwen3.5) otherwise reason for thousands of tokens per sample
            kwargs.setdefault("think", settings.ollama_think)
        model = kwargs.get("model")
        for attempt in range(settings.ollama_max_retries + 1):
            try:
                async with _gate(self._base_url):
                    return await getattr(self._client, method)(**kwargs)
            except ResponseError as exc:
                if exc.status_code == 404:
                    raise ModelUnavailableError(f"{model}: {exc.error}") from exc
                if exc.status_code < 500 or attempt == settings.ollama_max_retries:
                    raise
                reason = f"HTTP {exc.status_code}: {exc.error}"
            except _TRANSIENT_NETWORK as exc:
                if attempt == settings.ollama_max_retries:
                    raise
                reason = f"{type(exc).__name__}: {exc}"
            backoff = 2.0 * (attempt + 1)
            logger.warning(
                "Ollama {} on {} failed (attempt {}/{}), retrying in {:.0f}s | {}",
                method, model, attempt + 1, settings.ollama_max_retries + 1, backoff, reason,
            )
            await asyncio.sleep(backoff)
        raise AssertionError("unreachable")

    @staticmethod
    def _options(temperature: float, max_tokens: int, seed: int | None = None) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": settings.ollama_num_ctx,
        }
        if seed is not None:
            options["seed"] = seed
        return options

    @staticmethod
    def _to_result(response: Any, text: str, model: str, max_tokens: int) -> "GenerationResult":
        result = GenerationResult(
            text=text,
            model=getattr(response, "model", None) or model,
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
            total_duration_ns=response.total_duration or 0,
            eval_duration_ns=response.eval_duration or 0,
            done_reason=getattr(response, "done_reason", None) or "",
        )
        if result.truncated:
            logger.warning(
                "Ollama output hit num_predict={} on {} — response is cut off",
                max_tokens, result.model,
            )
        if result.prompt_tokens + result.completion_tokens >= settings.ollama_num_ctx - 1:
            logger.warning(
                "Ollama context window full on {} (prompt={} + completion={} >= num_ctx={}) "
                "— the prompt start was likely dropped",
                result.model, result.prompt_tokens, result.completion_tokens,
                settings.ollama_num_ctx,
            )
        return _record_tokens(result)

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stream: bool = False,
        seed: int | None = None,
    ) -> GenerationResult:
        """
        Generate a completion from the Ollama model.

        Args:
            prompt: The user prompt.
            model: Model name override.
            system: Optional system prompt.
            temperature: Sampling temperature (0.0 = greedy).
            max_tokens: Maximum tokens to generate.
            stream: Whether to stream the response.
            seed: Optional sampling seed for reproducible stochastic samples.

        Returns:
            GenerationResult with text, token counts, and timing info.
        """
        target = model or self._default_model
        options = self._options(temperature, max_tokens, seed)

        try:
            if stream:
                return await self._generate_streaming(prompt, target, system, options)

            response = await self._call(
                "generate",
                model=target,
                prompt=prompt,
                system=system,
                options=options,
                stream=False,
            )
            return self._to_result(response, response.response, target, max_tokens)

        except Exception as exc:
            logger.error("Ollama generation failed on {}: {}: {}", target, type(exc).__name__, exc)
            raise

    async def _generate_streaming(
        self,
        prompt: str,
        model: str,
        system: str,
        options: dict[str, Any],
    ) -> GenerationResult:
        """Collect a streaming response into a single GenerationResult."""
        chunks: list[str] = []
        final_response = None

        async with _gate(self._base_url):
            async for chunk in await self._client.generate(
                model=model,
                prompt=prompt,
                system=system,
                options=options,
                stream=True,
                keep_alive=settings.ollama_keep_alive,
                **({} if settings.ollama_think is None else {"think": settings.ollama_think}),
            ):
                chunks.append(chunk.response)
                if chunk.done:
                    final_response = chunk

        if final_response is None:
            raise RuntimeError(f"Ollama stream from {model} ended without a done chunk")
        return self._to_result(final_response, "".join(chunks), model, options["num_predict"])

    # -------------------------------------------------------------------------
    # Chat
    # -------------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        """
        Multi-turn chat conversation with the Ollama model.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
            model: Model name override.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            GenerationResult from the assistant turn.
        """
        target = model or self._default_model

        try:
            response = await self._call(
                "chat",
                model=target,
                messages=messages,
                options=self._options(temperature, max_tokens),
                stream=False,
            )
            return self._to_result(response, response.message.content, target, max_tokens)

        except Exception as exc:
            logger.error("Ollama chat failed on {}: {}: {}", target, type(exc).__name__, exc)
            raise

    # -------------------------------------------------------------------------
    # Structured Output Helper
    # -------------------------------------------------------------------------

    async def generate_json(
        self,
        prompt: str,
        model: str | None = None,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        """
        Generate a response with JSON format enforcement.

        Uses Ollama's format parameter to request JSON output. Note that
        format="json" constrains the output to a JSON *object*: prompts must
        ask for an object (e.g. {"items": [...]}), never a bare array.
        """
        target = model or self._default_model

        response = await self._call(
            "generate",
            model=target,
            prompt=prompt,
            system=system,
            format="json",
            options=self._options(temperature, max_tokens),
            stream=False,
        )
        return self._to_result(response, response.response, target, max_tokens)


# =============================================================================
# Result Data Class
# =============================================================================


class GenerationResult:
    """Structured result from an Ollama generation or chat call."""

    __slots__ = (
        "text",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_duration_ns",
        "eval_duration_ns",
        "done_reason",
    )

    def __init__(
        self,
        text: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_duration_ns: int = 0,
        eval_duration_ns: int = 0,
        done_reason: str = "",
    ) -> None:
        self.text = text
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_duration_ns = total_duration_ns
        self.eval_duration_ns = eval_duration_ns
        self.done_reason = done_reason

    @property
    def truncated(self) -> bool:
        """True when generation stopped at num_predict rather than naturally."""
        return self.done_reason == "length"

    @property
    def total_tokens(self) -> int:
        """Total token count (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def latency_ms(self) -> float:
        """Total duration in milliseconds."""
        return self.total_duration_ns / 1_000_000

    def __repr__(self) -> str:
        return (
            f"GenerationResult(model={self.model!r}, "
            f"tokens={self.total_tokens}, "
            f"latency={self.latency_ms:.1f}ms)"
        )
