"""
Async Ollama client wrapper for the Defense-in-Depth Framework.

Provides an asynchronous interface to the local Ollama server for:
- Text generation with logprob extraction
- Multi-turn chat conversations
- Model availability validation
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from loguru import logger
from ollama import AsyncClient, ResponseError

from app.config import settings


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
            "OllamaClient initialized | base_url={} | model={}",
            self._base_url,
            self._default_model,
        )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def validate_models(self) -> list[str]:
        """Fetch and cache the list of available Ollama models."""
        try:
            response = await self._client.list()
            self._available_models = [m.model for m in response.models]
            logger.info("Available Ollama models: {}", self._available_models)
            return self._available_models
        except Exception as exc:
            logger.error("Failed to list Ollama models: {}", exc)
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._base_url}: {exc}"
            ) from exc

    async def warmup(self, model: str | None = None) -> None:
        """
        Warm up a model by sending a minimal prompt.

        This ensures the model is loaded into memory before the first real query.
        """
        target = model or self._default_model
        logger.info("Warming up Ollama model: {}", target)
        try:
            await self._client.generate(
                model=target,
                prompt="Hello",
                options={"num_predict": 1},
            )
            logger.info("Model {} warmed up successfully", target)
        except ResponseError as exc:
            logger.warning("Warmup failed for {}: {}", target, exc)

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

        Returns:
            GenerationResult with text, token counts, and timing info.
        """
        target = model or self._default_model

        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }

        try:
            if stream:
                return await self._generate_streaming(prompt, target, system, options)

            response = await self._client.generate(
                model=target,
                prompt=prompt,
                system=system,
                options=options,
                stream=False,
            )

            return GenerationResult(
                text=response.response,
                model=response.model,
                prompt_tokens=response.prompt_eval_count or 0,
                completion_tokens=response.eval_count or 0,
                total_duration_ns=response.total_duration or 0,
                eval_duration_ns=response.eval_duration or 0,
            )

        except ResponseError as exc:
            logger.error("Ollama generation failed: {}", exc)
            raise
        except Exception as exc:
            logger.error("Unexpected error during generation: {}", exc)
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

        async for chunk in await self._client.generate(
            model=model,
            prompt=prompt,
            system=system,
            options=options,
            stream=True,
        ):
            chunks.append(chunk.response)
            if chunk.done:
                final_response = chunk

        full_text = "".join(chunks)

        return GenerationResult(
            text=full_text,
            model=model,
            prompt_tokens=final_response.prompt_eval_count if final_response else 0,
            completion_tokens=final_response.eval_count if final_response else 0,
            total_duration_ns=final_response.total_duration if final_response else 0,
            eval_duration_ns=final_response.eval_duration if final_response else 0,
        )

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
            response = await self._client.chat(
                model=target,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                stream=False,
            )

            return GenerationResult(
                text=response.message.content,
                model=response.model,
                prompt_tokens=response.prompt_eval_count or 0,
                completion_tokens=response.eval_count or 0,
                total_duration_ns=response.total_duration or 0,
                eval_duration_ns=response.eval_duration or 0,
            )

        except ResponseError as exc:
            logger.error("Ollama chat failed: {}", exc)
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

        Uses Ollama's format parameter to request JSON output.
        """
        target = model or self._default_model

        response = await self._client.generate(
            model=target,
            prompt=prompt,
            system=system,
            format="json",
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            stream=False,
        )

        return GenerationResult(
            text=response.response,
            model=response.model,
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
            total_duration_ns=response.total_duration or 0,
            eval_duration_ns=response.eval_duration or 0,
        )


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
    )

    def __init__(
        self,
        text: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_duration_ns: int = 0,
        eval_duration_ns: int = 0,
    ) -> None:
        self.text = text
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_duration_ns = total_duration_ns
        self.eval_duration_ns = eval_duration_ns

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
