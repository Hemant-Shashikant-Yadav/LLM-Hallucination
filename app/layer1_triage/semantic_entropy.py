"""
Layer 1: Sampling-based Semantic Entropy Triage.

Replaces the untrained surrogate probe with the semantic entropy estimator
of Kuhn et al. (2023) and Farquhar et al. (2024, Nature):

1. Draw K stochastic samples from the generator in addition to the greedy draft.
2. Cluster the K+1 responses by meaning:
     • quantitative queries  → cluster by the extracted final numeric answer
     • open-domain queries   → cluster by bidirectional NLI entailment (DeBERTa-v3)
3. Compute discrete semantic entropy over the cluster distribution,
       H_sem = −Σ_c p(c) log p(c) / log(K+1)   ∈ [0, 1]
4. H_sem < T_safe (per task type) → SAFE (fast path); otherwise route by task type:
     quantitative → REASONING_UNCERTAIN (Layer 3), open-domain → KNOWLEDGE_UNCERTAIN (Layer 2)

Also exposes whether the greedy draft agrees with the majority cluster,
which is a strong hallucination signal on its own.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import Counter
from dataclasses import dataclass, field

from loguru import logger

from app.answer_utils import (
    canonicalize,
    extract_choice,
    extract_final_answer,
    is_advanced_math,
    is_numeric_reasoning_query,
    leading_sentences,
    parse_options,
)
from app.config import settings
from app.layer2_nli_rag.nli_verifier import NLIVerifier
from app.llm_clients.ollama_client import OllamaClient
from app.models.schemas import ErrorClassification, TriageDecision, TriageResult
from app.telemetry import record_degradation


def math_sample_tokens(prompt: str) -> int:
    """Token budget for one maths sample: a cut-off solution is dropped and cannot vote."""
    return settings.se_max_tokens_advanced if is_advanced_math(prompt) else settings.se_max_tokens_math


@dataclass
class SemanticEntropyEstimate:
    """Raw output of the semantic entropy estimator."""

    entropy: float
    cluster_ids: list[int]
    responses: list[str]
    is_numeric: bool
    numeric_answers: list[str | None] = field(default_factory=list)

    @property
    def num_clusters(self) -> int:
        return len(set(self.cluster_ids))

    @property
    def draft_in_majority(self) -> bool:
        counts = Counter(self.cluster_ids)
        top = max(counts.values())
        return counts[self.cluster_ids[0]] == top


class SemanticEntropyTriage:
    """Layer 1 triage via sampled semantic entropy."""

    def __init__(
        self,
        generator: OllamaClient,
        nli_verifier: NLIVerifier,
        num_samples: int | None = None,
        temperature: float | None = None,
        t_safe: float | None = None,
    ) -> None:
        self._generator = generator
        self._nli = nli_verifier
        self._num_samples = num_samples or settings.se_num_samples
        self._temperature = (
            temperature if temperature is not None else settings.se_sample_temperature
        )
        self._t_safe = t_safe if t_safe is not None else settings.t_safe
        logger.info(
            "SemanticEntropyTriage initialized | K={} | T={} | T_safe={}",
            self._num_samples,
            self._temperature,
            self._t_safe,
        )

    async def triage(
        self,
        prompt: str,
        draft_response: str,
        model: str | None = None,
        task_type: str | None = None,
    ) -> tuple[TriageResult, SemanticEntropyEstimate]:
        start = time.perf_counter()

        is_numeric = (
            task_type == "math" if task_type is not None else is_numeric_reasoning_query(prompt)
        )
        advanced = is_numeric and is_advanced_math(prompt)
        options = parse_options(prompt)
        # Worked answers (maths, or reasoning through each option) need room to finish
        samples = await self._sample(
            prompt, model, max_tokens=math_sample_tokens(prompt) if is_numeric or options else None,
            drop_truncated=is_numeric or bool(options),
        )
        responses = [draft_response, *samples]

        if options:
            # Multiple choice: the meaning of an answer is the option it picks
            cluster_ids, numeric_answers = self._cluster_keys(
                [extract_choice(r, options) for r in responses]
            )
        elif is_numeric:
            cluster_ids, numeric_answers = self._cluster_numeric(responses)
        else:
            cluster_ids = await self._cluster_semantic(prompt, responses)
            numeric_answers = []

        entropy = self._discrete_entropy(cluster_ids)
        estimate = SemanticEntropyEstimate(
            entropy=entropy,
            cluster_ids=cluster_ids,
            responses=responses,
            is_numeric=is_numeric,
            numeric_answers=numeric_answers,
        )

        t_safe = settings.t_safe_reasoning if is_numeric else self._t_safe
        # With too few samples H_sem is not evidence of certainty: (draft + 0 samples)
        # gives H_sem = 0 and would send every query down the fast path.
        enough_samples = len(samples) >= min(settings.se_min_samples, self._num_samples)
        if not enough_samples:
            record_degradation(
                "layer1.semantic_entropy",
                f"only {len(samples)}/{self._num_samples} samples succeeded; "
                "H_sem unreliable, escalating instead of fast path",
            )
        if task_type == "time_sensitive":
            # A confident answer about a changing fact can simply be out of date:
            # agreement between samples says nothing about freshness.
            decision = TriageDecision.KNOWLEDGE_UNCERTAIN
        elif advanced and settings.verify_advanced_math:
            # On hard problems all samples can share one mistake: always verify
            decision = TriageDecision.REASONING_UNCERTAIN
        elif enough_samples and entropy < t_safe and estimate.draft_in_majority:
            decision = TriageDecision.SAFE
        elif is_numeric:
            decision = TriageDecision.REASONING_UNCERTAIN
        else:
            decision = TriageDecision.KNOWLEDGE_UNCERTAIN

        error_type = "reasoning" if is_numeric else "knowledge"
        result = TriageResult(
            decision=decision,
            semantic_entropy=entropy,
            hallucination_probability=entropy,
            error_classification=ErrorClassification(
                semantic_entropy=entropy,
                error_type=error_type,
                knowledge_logit=0.0 if is_numeric else 1.0,
                reasoning_logit=1.0 if is_numeric else 0.0,
                routing_confidence=1.0,
            ),
            threshold_used=t_safe,
            draft_response=draft_response,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

        logger.info(
            "SE triage | decision={} | H_sem={:.3f} | samples_ok={}/{} | clusters={}/{} | "
            "draft_in_majority={} | type={} | {:.0f}ms",
            decision.value,
            entropy,
            len(samples),
            self._num_samples,
            estimate.num_clusters,
            len(responses),
            estimate.draft_in_majority,
            error_type,
            result.latency_ms,
        )
        return result, estimate

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    async def sample(self, prompt: str, model: str | None, n: int, seed_base: int,
                     max_tokens: int | None = None) -> list[str]:
        """Draw n more stochastic samples (used by Layer 3's self-consistency vote)."""
        return await self._sample(prompt, model, n=n, seed_base=seed_base, max_tokens=max_tokens,
                                  drop_truncated=True)

    async def _sample(
        self, prompt: str, model: str | None, n: int | None = None, seed_base: int = 1000,
        max_tokens: int | None = None, drop_truncated: bool = False,
    ) -> list[str]:
        async def _one(i: int) -> str:
            try:
                res = await self._generator.generate(
                    prompt=prompt,
                    model=model,
                    temperature=self._temperature,
                    max_tokens=max_tokens or settings.se_max_tokens,
                    seed=seed_base + i,
                )
                if drop_truncated and getattr(res, "truncated", False):
                    # A maths sample cut off mid-calculation has no final answer, only
                    # whatever digit came last: it must not vote (seen live on m12)
                    logger.info("Sample {} hit the token limit; excluded from the vote", i)
                    return ""
                return res.text
            except Exception as exc:
                record_degradation("layer1.sample", f"sample {i} failed: {type(exc).__name__}: {exc}")
                return ""

        samples = await asyncio.gather(*(_one(i) for i in range(n or self._num_samples)))
        return [s for s in samples if s.strip()]

    # -------------------------------------------------------------------------
    # Clustering
    # -------------------------------------------------------------------------

    @staticmethod
    def _cluster_numeric(responses: list[str]) -> tuple[list[int], list[str | None]]:
        # Canonical answers: fractions, symbolic answers and rounding differences
        # (142857.14 vs 142857.142857) land in the same cluster
        return SemanticEntropyTriage._cluster_keys(
            canonicalize([extract_final_answer(r) for r in responses])
        )

    @staticmethod
    def _cluster_keys(answers: list[str | None]) -> tuple[list[int], list[str | None]]:
        mapping: dict[str, int] = {}
        cluster_ids: list[int] = []
        for ans in answers:
            # Unparseable responses each form their own singleton cluster
            key = ans if ans is not None else f"__none_{len(cluster_ids)}"
            if key not in mapping:
                mapping[key] = len(mapping)
            cluster_ids.append(mapping[key])
        return cluster_ids, answers

    async def _cluster_semantic(self, prompt: str, responses: list[str]) -> list[int]:
        """Greedy bidirectional-entailment clustering (Kuhn et al., 2023)."""
        if not self._nli.is_loaded:
            # Without NLI every response is its own meaning class (maximally uncertain)
            record_degradation(
                "layer1.clustering", "NLI model not loaded; H_sem forced to 1.0 (always escalates)"
            )
            return list(range(len(responses)))

        texts = [f"{prompt} {leading_sentences(r)}" for r in responses]
        cluster_ids: list[int] = [0]
        representatives: list[int] = [0]

        for i in range(1, len(texts)):
            pairs: list[tuple[str, str]] = []
            for rep in representatives:
                pairs.append((texts[rep], texts[i]))
                pairs.append((texts[i], texts[rep]))
            scores = await self._nli.score_pairs(pairs)

            assigned = None
            for k, rep in enumerate(representatives):
                fwd, bwd = scores[2 * k], scores[2 * k + 1]
                if self._entails(fwd) and self._entails(bwd):
                    assigned = cluster_ids[rep]
                    break
            if assigned is None:
                assigned = len(representatives)
                representatives.append(i)
            cluster_ids.append(assigned)

        return cluster_ids

    @staticmethod
    def _entails(score: dict[str, float]) -> bool:
        return score["entailment"] >= max(score["neutral"], score["contradiction"])

    @staticmethod
    def _discrete_entropy(cluster_ids: list[int]) -> float:
        n = len(cluster_ids)
        if n <= 1:
            return 0.0
        counts = Counter(cluster_ids)
        h = -sum((c / n) * math.log(c / n) for c in counts.values())
        return min(max(h / math.log(n), 0.0), 1.0)
