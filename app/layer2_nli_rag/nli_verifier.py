"""
Layer 2: NLI Verifier using DeBERTa-v3.

Loads a HuggingFace NLI model (cross-encoder/nli-deberta-v3-base) to
verify Atomic Propositions against retrieved evidence. Each AP is
classified as Entailment, Contradiction, or Neutral.

Implements the three-state verification from the thesis:
- Entailment: Claim approved for generation compilation
- Contradiction: AP rejected with negative constraints injected
- Neutral: Evidence inconclusive, triggers expanded search
"""

from __future__ import annotations

import asyncio
from typing import Any

import torch
from loguru import logger

from app.config import settings
from app.models.schemas import (
    AtomicProposition,
    Layer2Result,
    NLILabel,
    NLIResult,
    SearchResult,
)


class NLIVerifier:
    """
    Natural Language Inference verifier using DeBERTa-v3.

    Classifies each (premise, hypothesis) pair into:
    Entailment / Contradiction / Neutral
    """

    # Label mapping for cross-encoder/nli-deberta-v3-base
    # This model maps: 0 = contradiction, 1 = entailment, 2 = neutral
    _LABEL_MAP = {
        0: NLILabel.CONTRADICTION,
        1: NLILabel.ENTAILMENT,
        2: NLILabel.NEUTRAL,
    }

    _LABEL_NAMES = ["contradiction", "entailment", "neutral"]

    def __init__(
        self,
        model_name: str | None = None,
        confidence_threshold: float | None = None,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name or settings.nli_model_name
        self._confidence_threshold = (
            confidence_threshold or settings.nli_confidence_threshold
        )
        self._device = device
        self._model = None
        self._tokenizer = None
        logger.info(
            "NLIVerifier initialized | model={} | threshold={}",
            self._model_name,
            self._confidence_threshold,
        )

    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------

    async def load_model(self) -> None:
        """Load the NLI model and tokenizer (runs in thread pool)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self) -> None:
        """Synchronous model loading."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading NLI model: {}", self._model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name
        )
        self._model.to(self._device)
        self._model.eval()

        # Verify label mapping from model config
        if hasattr(self._model.config, "id2label"):
            logger.info("Model label mapping: {}", self._model.config.id2label)

        logger.info("NLI model loaded successfully")

    # -------------------------------------------------------------------------
    # Single Claim Verification
    # -------------------------------------------------------------------------

    async def verify_claim(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[NLILabel, float, dict[str, float]]:
        """
        Verify a single claim against evidence.

        Args:
            premise: The source evidence text.
            hypothesis: The claim to verify.

        Returns:
            Tuple of (label, confidence, all_scores).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._verify_sync, premise, hypothesis
        )

    def _verify_sync(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[NLILabel, float, dict[str, float]]:
        """Synchronous NLI inference."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("NLI model not loaded. Call load_model() first.")

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits

        # Softmax probabilities
        probs = torch.softmax(logits, dim=1).squeeze()

        # Get predicted class
        predicted_idx = torch.argmax(probs).item()
        confidence = probs[predicted_idx].item()

        # Build scores dict
        all_scores = {
            name: probs[i].item()
            for i, name in enumerate(self._LABEL_NAMES)
        }

        # Apply confidence threshold for entailment
        label = self._LABEL_MAP.get(predicted_idx, NLILabel.NEUTRAL)
        if label == NLILabel.ENTAILMENT and confidence < self._confidence_threshold:
            # Downgrade to neutral if confidence is below threshold
            label = NLILabel.NEUTRAL

        return label, confidence, all_scores

    # -------------------------------------------------------------------------
    # Batch Verification
    # -------------------------------------------------------------------------

    async def verify_propositions(
        self,
        propositions: list[AtomicProposition],
        evidence_map: dict[int, list[SearchResult]],
    ) -> list[NLIResult]:
        """
        Verify multiple Atomic Propositions against their evidence.

        For each AP, it selects the best evidence and runs NLI.
        If multiple evidence pieces exist, the most favorable result wins.

        Args:
            propositions: List of APs to verify.
            evidence_map: Dict mapping prop ID → search results.

        Returns:
            List of NLIResult objects.
        """
        tasks = []
        for prop in propositions:
            evidence = evidence_map.get(prop.id, [])
            tasks.append(self._verify_proposition_against_evidence(prop, evidence))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        nli_results: list[NLIResult] = []
        for prop, result in zip(propositions, results):
            if isinstance(result, Exception):
                logger.warning("NLI verification failed for AP {}: {}", prop.id, result)
                # Default to neutral on error
                nli_results.append(
                    NLIResult(
                        proposition=prop,
                        label=NLILabel.NEUTRAL,
                        confidence=0.0,
                        premise_used="[Error during verification]",
                        all_scores={},
                    )
                )
            else:
                nli_results.append(result)

        return nli_results

    async def _verify_proposition_against_evidence(
        self,
        proposition: AtomicProposition,
        evidence: list[SearchResult],
    ) -> NLIResult:
        """Verify a single AP against all available evidence, pick best result."""
        if not evidence:
            return NLIResult(
                proposition=proposition,
                label=NLILabel.NEUTRAL,
                confidence=0.0,
                premise_used="[No evidence found]",
                all_scores={},
            )

        best_result: NLIResult | None = None
        best_entailment_score = -1.0

        for source in evidence:
            label, confidence, all_scores = await self.verify_claim(
                premise=source.snippet,
                hypothesis=proposition.text,
            )

            result = NLIResult(
                proposition=proposition,
                label=label,
                confidence=confidence,
                premise_used=source.snippet,
                all_scores=all_scores,
            )

            # If we find an entailment, that's the best outcome
            if label == NLILabel.ENTAILMENT:
                return result

            # Track the result with highest entailment score as fallback
            entailment_score = all_scores.get("entailment", 0.0)
            if entailment_score > best_entailment_score:
                best_entailment_score = entailment_score
                best_result = result

            # If contradiction found, return immediately
            if label == NLILabel.CONTRADICTION:
                return result

        return best_result or NLIResult(
            proposition=proposition,
            label=NLILabel.NEUTRAL,
            confidence=0.0,
            premise_used=evidence[0].snippet if evidence else "",
            all_scores={},
        )

    # -------------------------------------------------------------------------
    # Result Aggregation
    # -------------------------------------------------------------------------

    @staticmethod
    def aggregate_results(
        propositions: list[AtomicProposition],
        nli_results: list[NLIResult],
        draft_response: str,
    ) -> Layer2Result:
        """
        Aggregate NLI results and produce a refined response.

        Contradicted propositions are removed from the response.
        Neutral propositions are flagged but kept with caveats.
        """
        import time

        entailed = [r for r in nli_results if r.label == NLILabel.ENTAILMENT]
        contradicted = [r for r in nli_results if r.label == NLILabel.CONTRADICTION]
        neutral = [r for r in nli_results if r.label == NLILabel.NEUTRAL]

        # Build refined response by removing contradicted claims
        refined = draft_response
        for r in contradicted:
            # Remove contradicted proposition text from the response
            refined = refined.replace(r.proposition.source_span, "")
            # If we can't remove by source_span, remove by proposition text
            if r.proposition.text in refined:
                refined = refined.replace(r.proposition.text, "[REMOVED: contradicted by evidence]")

        # Clean up empty lines
        lines = [line for line in refined.split('\n') if line.strip()]
        refined = '\n'.join(lines)

        if not refined.strip():
            refined = draft_response  # Fallback to original if everything was removed

        return Layer2Result(
            propositions=propositions,
            nli_results=nli_results,
            entailed_count=len(entailed),
            contradicted_count=len(contradicted),
            neutral_count=len(neutral),
            refined_response=refined,
            search_results_used=sum(
                1 for r in nli_results if r.premise_used != "[No evidence found]"
            ),
            latency_ms=0.0,  # Set by caller
        )

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def unload(self) -> None:
        """Unload the NLI model to free memory."""
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("NLI model unloaded")
