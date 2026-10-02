"""
Layer 2: Conflict-Aware NLI Verifier using DeBERTa-v3.

Algorithmic Novelty (for M.Tech thesis):

**Conflict-Aware Weighted NLI Consensus Scoring**

Instead of simple "first entailment wins" or "best entailment score" logic,
this module implements a weighted consensus score across ALL retrieved passages:

    S(p_i) = Σ_j  w_j · [P(entail)_j − P(contradict)_j]

where:
    p_i         = the i-th atomic proposition (claim to verify)
    j           = index over retrieved evidence passages
    w_j         = retrieval relevance weight for passage j (from the retriever)
    P(entail)_j = NLI entailment probability for (passage_j, p_i)
    P(contra)_j = NLI contradiction probability for (passage_j, p_i)

The consensus score is compared against two configurable thresholds:
    • accept_threshold (default 0.3):  S ≥ accept_threshold → ACCEPT
    • reject_threshold (default −0.3): S ≤ reject_threshold → REJECT
    •                                  otherwise → NEUTRAL

This captures conflicting evidence signals rather than short-circuiting
on the first entailment or contradiction, producing a more nuanced
and publication-worthy verification result.
"""

from __future__ import annotations

import asyncio
from typing import Any

import torch
from loguru import logger

from app.config import settings
from app.models.schemas import (
    AtomicProposition,
    ConflictAwareNLIResult,
    ConsensusVerdict,
    Layer2Result,
    NLILabel,
    NLIPassageScore,
    NLIResult,
    SearchResult,
)


class NLIVerifier:
    """
    Natural Language Inference verifier using DeBERTa-v3 with
    conflict-aware weighted consensus scoring.

    Classifies each (premise, hypothesis) pair into:
    Entailment / Contradiction / Neutral, then aggregates
    across passages using retrieval-weighted consensus.
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
        accept_threshold: float = 0.3,
        reject_threshold: float = -0.3,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_name: HuggingFace NLI model identifier.
            confidence_threshold: Min softmax confidence for simple entailment.
            accept_threshold: Consensus score S ≥ this → ACCEPT.
            reject_threshold: Consensus score S ≤ this → REJECT.
            device: Torch device.
        """
        self._model_name = model_name or settings.nli_model_name
        self._confidence_threshold = (
            confidence_threshold or settings.nli_confidence_threshold
        )
        self._accept_threshold = accept_threshold
        self._reject_threshold = reject_threshold
        self._device = device
        self._model = None
        self._tokenizer = None
        logger.info(
            "NLIVerifier initialized | model={} | threshold={} "
            "| accept_T={} | reject_T={}",
            self._model_name,
            self._confidence_threshold,
            self._accept_threshold,
            self._reject_threshold,
        )

    # -------------------------------------------------------------------------
    # Model Loading (cold-start safe)
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
    # Single-Pair NLI Inference
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
    # Conflict-Aware Weighted Consensus
    # -------------------------------------------------------------------------

    def _compute_consensus(
        self,
        passage_scores: list[NLIPassageScore],
    ) -> ConflictAwareNLIResult:
        """
        Compute the weighted consensus score across all passages.

        S(p_i) = Σ_j  w_j · [P(entail)_j − P(contradict)_j]

        Args:
            passage_scores: Per-passage NLI breakdown with relevance weights.

        Returns:
            ConflictAwareNLIResult with the aggregate score and verdict.
        """
        if not passage_scores:
            return ConflictAwareNLIResult(
                consensus_score=0.0,
                verdict=ConsensusVerdict.NEUTRAL,
                passage_scores=[],
                accept_threshold=self._accept_threshold,
                reject_threshold=self._reject_threshold,
            )

        consensus = sum(ps.weighted_contribution for ps in passage_scores)

        # Normalize by sum of weights to keep S in [-1, 1]
        total_weight = sum(ps.relevance_weight for ps in passage_scores)
        if total_weight > 0:
            consensus /= total_weight

        # Apply thresholds
        if consensus >= self._accept_threshold:
            verdict = ConsensusVerdict.ACCEPT
        elif consensus <= self._reject_threshold:
            verdict = ConsensusVerdict.REJECT
        else:
            verdict = ConsensusVerdict.NEUTRAL

        return ConflictAwareNLIResult(
            consensus_score=round(consensus, 6),
            verdict=verdict,
            passage_scores=passage_scores,
            accept_threshold=self._accept_threshold,
            reject_threshold=self._reject_threshold,
        )

    @staticmethod
    def _consensus_verdict_to_nli_label(verdict: ConsensusVerdict) -> NLILabel:
        """Map a ConsensusVerdict to the standard NLILabel enum."""
        if verdict == ConsensusVerdict.ACCEPT:
            return NLILabel.ENTAILMENT
        elif verdict == ConsensusVerdict.REJECT:
            return NLILabel.CONTRADICTION
        return NLILabel.NEUTRAL

    # -------------------------------------------------------------------------
    # Batch Verification with Consensus
    # -------------------------------------------------------------------------

    async def verify_propositions(
        self,
        propositions: list[AtomicProposition],
        evidence_map: dict[int, list[SearchResult]],
    ) -> list[NLIResult]:
        """
        Verify multiple Atomic Propositions against their evidence
        using conflict-aware weighted consensus.

        For each AP, ALL retrieved passages are scored. The weighted
        consensus formula is applied to produce a holistic verdict
        rather than short-circuiting on the first entailment.

        Args:
            propositions: List of APs to verify.
            evidence_map: Dict mapping prop ID → search results.

        Returns:
            List of NLIResult objects with conflict_aware_result populated.
        """
        tasks = []
        for prop in propositions:
            evidence = evidence_map.get(prop.id, [])
            tasks.append(
                self._verify_proposition_consensus(prop, evidence)
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        nli_results: list[NLIResult] = []
        for prop, result in zip(propositions, results):
            if isinstance(result, Exception):
                logger.warning(
                    "NLI verification failed for AP {}: {}", prop.id, result
                )
                # Default to neutral on error
                nli_results.append(
                    NLIResult(
                        proposition=prop,
                        label=NLILabel.NEUTRAL,
                        confidence=0.0,
                        premise_used="[Error during verification]",
                        all_scores={},
                        conflict_aware_result=None,
                    )
                )
            else:
                nli_results.append(result)

        return nli_results

    async def _verify_proposition_consensus(
        self,
        proposition: AtomicProposition,
        evidence: list[SearchResult],
    ) -> NLIResult:
        """
        Verify a single AP against ALL available evidence using weighted
        consensus scoring.

        Each passage contributes:
            w_j · (P(entail) − P(contradict))

        to the final consensus score S(p_i).
        """
        if not evidence:
            return NLIResult(
                proposition=proposition,
                label=NLILabel.NEUTRAL,
                confidence=0.0,
                premise_used="[No evidence found]",
                all_scores={},
                conflict_aware_result=None,
            )

        # Score ALL passages (no short-circuiting)
        passage_scores: list[NLIPassageScore] = []
        best_premise = evidence[0].snippet
        best_entailment_score = -1.0

        for source in evidence:
            _label, _confidence, all_scores = await self.verify_claim(
                premise=source.snippet,
                hypothesis=proposition.text,
            )

            # Retrieval relevance weight w_j
            w_j = max(source.relevance_score, 0.01)  # Floor at 0.01

            p_entail = all_scores.get("entailment", 0.0)
            p_contra = all_scores.get("contradiction", 0.0)
            p_neutral = all_scores.get("neutral", 0.0)
            contribution = w_j * (p_entail - p_contra)

            passage_scores.append(
                NLIPassageScore(
                    passage_snippet=source.snippet[:500],  # Truncate for serialization
                    source_url=source.url,
                    relevance_weight=round(w_j, 6),
                    entailment_prob=round(p_entail, 6),
                    contradiction_prob=round(p_contra, 6),
                    neutral_prob=round(p_neutral, 6),
                    weighted_contribution=round(contribution, 6),
                )
            )

            # Track best premise for backward-compatible `premise_used` field
            if p_entail > best_entailment_score:
                best_entailment_score = p_entail
                best_premise = source.snippet

        # Aggregate consensus
        consensus_result = self._compute_consensus(passage_scores)

        # Derive the top-level NLI label from the consensus verdict
        consensus_label = self._consensus_verdict_to_nli_label(
            consensus_result.verdict
        )

        # Confidence = absolute magnitude of the consensus score (0–1 range)
        consensus_confidence = min(abs(consensus_result.consensus_score), 1.0)

        # Build the all_scores dict from the best-scoring passage for backward compat
        best_all_scores = {}
        if passage_scores:
            best_ps = max(
                passage_scores, key=lambda ps: ps.entailment_prob
            )
            best_all_scores = {
                "entailment": best_ps.entailment_prob,
                "contradiction": best_ps.contradiction_prob,
                "neutral": best_ps.neutral_prob,
            }

        return NLIResult(
            proposition=proposition,
            label=consensus_label,
            confidence=round(consensus_confidence, 6),
            premise_used=best_premise,
            all_scores=best_all_scores,
            conflict_aware_result=consensus_result,
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
                refined = refined.replace(
                    r.proposition.text,
                    "[REMOVED: contradicted by evidence]",
                )

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("NLI model unloaded")
