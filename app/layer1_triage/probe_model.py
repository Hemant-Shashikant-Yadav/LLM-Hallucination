"""
Layer 1: Semantic Entropy Probe (SEP) for Hallucination Triage.

Implements the core equation from the thesis:
    P(Hallucination | h_t^(l)) = σ(W_SEP · h_t^(l) + b_SEP)

The probe is a lightweight logistic classifier trained on hidden-state
activations. It computes semantic entropy H_sem and applies the safety
threshold T_safe to route queries through the appropriate pipeline layer.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

from app.config import settings
from app.layer1_triage.activation_hook import ActivationBundle
from app.models.schemas import TriageDecision, TriageResult


class SemanticEntropyProbe(nn.Module):
    """
    Lightweight logistic regression probe for hallucination detection.

    Maps hidden-state activations h_t^(l) ∈ ℝ^d to a scalar
    hallucination probability via a single linear layer + sigmoid.
    """

    def __init__(self, input_dim: int) -> None:
        """
        Args:
            input_dim: Dimensionality of the input hidden-state vector.
                       For flattened multi-layer input: num_layers * hidden_dim.
        """
        super().__init__()
        # W_SEP ∈ ℝ^(1 × d), b_SEP ∈ ℝ^1
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        logger.info("SemanticEntropyProbe initialized | input_dim={}", input_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute P(Hallucination | h).

        Args:
            h: Hidden-state vector of shape [batch, input_dim] or [input_dim].

        Returns:
            Tensor of shape [batch, 1] or [1] with hallucination probabilities.
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)
        logits = self.linear(h)
        return self.sigmoid(logits)


class TriageEngine:
    """
    Orchestrates the Layer 1 triage pipeline:
    1. Receives activation bundle from the forward hook
    2. Runs the SEP probe to get P(Hallucination)
    3. Computes semantic entropy H_sem
    4. Applies T_safe threshold for routing decision
    """

    def __init__(
        self,
        t_safe: float | None = None,
        probe_weights_path: Path | None = None,
    ) -> None:
        self._t_safe = t_safe or settings.t_safe
        self._probe_weights_path = probe_weights_path or settings.probe_weights_path
        self._probe: SemanticEntropyProbe | None = None
        self._is_trained = False

        logger.info("TriageEngine initialized | T_safe={}", self._t_safe)

    # -------------------------------------------------------------------------
    # Probe Lifecycle
    # -------------------------------------------------------------------------

    def initialize_probe(self, input_dim: int) -> None:
        """
        Initialize the probe with the correct input dimensionality.

        If pretrained weights exist, load them. Otherwise, the probe
        operates in proxy mode using logprob-based entropy estimation.
        """
        self._probe = SemanticEntropyProbe(input_dim)

        if self._probe_weights_path and self._probe_weights_path.exists():
            state_dict = torch.load(
                self._probe_weights_path,
                map_location="cpu",
                weights_only=True,
            )
            self._probe.load_state_dict(state_dict)
            self._is_trained = True
            logger.info("Loaded pretrained probe weights from {}", self._probe_weights_path)
        else:
            self._is_trained = False
            logger.warning(
                "No pretrained probe weights found. Using logprob-based entropy proxy."
            )

        self._probe.eval()

    # -------------------------------------------------------------------------
    # Triage Logic
    # -------------------------------------------------------------------------

    async def triage(
        self,
        activation_bundle: ActivationBundle,
        draft_response: str,
        generation_logprobs: list[float] | None = None,
    ) -> TriageResult:
        """
        Perform Layer 1 triage on a draft LLM response.

        Args:
            activation_bundle: Hidden-state activations from the probe model.
            draft_response: The raw LLM-generated text.
            generation_logprobs: Optional token-level log probabilities
                                 (used as proxy when probe is untrained).

        Returns:
            TriageResult with decision, entropy, and probability scores.
        """
        import time

        start = time.perf_counter()

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._triage_sync,
            activation_bundle,
            draft_response,
            generation_logprobs,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        result.latency_ms = elapsed_ms

        logger.info(
            "Triage complete | decision={} | H_sem={:.4f} | P(hall)={:.4f} | {:.1f}ms",
            result.decision.value,
            result.semantic_entropy,
            result.hallucination_probability,
            elapsed_ms,
        )

        return result

    def _triage_sync(
        self,
        activation_bundle: ActivationBundle,
        draft_response: str,
        generation_logprobs: list[float] | None,
    ) -> TriageResult:
        """Synchronous triage logic — called from executor."""

        if self._is_trained and self._probe is not None:
            # Use the trained SEP probe
            h = activation_bundle.flattened()  # [num_layers * hidden_dim]
            with torch.no_grad():
                prob = self._probe(h).item()
        else:
            # Fallback: estimate uncertainty from generation logprobs
            prob = self._estimate_from_logprobs(generation_logprobs)

        # Compute semantic entropy
        h_sem = self._compute_semantic_entropy(prob)

        # Apply threshold
        decision = self._apply_threshold(h_sem, draft_response)

        return TriageResult(
            decision=decision,
            semantic_entropy=h_sem,
            hallucination_probability=prob,
            threshold_used=self._t_safe,
            draft_response=draft_response,
            latency_ms=0.0,  # Will be set by caller
        )

    # -------------------------------------------------------------------------
    # Entropy Computation
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_semantic_entropy(hallucination_prob: float) -> float:
        """
        Compute Shannon entropy over the binary hallucination distribution.

        H_sem = -[p·log₂(p) + (1-p)·log₂(1-p)]

        Args:
            hallucination_prob: P(Hallucination) ∈ [0, 1].

        Returns:
            Semantic entropy H_sem ∈ [0, 1].
        """
        p = max(min(hallucination_prob, 1.0 - 1e-10), 1e-10)
        q = 1.0 - p
        return -(p * math.log2(p) + q * math.log2(q))

    @staticmethod
    def _estimate_from_logprobs(logprobs: list[float] | None) -> float:
        """
        Proxy estimation: derive hallucination probability from logprobs.

        Low mean logprob → high uncertainty → higher hallucination probability.
        This is a simple heuristic used when the probe is untrained.
        """
        if not logprobs:
            # No logprobs available → maximum uncertainty
            return 0.5

        mean_logprob = sum(logprobs) / len(logprobs)

        # Map logprobs to [0, 1] probability range
        # More negative logprob → higher uncertainty → higher hall. prob
        # Typical logprobs range from -15 (very uncertain) to 0 (certain)
        # We use a sigmoid-like mapping centered around -2.0
        prob = 1.0 / (1.0 + math.exp(2.0 * (mean_logprob + 2.0)))
        return max(min(prob, 0.99), 0.01)

    def _apply_threshold(
        self,
        h_sem: float,
        draft_response: str,
    ) -> TriageDecision:
        """
        Apply the safety threshold T_safe and classify uncertainty type.

        Args:
            h_sem: Computed semantic entropy.
            draft_response: The draft response text (used for heuristic
                           classification of knowledge vs. reasoning uncertainty).

        Returns:
            TriageDecision enum value.
        """
        if h_sem < self._t_safe:
            return TriageDecision.SAFE

        # Heuristic: classify uncertainty type based on response characteristics
        # Reasoning-type indicators: mathematical expressions, logical connectives,
        # step-by-step patterns, conditional statements
        reasoning_indicators = [
            "therefore", "because", "if ", "then ", "step ",
            "first,", "second,", "thus", "hence",
            "calculate", "compute", "solve", "proof",
            "=", "≥", "≤", "+", "×",
        ]

        lower_response = draft_response.lower()
        reasoning_score = sum(
            1 for indicator in reasoning_indicators
            if indicator in lower_response
        )

        if reasoning_score >= 3:
            return TriageDecision.REASONING_UNCERTAIN
        else:
            return TriageDecision.KNOWLEDGE_UNCERTAIN

    # -------------------------------------------------------------------------
    # Training Interface
    # -------------------------------------------------------------------------

    def train_probe(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> dict[str, float]:
        """
        Train the SEP probe on labeled activation data.

        Args:
            activations: Tensor of shape [N, input_dim].
            labels: Binary labels [N] (1 = hallucination, 0 = factual).
            epochs: Number of training epochs.
            lr: Learning rate.

        Returns:
            Training metrics dict with final loss and accuracy.
        """
        if self._probe is None:
            raise RuntimeError("Probe not initialized. Call initialize_probe() first.")

        self._probe.train()
        optimizer = torch.optim.Adam(self._probe.parameters(), lr=lr)
        criterion = nn.BCELoss()

        labels_float = labels.float().unsqueeze(1)

        best_loss = float("inf")
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self._probe(activations)
            loss = criterion(predictions, labels_float)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

            if (epoch + 1) % 20 == 0:
                with torch.no_grad():
                    preds_binary = (predictions > 0.5).float()
                    accuracy = (preds_binary == labels_float).float().mean().item()
                logger.info(
                    "Probe training | epoch={}/{} | loss={:.4f} | acc={:.4f}",
                    epoch + 1, epochs, loss.item(), accuracy,
                )

        self._probe.eval()
        self._is_trained = True

        # Final metrics
        with torch.no_grad():
            final_preds = self._probe(activations)
            final_loss = criterion(final_preds, labels_float).item()
            final_acc = ((final_preds > 0.5).float() == labels_float).float().mean().item()

        logger.info(
            "Probe training complete | final_loss={:.4f} | final_acc={:.4f}",
            final_loss, final_acc,
        )

        return {"loss": final_loss, "accuracy": final_acc}

    def save_probe(self, path: Path) -> None:
        """Save the trained probe weights to disk."""
        if self._probe is None:
            raise RuntimeError("No probe to save.")

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._probe.state_dict(), path)
        logger.info("Probe weights saved to {}", path)
