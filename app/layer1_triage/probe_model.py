"""
Layer 1: Surrogate Proxy Probing + Dual-Head Uncertainty Classifier.

Algorithmic Novelties (for M.Tech thesis):

1. **Surrogate Proxy Probing**
   We cannot extract hidden states from the quantized Ollama generator.
   Instead, we use TinyLlama-1.1B as a white-box surrogate: the Ollama-
   generated draft and original prompt are concatenated and fed through
   TinyLlama. A PyTorch forward hook on the **final** transformer layer
   captures the last-token hidden state h_t^(L) ∈ ℝ^d for probing.

2. **Dual-Head Uncertainty Classifier**
   Replaces the simple logistic probe. A single shared linear projection
   feeds two heads:
     • Head 1 (Semantic Entropy):  H_sem = σ(W₁ · h + b₁) ∈ [0, 1]
     • Head 2 (Error Router):     logits = W₂ · h + b₂  ∈ ℝ²
       with softmax → P(knowledge), P(reasoning)

   The T_safe threshold is applied to H_sem, and the argmax of Head 2
   replaces the old keyword-heuristic routing.
"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

from app.config import settings
from app.layer1_triage.activation_hook import ActivationBundle
from app.models.schemas import (
    ErrorClassification,
    TriageDecision,
    TriageResult,
)


# =============================================================================
# Dual-Head Uncertainty Classifier
# =============================================================================


class DualHeadUncertaintyClassifier(nn.Module):
    """
    Dual-head linear classifier over surrogate hidden states.

    Head 1 — Semantic Entropy:
        H_sem = σ(W₁ · h + b₁)   ∈ [0, 1]

    Head 2 — Error-type Router:
        logits = W₂ · h + b₂     ∈ ℝ²  (knowledge, reasoning)
        P(class) = softmax(logits)
    """

    ERROR_CLASSES: list[str] = ["knowledge", "reasoning"]

    def __init__(self, input_dim: int) -> None:
        """
        Args:
            input_dim: Dimensionality of the input hidden-state vector h_t^(L).
        """
        super().__init__()

        # Head 1: Semantic Entropy — scalar output with sigmoid
        self.entropy_head = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Head 2: Error Router — 2-class logits (knowledge / reasoning)
        self.router_head = nn.Linear(input_dim, 2)

        logger.info(
            "DualHeadUncertaintyClassifier initialized | input_dim={} | heads=2",
            input_dim,
        )

    def forward(
        self,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both heads.

        Args:
            h: Hidden-state vector [batch, input_dim] or [input_dim].

        Returns:
            (h_sem, router_logits):
                h_sem:         [batch, 1]  — semantic entropy predictions
                router_logits: [batch, 2]  — raw logits for error routing
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)

        h_sem = self.sigmoid(self.entropy_head(h))  # [B, 1]
        router_logits = self.router_head(h)          # [B, 2]

        return h_sem, router_logits


# =============================================================================
# Surrogate Hidden-State Extractor
# =============================================================================


class SurrogateExtractor:
    """
    Extracts the final-layer hidden state h_t^(L) from TinyLlama
    for a given (prompt, draft_response) pair.

    Uses a PyTorch forward hook on the last transformer layer.
    This acts as a white-box surrogate for the opaque Ollama generator.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name or settings.probe_model_name
        self._device = device
        self._model = None
        self._tokenizer = None
        self._final_hidden_state: torch.Tensor | None = None
        self._hook_handle = None
        self._hidden_dim: int = 0
        self._is_loaded = False

        logger.info(
            "SurrogateExtractor created | model={} | device={}",
            self._model_name,
            self._device,
        )

    @property
    def hidden_dim(self) -> int:
        """Dimensionality of the surrogate's hidden state."""
        return self._hidden_dim

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    # -------------------------------------------------------------------------
    # Model Loading (cold-start safe)
    # -------------------------------------------------------------------------

    async def load_model(self) -> None:
        """Load TinyLlama in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self) -> None:
        """Synchronous model loading with cold-start handling."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading surrogate probe model: {}", self._model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=torch.float32,
            device_map=self._device,
            trust_remote_code=True,
        )
        self._model.eval()

        # Resolve hidden dim from model config
        config = self._model.config
        for attr in ("hidden_size", "d_model", "n_embd"):
            if hasattr(config, attr):
                self._hidden_dim = getattr(config, attr)
                break
        else:
            raise ValueError(
                f"Cannot determine hidden_dim for {type(config).__name__}"
            )

        self._is_loaded = True
        logger.info(
            "Surrogate model loaded | hidden_dim={} | model={}",
            self._hidden_dim,
            self._model_name,
        )

    # -------------------------------------------------------------------------
    # Forward Hook on Final Layer
    # -------------------------------------------------------------------------

    def _get_final_layer_module(self) -> torch.nn.Module:
        """Resolve the final transformer layer module."""
        if self._model is None:
            raise RuntimeError("Surrogate model not loaded.")

        model_base = self._model
        # Unwrap causal LM wrapper
        if hasattr(model_base, "model"):
            model_base = model_base.model
        elif hasattr(model_base, "transformer"):
            model_base = model_base.transformer
        elif hasattr(model_base, "gpt_neox"):
            model_base = model_base.gpt_neox

        # Find the layer container
        for attr in ("layers", "h", "block", "blocks"):
            if hasattr(model_base, attr):
                layers = getattr(model_base, attr)
                if isinstance(layers, nn.ModuleList):
                    return layers[-1]  # Final layer

        raise ValueError(
            f"Cannot locate transformer layers in {type(self._model).__name__}"
        )

    def _hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
    ) -> None:
        """Capture the final-layer hidden state (last token)."""
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # hidden: [batch, seq_len, hidden_dim] — take last token
        self._final_hidden_state = hidden[:, -1, :].detach()

    # -------------------------------------------------------------------------
    # Extraction
    # -------------------------------------------------------------------------

    async def extract_hidden_state(
        self,
        prompt: str,
        draft_response: str,
        max_length: int = 512,
    ) -> torch.Tensor:
        """
        Run the surrogate forward pass and extract h_t^(L).

        The prompt and draft are concatenated to give the surrogate
        full context about what the generator produced and why.

        Args:
            prompt: The original user prompt.
            draft_response: The Ollama-generated draft text.
            max_length: Maximum sequence length for tokenization.

        Returns:
            Tensor of shape [hidden_dim] — the final-layer last-token hidden state.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._extract_sync, prompt, draft_response, max_length
        )

    def _extract_sync(
        self,
        prompt: str,
        draft_response: str,
        max_length: int,
    ) -> torch.Tensor:
        """Synchronous extraction — called from executor."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "Surrogate model not loaded. Call load_model() first."
            )

        # Concatenate prompt + draft for surrogate probing
        surrogate_input = (
            f"<|user|>\n{prompt}\n<|assistant|>\n{draft_response}"
        )

        inputs = self._tokenizer(
            surrogate_input,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(self._device)

        # Register hook on the final transformer layer
        final_layer = self._get_final_layer_module()
        self._final_hidden_state = None
        handle = final_layer.register_forward_hook(self._hook_fn)

        try:
            with torch.no_grad():
                self._model(**inputs)
        finally:
            handle.remove()

        if self._final_hidden_state is None:
            raise RuntimeError(
                "Forward hook did not capture hidden state. "
                "Check model architecture compatibility."
            )

        # Squeeze batch dimension: [1, hidden_dim] → [hidden_dim]
        return self._final_hidden_state.squeeze(0)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def unload(self) -> None:
        """Release surrogate model memory."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Surrogate model unloaded")


# =============================================================================
# Triage Engine (Orchestrator)
# =============================================================================


class TriageEngine:
    """
    Orchestrates the Layer 1 triage pipeline with the novel dual-head
    classifier over surrogate-extracted hidden states:

    1. SurrogateExtractor passes (prompt, draft) through TinyLlama
       → extracts h_t^(L)
    2. DualHeadUncertaintyClassifier produces H_sem and error-type logits
    3. T_safe threshold applied for routing decision

    Falls back to logprob-based proxy when the probe is untrained or
    the surrogate model is unavailable (graceful cold-start handling).
    """

    def __init__(
        self,
        t_safe: float | None = None,
        probe_weights_path: Path | None = None,
    ) -> None:
        self._t_safe = t_safe or settings.t_safe
        self._probe_weights_path = probe_weights_path or settings.probe_weights_path
        self._classifier: DualHeadUncertaintyClassifier | None = None
        self._surrogate: SurrogateExtractor | None = None
        self._is_trained = False

        logger.info("TriageEngine initialized | T_safe={}", self._t_safe)

    # -------------------------------------------------------------------------
    # Probe Lifecycle
    # -------------------------------------------------------------------------

    def initialize_probe(self, input_dim: int) -> None:
        """
        Initialize the dual-head classifier with the correct input dimensionality.

        If pretrained weights exist, load them. Otherwise, the classifier
        operates in proxy mode using logprob-based entropy estimation.
        """
        self._classifier = DualHeadUncertaintyClassifier(input_dim)

        if self._probe_weights_path and self._probe_weights_path.exists():
            state_dict = torch.load(
                self._probe_weights_path,
                map_location="cpu",
                weights_only=True,
            )
            self._classifier.load_state_dict(state_dict)
            self._is_trained = True
            logger.info(
                "Loaded pretrained dual-head weights from {}",
                self._probe_weights_path,
            )
        else:
            self._is_trained = False
            logger.warning(
                "No pretrained dual-head weights found. "
                "Using logprob-based entropy proxy."
            )

        self._classifier.eval()

    def set_surrogate(self, surrogate: SurrogateExtractor) -> None:
        """Attach a loaded SurrogateExtractor instance."""
        self._surrogate = surrogate

    # -------------------------------------------------------------------------
    # Triage Logic
    # -------------------------------------------------------------------------

    async def triage(
        self,
        prompt: str,
        draft_response: str,
        activation_bundle: ActivationBundle | None = None,
        generation_logprobs: list[float] | None = None,
    ) -> TriageResult:
        """
        Perform Layer 1 triage on a draft LLM response.

        The preferred path uses the surrogate extractor + dual-head classifier.
        Falls back gracefully when components are unavailable.

        Args:
            prompt: The original user prompt.
            draft_response: The raw Ollama-generated text.
            activation_bundle: Hidden-state activations (legacy; used if
                               surrogate is unavailable but extractor ran).
            generation_logprobs: Optional token-level log probabilities
                                 (used as proxy when probe is untrained).

        Returns:
            TriageResult with decision, entropy, probability, and
            error_classification from the dual-head classifier.
        """
        start = time.perf_counter()

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._triage_sync,
            prompt,
            draft_response,
            activation_bundle,
            generation_logprobs,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        result.latency_ms = elapsed_ms

        logger.info(
            "Triage complete | decision={} | H_sem={:.4f} | P(hall)={:.4f} | "
            "error_type={} | {:.1f}ms",
            result.decision.value,
            result.semantic_entropy,
            result.hallucination_probability,
            (
                result.error_classification.error_type
                if result.error_classification
                else "n/a"
            ),
            elapsed_ms,
        )

        return result

    def _triage_sync(
        self,
        prompt: str,
        draft_response: str,
        activation_bundle: ActivationBundle | None,
        generation_logprobs: list[float] | None,
    ) -> TriageResult:
        """Synchronous triage logic — called from executor."""
        error_classification: ErrorClassification | None = None

        # -----------------------------------------------------------------
        # Path A: Surrogate Proxy Probing + Dual-Head Classifier
        # -----------------------------------------------------------------
        if (
            self._surrogate is not None
            and self._surrogate.is_loaded
            and self._classifier is not None
        ):
            try:
                h = self._surrogate._extract_sync(
                    prompt, draft_response, max_length=512
                )

                with torch.no_grad():
                    h_sem_tensor, router_logits = self._classifier(h)

                h_sem_val = h_sem_tensor.item()
                prob = h_sem_val  # H_sem doubles as hallucination probability

                # Error routing from Head 2
                router_probs = torch.softmax(router_logits, dim=1).squeeze(0)
                knowledge_logit = router_logits[0, 0].item()
                reasoning_logit = router_logits[0, 1].item()
                predicted_idx = torch.argmax(router_probs).item()
                routing_confidence = router_probs[predicted_idx].item()
                error_type = DualHeadUncertaintyClassifier.ERROR_CLASSES[
                    predicted_idx
                ]

                error_classification = ErrorClassification(
                    semantic_entropy=h_sem_val,
                    error_type=error_type,
                    knowledge_logit=knowledge_logit,
                    reasoning_logit=reasoning_logit,
                    routing_confidence=routing_confidence,
                )

                decision = self._apply_threshold_from_classifier(
                    h_sem_val, error_type
                )

                return TriageResult(
                    decision=decision,
                    semantic_entropy=h_sem_val,
                    hallucination_probability=prob,
                    error_classification=error_classification,
                    threshold_used=self._t_safe,
                    draft_response=draft_response,
                    latency_ms=0.0,
                )

            except Exception as exc:
                logger.warning(
                    "Surrogate + dual-head path failed, falling back: {}",
                    exc,
                )

        # -----------------------------------------------------------------
        # Path B: Legacy ActivationBundle + Dual-Head Classifier
        # -----------------------------------------------------------------
        if (
            activation_bundle is not None
            and self._classifier is not None
            and self._is_trained
        ):
            h = activation_bundle.flattened()

            # If the activation bundle dim doesn't match classifier input,
            # we must fall back
            expected_dim = self._classifier.entropy_head.in_features
            if h.shape[0] == expected_dim:
                with torch.no_grad():
                    h_sem_tensor, router_logits = self._classifier(h)

                h_sem_val = h_sem_tensor.item()
                prob = h_sem_val

                router_probs = torch.softmax(router_logits, dim=1).squeeze(0)
                knowledge_logit = router_logits[0, 0].item()
                reasoning_logit = router_logits[0, 1].item()
                predicted_idx = torch.argmax(router_probs).item()
                routing_confidence = router_probs[predicted_idx].item()
                error_type = DualHeadUncertaintyClassifier.ERROR_CLASSES[
                    predicted_idx
                ]

                error_classification = ErrorClassification(
                    semantic_entropy=h_sem_val,
                    error_type=error_type,
                    knowledge_logit=knowledge_logit,
                    reasoning_logit=reasoning_logit,
                    routing_confidence=routing_confidence,
                )

                decision = self._apply_threshold_from_classifier(
                    h_sem_val, error_type
                )

                return TriageResult(
                    decision=decision,
                    semantic_entropy=h_sem_val,
                    hallucination_probability=prob,
                    error_classification=error_classification,
                    threshold_used=self._t_safe,
                    draft_response=draft_response,
                    latency_ms=0.0,
                )

        # -----------------------------------------------------------------
        # Path C: Logprob-based Proxy Fallback
        # -----------------------------------------------------------------
        prob = self._estimate_from_logprobs(generation_logprobs)
        h_sem = self._compute_semantic_entropy(prob)
        decision = self._apply_threshold_heuristic(h_sem, draft_response)

        return TriageResult(
            decision=decision,
            semantic_entropy=h_sem,
            hallucination_probability=prob,
            error_classification=None,  # No classifier output in proxy mode
            threshold_used=self._t_safe,
            draft_response=draft_response,
            latency_ms=0.0,
        )

    # -------------------------------------------------------------------------
    # Threshold & Routing
    # -------------------------------------------------------------------------

    def _apply_threshold_from_classifier(
        self,
        h_sem: float,
        error_type: str,
    ) -> TriageDecision:
        """
        Apply T_safe on the dual-head H_sem and use the classifier's
        error-type prediction for routing (no keyword heuristics).
        """
        if h_sem < self._t_safe:
            return TriageDecision.SAFE

        if error_type == "reasoning":
            return TriageDecision.REASONING_UNCERTAIN
        return TriageDecision.KNOWLEDGE_UNCERTAIN

    def _apply_threshold_heuristic(
        self,
        h_sem: float,
        draft_response: str,
    ) -> TriageDecision:
        """
        Legacy heuristic fallback when the dual-head classifier is unavailable.

        Uses keyword indicators as a proxy for error-type classification.
        """
        if h_sem < self._t_safe:
            return TriageDecision.SAFE

        # Reasoning-type indicators
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
        return TriageDecision.KNOWLEDGE_UNCERTAIN

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

    # -------------------------------------------------------------------------
    # Training Interface (Dual-Head)
    # -------------------------------------------------------------------------

    def train_probe(
        self,
        activations: torch.Tensor,
        entropy_labels: torch.Tensor,
        route_labels: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
        entropy_loss_weight: float = 0.5,
    ) -> dict[str, float]:
        """
        Train the dual-head classifier on labeled activation data.

        Args:
            activations: Tensor of shape [N, input_dim].
            entropy_labels: Float labels [N] ∈ [0, 1] for H_sem regression.
            route_labels: Integer labels [N] ∈ {0, 1} for error-type classification
                          (0 = knowledge, 1 = reasoning).
            epochs: Number of training epochs.
            lr: Learning rate.
            entropy_loss_weight: Weight α for the entropy loss.
                                 Route loss weight = (1 - α).

        Returns:
            Training metrics dict with final losses and accuracies.
        """
        if self._classifier is None:
            raise RuntimeError(
                "Classifier not initialized. Call initialize_probe() first."
            )

        self._classifier.train()
        optimizer = torch.optim.Adam(self._classifier.parameters(), lr=lr)
        bce_loss_fn = nn.BCELoss()
        ce_loss_fn = nn.CrossEntropyLoss()

        entropy_labels_2d = entropy_labels.float().unsqueeze(1)
        route_labels_long = route_labels.long()

        best_loss = float("inf")
        for epoch in range(epochs):
            optimizer.zero_grad()

            h_sem_pred, router_logits = self._classifier(activations)

            # Head 1: Binary cross-entropy for semantic entropy
            loss_entropy = bce_loss_fn(h_sem_pred, entropy_labels_2d)

            # Head 2: Cross-entropy for error-type routing
            loss_route = ce_loss_fn(router_logits, route_labels_long)

            # Combined loss
            loss = (
                entropy_loss_weight * loss_entropy
                + (1.0 - entropy_loss_weight) * loss_route
            )

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

            if (epoch + 1) % 20 == 0:
                with torch.no_grad():
                    route_preds = torch.argmax(router_logits, dim=1)
                    route_acc = (
                        (route_preds == route_labels_long).float().mean().item()
                    )
                    entropy_mae = (
                        (h_sem_pred.squeeze() - entropy_labels.float())
                        .abs()
                        .mean()
                        .item()
                    )
                logger.info(
                    "Dual-head training | epoch={}/{} | loss={:.4f} "
                    "| entropy_mae={:.4f} | route_acc={:.4f}",
                    epoch + 1,
                    epochs,
                    loss.item(),
                    entropy_mae,
                    route_acc,
                )

        self._classifier.eval()
        self._is_trained = True

        # Final metrics
        with torch.no_grad():
            h_sem_final, logits_final = self._classifier(activations)
            final_entropy_loss = bce_loss_fn(
                h_sem_final, entropy_labels_2d
            ).item()
            final_route_loss = ce_loss_fn(
                logits_final, route_labels_long
            ).item()
            final_route_acc = (
                (torch.argmax(logits_final, dim=1) == route_labels_long)
                .float()
                .mean()
                .item()
            )
            final_entropy_mae = (
                (h_sem_final.squeeze() - entropy_labels.float())
                .abs()
                .mean()
                .item()
            )

        logger.info(
            "Dual-head training complete | entropy_loss={:.4f} | "
            "route_loss={:.4f} | route_acc={:.4f} | entropy_mae={:.4f}",
            final_entropy_loss,
            final_route_loss,
            final_route_acc,
            final_entropy_mae,
        )

        return {
            "entropy_loss": final_entropy_loss,
            "route_loss": final_route_loss,
            "route_accuracy": final_route_acc,
            "entropy_mae": final_entropy_mae,
        }

    def save_probe(self, path: Path) -> None:
        """Save the trained dual-head classifier weights to disk."""
        if self._classifier is None:
            raise RuntimeError("No classifier to save.")

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._classifier.state_dict(), path)
        logger.info("Dual-head classifier weights saved to {}", path)
