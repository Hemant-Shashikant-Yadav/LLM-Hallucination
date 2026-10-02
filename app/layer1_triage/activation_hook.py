"""
Layer 1: Hidden-State Activation Extraction via PyTorch Forward Hooks.

Loads a compact HuggingFace causal language model and registers forward
hooks on the final third of transformer layers to extract hidden-state
tensors h_t^(l) ∈ ℝ^d for each token in the generated sequence.

These activations are consumed by the Semantic Entropy Probe
(probe_model.py) to estimate hallucination probability.
"""

from __future__ import annotations

import asyncio
from typing import Any

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings


class ActivationExtractor:
    """
    Extracts hidden-state activations from a HuggingFace causal LM.

    Uses PyTorch forward hooks to capture intermediate representations
    from the final third of transformer layers during inference.
    """

    def __init__(
        self,
        model_name: str | None = None,
        layer_fraction: float | None = None,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the activation extractor.

        Args:
            model_name: HuggingFace model identifier (defaults to config).
            layer_fraction: Fraction threshold; layers beyond this are probed.
            device: Torch device ('cpu' or 'cuda').
        """
        self._model_name = model_name or settings.probe_model_name
        self._layer_fraction = layer_fraction or settings.probe_layer_fraction
        self._device = device
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []
        self._captured_activations: dict[int, torch.Tensor] = {}
        self._target_layer_indices: list[int] = []

        logger.info(
            "ActivationExtractor initialized | model={} | device={}",
            self._model_name,
            self._device,
        )

    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------

    async def load_model(self) -> None:
        """Load the HuggingFace model and tokenizer (runs in thread pool)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self) -> None:
        """Synchronous model loading — called from executor."""
        logger.info("Loading probe model: {}", self._model_name)

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
            output_hidden_states=True,
        )
        self._model.eval()

        # Identify target layers (final third of the transformer stack)
        total_layers = self._get_total_layers()
        start_layer = int(total_layers * self._layer_fraction)
        self._target_layer_indices = list(range(start_layer, total_layers))

        logger.info(
            "Model loaded | total_layers={} | probing layers {} (indices {}–{})",
            total_layers,
            len(self._target_layer_indices),
            self._target_layer_indices[0] if self._target_layer_indices else "?",
            self._target_layer_indices[-1] if self._target_layer_indices else "?",
        )

    def _get_total_layers(self) -> int:
        """Determine the number of transformer layers in the loaded model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        config = self._model.config
        # Different architectures use different attribute names
        for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            if hasattr(config, attr):
                return getattr(config, attr)

        raise ValueError(
            f"Cannot determine layer count for model config: {type(config).__name__}"
        )

    # -------------------------------------------------------------------------
    # Hook Registration
    # -------------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Register forward hooks on target transformer layers."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        # Clear any existing hooks
        self._remove_hooks()
        self._captured_activations.clear()

        # Find the transformer layer modules
        layer_modules = self._get_layer_modules()

        for layer_idx in self._target_layer_indices:
            if layer_idx < len(layer_modules):
                handle = layer_modules[layer_idx].register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self._hook_handles.append(handle)

        logger.debug("Registered {} forward hooks", len(self._hook_handles))

    def _get_layer_modules(self) -> list[torch.nn.Module]:
        """Extract the list of transformer layer modules from the model."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        # Try common architecture patterns
        model_base = self._model

        # Unwrap the outer causal LM wrapper to get the base model
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
                if isinstance(layers, torch.nn.ModuleList):
                    return list(layers)

        raise ValueError(
            f"Cannot find transformer layers in model architecture: "
            f"{type(self._model).__name__}"
        )

    def _make_hook(self, layer_idx: int):
        """Create a forward hook closure for a specific layer."""

        def hook_fn(
            module: torch.nn.Module,
            input: Any,
            output: Any,
        ) -> None:
            # Output can be a tuple; the hidden state is typically the first element
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Store detached copy to avoid memory leaks
            self._captured_activations[layer_idx] = hidden_states.detach()

        return hook_fn

    def _remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    # -------------------------------------------------------------------------
    # Activation Extraction
    # -------------------------------------------------------------------------

    async def extract_activations(
        self,
        text: str,
        max_length: int = 512,
    ) -> ActivationBundle:
        """
        Extract hidden-state activations for the given text.

        Args:
            text: Input text to process through the model.
            max_length: Maximum sequence length.

        Returns:
            ActivationBundle containing stacked activation tensors.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._extract_sync, text, max_length
        )

    def _extract_sync(self, text: str, max_length: int) -> ActivationBundle:
        """Synchronous extraction — called from executor."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(self._device)

        # Register hooks and run forward pass
        self._register_hooks()
        try:
            with torch.no_grad():
                outputs = self._model(**inputs)
        finally:
            self._remove_hooks()

        # Stack captured activations: shape [num_layers, seq_len, hidden_dim]
        if not self._captured_activations:
            raise RuntimeError("No activations captured. Check hook registration.")

        sorted_layers = sorted(self._captured_activations.keys())
        stacked = torch.stack(
            [self._captured_activations[idx] for idx in sorted_layers],
            dim=0,
        )

        # Remove batch dimension: [num_layers, seq_len, hidden_dim]
        if stacked.dim() == 4 and stacked.size(1) == 1:
            stacked = stacked.squeeze(1)

        seq_len = stacked.size(1)
        hidden_dim = stacked.size(2)

        logger.debug(
            "Extracted activations | layers={} | seq_len={} | hidden_dim={}",
            len(sorted_layers),
            seq_len,
            hidden_dim,
        )

        return ActivationBundle(
            activations=stacked,
            layer_indices=sorted_layers,
            hidden_dim=hidden_dim,
            token_count=seq_len,
            model_name=self._model_name,
        )

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def unload(self) -> None:
        """Unload the model and free memory."""
        self._remove_hooks()
        self._captured_activations.clear()
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Probe model unloaded")


# =============================================================================
# Activation Bundle
# =============================================================================


class ActivationBundle:
    """
    Container for extracted hidden-state activations.

    Provides convenient access to the stacked tensor and metadata.
    """

    __slots__ = (
        "activations",
        "layer_indices",
        "hidden_dim",
        "token_count",
        "model_name",
    )

    def __init__(
        self,
        activations: torch.Tensor,
        layer_indices: list[int],
        hidden_dim: int,
        token_count: int,
        model_name: str,
    ) -> None:
        self.activations = activations  # [num_layers, seq_len, hidden_dim]
        self.layer_indices = layer_indices
        self.hidden_dim = hidden_dim
        self.token_count = token_count
        self.model_name = model_name

    def mean_pooled(self) -> torch.Tensor:
        """
        Mean-pool across sequence length for each layer.

        Returns:
            Tensor of shape [num_layers, hidden_dim].
        """
        return self.activations.mean(dim=1)

    def last_token(self) -> torch.Tensor:
        """
        Extract the last token's hidden state from each layer.

        Returns:
            Tensor of shape [num_layers, hidden_dim].
        """
        return self.activations[:, -1, :]

    def flattened(self) -> torch.Tensor:
        """
        Flatten all layer activations into a single vector (last token).

        Returns:
            Tensor of shape [num_layers * hidden_dim].
        """
        return self.last_token().flatten()

    def __repr__(self) -> str:
        return (
            f"ActivationBundle(model={self.model_name!r}, "
            f"layers={len(self.layer_indices)}, "
            f"tokens={self.token_count}, "
            f"hidden_dim={self.hidden_dim})"
        )
