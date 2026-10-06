"""
Central configuration for the Defense-in-Depth Framework.

All tunable thresholds, model identifiers, API endpoints, and paths
are managed here via Pydantic BaseSettings with .env file support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve project root relative to this file
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class FrameworkSettings(BaseSettings):
    """
    Global configuration for the Defense-in-Depth pipeline.

    Values can be overridden via environment variables or a `.env` file
    placed in the project root directory.
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="DID_",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Compute Hardware Configuration
    # -------------------------------------------------------------------------
    device: str = Field(
        default_factory=_default_device,
        description="Compute device ('cuda' or 'cpu'). Auto-detects GPU if available.",
    )
    use_fp16: bool = Field(
        default=True,
        description="Use FP16 precision on CUDA for significant VRAM reduction and speedup.",
    )

    # -------------------------------------------------------------------------
    # Layer 1: Semantic Entropy Probe
    # -------------------------------------------------------------------------
    t_safe: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Safety threshold T_safe for semantic entropy. "
            "Queries with H_sem < T_safe bypass external verification."
        ),
    )
    probe_model_name: str = Field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description=(
            "HuggingFace model used for hidden-state extraction. "
            "Must be small enough for CPU inference."
        ),
    )
    probe_layer_fraction: float = Field(
        default=0.67,
        ge=0.0,
        le=1.0,
        description=(
            "Extract activations from the final (1 - fraction) of layers. "
            "0.67 means probe the last third of layers."
        ),
    )
    probe_weights_path: Path | None = Field(
        default=None,
        description="Path to pretrained SEP probe weights (.pt file). None = use logprob proxy.",
    )

    # -------------------------------------------------------------------------
    # Layer 2: NLI-RAG Verification
    # -------------------------------------------------------------------------
    nli_model_name: str = Field(
        default="cross-encoder/nli-deberta-v3-base",
        description="HuggingFace NLI model for claim verification.",
    )
    nli_confidence_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum softmax confidence to accept an Entailment label.",
    )
    nli_accept_threshold: float = Field(
        default=0.15,
        ge=-1.0,
        le=1.0,
        description=(
            "Consensus score S ≥ this → ACCEPT the claim. "
            "Used by the conflict-aware weighted NLI verifier."
        ),
    )
    nli_reject_threshold: float = Field(
        default=-0.15,
        ge=-1.0,
        le=1.0,
        description=(
            "Consensus score S ≤ this → REJECT the claim. "
            "Used by the conflict-aware weighted NLI verifier."
        ),
    )
    search_backend: Literal["duckduckgo", "tavily", "both"] = Field(
        default="duckduckgo",
        description="Web search backend for evidence retrieval.",
    )
    tavily_api_key: str = Field(
        default="",
        description="Tavily API key (required only if search_backend includes 'tavily').",
    )
    chromadb_path: str = Field(
        default=str(_PROJECT_ROOT / "data" / "chromadb"),
        description="Directory path for ChromaDB persistent storage.",
    )
    max_search_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to retrieve per AP.",
    )

    # -------------------------------------------------------------------------
    # Layer 3: HalluClean Structured Reasoning
    # -------------------------------------------------------------------------
    halluclean_max_steps: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum number of plan steps in the execution graph.",
    )
    halluclean_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for HalluClean planning/reasoning prompts.",
    )

    # -------------------------------------------------------------------------
    # Ollama Configuration
    # -------------------------------------------------------------------------
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the local Ollama server.",
    )
    ollama_model: str = Field(
        default="llama3.1:8b",
        description="Default Ollama model for text generation.",
    )
    ollama_timeout: float = Field(
        default=600.0,
        ge=5.0,
        description="Timeout in seconds for Ollama API calls.",
    )

    # -------------------------------------------------------------------------
    # Server Configuration
    # -------------------------------------------------------------------------
    host: str = Field(default="0.0.0.0", description="FastAPI server bind host.")
    port: int = Field(default=8000, ge=1, le=65535, description="FastAPI server bind port.")
    debug: bool = Field(default=False, description="Enable debug mode with auto-reload.")
    log_level: str = Field(default="INFO", description="Logging level.")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    eval_output_dir: str = Field(
        default=str(_PROJECT_ROOT / "evaluation" / "results"),
        description="Directory for benchmark output files.",
    )


# Singleton settings instance — import this from anywhere
settings = FrameworkSettings()
