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
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
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
    t_safe_reasoning: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "T_safe for quantitative (reasoning) queries. Calibrated on the GSM8K dev "
            "slice: llama2:7b drafts were mostly wrong even at H_sem = 0, so 0.0 sends "
            "every quantitative query to Layer 3."
        ),
    )
    triage_mode: Literal["semantic_entropy", "surrogate_probe"] = Field(
        default="semantic_entropy",
        description=(
            "Layer 1 uncertainty signal. 'semantic_entropy' samples the generator "
            "and clusters answers by meaning (Farquhar et al., 2024). "
            "'surrogate_probe' uses the TinyLlama dual-head probe (needs trained weights)."
        ),
    )
    task_router_mode: Literal["llm", "heuristic"] = Field(
        default="llm",
        description=(
            "'llm': rules decide clear-cut questions, the pipeline model decides ambiguous "
            "ones. 'heuristic': rules only (no extra LLM call)."
        ),
    )
    se_num_samples: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Extra stochastic samples drawn from the generator for semantic entropy.",
    )
    se_sample_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the semantic entropy samples.",
    )
    se_max_tokens: int = Field(
        default=512,
        ge=32,
        description="Max tokens per semantic entropy sample.",
    )
    se_min_samples: int = Field(
        default=2,
        ge=1,
        description=(
            "Minimum successful samples needed to trust H_sem. With fewer, entropy "
            "over (draft + samples) collapses towards 0 and would wrongly mark the "
            "query SAFE, so triage escalates instead."
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
    layer2_revision_mode: Literal["evidence_gated", "full_rewrite"] = Field(
        default="evidence_gated",
        description=(
            "'evidence_gated' edits only claims that evidence contradicts and keeps the "
            "draft otherwise (do-no-harm). 'full_rewrite' is the Phase 1 rewrite + CoVe, "
            "which helped llama2:7b but over-corrected the stronger qwen2.5:3b drafts."
        ),
    )
    layer2_question_evidence_for_claims: bool = Field(
        default=False,
        description=(
            "Also verify each claim against the question-level search results. Raises "
            "evidence coverage but, on the qwen2.5:3b dev slice, added spurious "
            "contradictions (truthful 77.5% → 70.0% vs 77.5% → 77.5% without)."
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
        default=6,
        ge=1,
        le=20,
        description="Maximum number of plan steps in the execution graph.",
    )
    halluclean_reasoning_mode: Literal["program_vote", "plan_judge"] = Field(
        default="program_vote",
        description=(
            "Layer 3 for quantitative queries. 'program_vote' = program-verified "
            "self-consistency (PAL + PROVE); 'plan_judge' = LLM judge + refiner."
        ),
    )
    program_vote_num_programs: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Independent programs written per quantitative query in program_vote mode.",
    )
    program_vote_weighting: Literal["sample_primary", "program_primary"] = Field(
        default="sample_primary",
        description=(
            "'sample_primary': majority over reasoning samples, programs add "
            "program_vote_weight each. 'program_primary' (Phase 1): only "
            "program-confirmed answers can win."
        ),
    )
    program_vote_weight: float = Field(
        default=0.5,
        ge=0.0,
        description="Vote weight of one program relative to one reasoning sample.",
    )
    l3_extra_samples: int = Field(
        default=6,
        ge=0,
        le=20,
        description=(
            "Extra reasoning samples drawn in Layer 3, added to the draft and the "
            "Layer 1 samples for the self-consistency vote."
        ),
    )
    halluclean_executor_mode: Literal["single_pass", "per_step"] = Field(
        default="single_pass",
        description=(
            "'single_pass' executes the whole plan in one plan-guided call; "
            "'per_step' runs one LLM call per plan node."
        ),
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
        default="llama2:7b",
        description="Default Ollama model for text generation.",
    )
    ollama_judge_model: str = Field(
        default="llama3.1:8b",
        description="Stronger model used exclusively for Layer 2 NLI parsing and Layer 3 HalluClean reasoning.",
    )
    ollama_timeout: float = Field(
        default=600.0,
        ge=5.0,
        description=(
            "Timeout in seconds for one Ollama call. Measured from when the request "
            "is sent, after the client-side concurrency gate (queue time excluded)."
        ),
    )
    ollama_num_ctx: int = Field(
        default=4096,
        ge=512,
        description=(
            "Context window (prompt + completion) sent as num_ctx on every call. "
            "Ollama silently drops the start of prompts that exceed it. "
            "llama2:7b supports at most 4096."
        ),
    )
    ollama_max_concurrency: int = Field(
        default=1,
        ge=1,
        description=(
            "Max in-flight Ollama requests from this process. Match the server's "
            "OLLAMA_NUM_PARALLEL; extra requests otherwise queue on the server and "
            "burn their HTTP timeout while waiting."
        ),
    )
    ollama_max_retries: int = Field(
        default=2,
        ge=0,
        description="Retries for transient Ollama failures (5xx, runner crash, connection reset).",
    )
    ollama_keep_alive: str = Field(
        default="10m",
        description="How long Ollama keeps a model loaded after a call.",
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
