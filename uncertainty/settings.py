
"""
Settings for the uncertainty study (Objectives 1 and 2).

Separate from the pipeline's `app/config.py`: the prefix is UQ_ and the optional env file
is uncertainty/uq.env, so pipeline (DID_) settings are never read or changed here.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


class UQSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PACKAGE_DIR / "uq.env"),
        env_file_encoding="utf-8",
        env_prefix="UQ_",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Ollama ---------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: float = Field(default=900.0, description="Seconds per call, after the client gate.")
    ollama_keep_alive: str = "30m"
    num_ctx: int = Field(default=4096, description="Context window sent on every call.")
    concurrency: int = Field(
        default=4,
        description="In-flight requests from this process. Match the server's OLLAMA_NUM_PARALLEL.",
    )
    item_concurrency: int = Field(
        default=2, description="Questions processed at once, so the server's slots stay busy."
    )
    max_retries: int = 2

    # --- Generation protocol (PLAN §6) -----------------------------------------
    k_samples: int = Field(default=10, description="Stochastic samples per question (Farquhar et al.).")
    sample_temperature: float = 1.0
    top_logprobs: int = Field(default=5, description="Alternatives recorded per greedy token.")
    max_tokens_short: int = 128
    max_tokens_choice: int = 768
    max_tokens_number: int = 1024
    max_tokens_expr: int = 1536
    logprob_decimals: int = 4
    seed_base: int = 1000

    # --- Objective 2 ------------------------------------------------------------
    intervention_samples: int = Field(default=5, description="Samples per intervention (PLAN §9.3).")

    # --- NLI (meaning equivalence, as Layer 1) ----------------------------------
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_batch: int = 32
    nli_device: str = Field(default="auto", description="auto | cpu | cuda | mps")

    # --- Data and outputs -------------------------------------------------------
    sets_dir: Path = PROJECT_ROOT / "data" / "uq_sets"
    results_dir: Path = PACKAGE_DIR / "results"
    fake_results_dir: Path = PACKAGE_DIR / "results_fake"
    split_seed: int = 7
    dev_fraction: float = 0.3

    # --- Statistics -------------------------------------------------------------
    bootstrap: int = 1000
    ece_bins: int = 15


settings = UQSettings()
