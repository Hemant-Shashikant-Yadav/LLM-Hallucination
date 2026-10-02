"""
M.Tech Thesis Benchmark Runner — Three-Configuration Evaluation.

Evaluates the Defense-in-Depth framework across three configurations:
  1. Baseline (Raw LLM):  POST /api/v1/query/layer1
  2. Naive RAG:           POST /api/v1/query/layer2
  3. Full Framework:      POST /api/v1/query

Datasets:
  - TruthfulQA (30 validation samples, generation split)
  - GSM8K      (30 test samples, main split)

Produces:
  - evaluation/results_truthfulqa.csv
  - evaluation/results_gsm8k.csv
  - LaTeX comparison table printed to stdout
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from loguru import logger

# ── Logging Setup ────────────────────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)

# ── Constants ────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"
N_SAMPLES = 30
REQUEST_TIMEOUT = 1200.0  # 20 minutes — generous for cold Ollama models
OUTPUT_DIR = Path(__file__).resolve().parent
CONCURRENCY = 1  # Sequential to avoid overwhelming local Ollama


# ── Dataset Loaders ──────────────────────────────────────────────────────────


def load_truthfulqa(n: int = N_SAMPLES) -> list[dict[str, Any]]:
    """Load *n* validation samples from truthfulqa/truthful_qa (generation)."""
    from datasets import load_dataset

    logger.info("Loading TruthfulQA (generation split, n={})…", n)
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    samples = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        # Collect ALL correct answers for flexible matching
        correct_answers: list[str] = []
        if row.get("best_answer"):
            correct_answers.append(row["best_answer"])
        if row.get("correct_answers"):
            correct_answers.extend(row["correct_answers"])
        samples.append({
            "id": f"tqa_{i}",
            "prompt": row["question"],
            "correct_answers": correct_answers,
            "category": row.get("category", ""),
        })
    logger.info("Loaded {} TruthfulQA samples", len(samples))
    return samples


def load_gsm8k(n: int = N_SAMPLES) -> list[dict[str, Any]]:
    """Load *n* test samples from gsm8k (main split)."""
    from datasets import load_dataset

    logger.info("Loading GSM8K (main split, n={})…", n)
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception:
        ds = load_dataset("gsm8k", "main", split="test")
    samples = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        # Extract the numerical answer from the #### delimiter
        answer_text = row.get("answer", "")
        numerical = ""
        if "####" in answer_text:
            numerical = answer_text.split("####")[-1].strip()
            # Normalise: remove commas, dollar signs, spaces
            numerical = numerical.replace(",", "").replace("$", "").strip()
        samples.append({
            "id": f"gsm_{i}",
            "prompt": row["question"],
            "correct_answers": [numerical] if numerical else [],
            "numerical_answer": numerical,
        })
    logger.info("Loaded {} GSM8K samples", len(samples))
    return samples


# ── Accuracy Evaluators ──────────────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def eval_truthfulqa(answer: str, correct_answers: list[str]) -> bool:
    """
    TruthfulQA accuracy: substring overlap.

    Returns True if any correct answer appears as a substring (case-insensitive)
    in the model's response.
    """
    if not correct_answers:
        return False
    ans_lower = answer.lower()
    for ca in correct_answers:
        if ca and ca.lower().strip() in ans_lower:
            return True
    return False


def eval_gsm8k(answer: str, numerical_answer: str) -> bool:
    """
    GSM8K accuracy: check if the numerical ground truth appears in the
    generated answer (after stripping commas/dollars).
    """
    if not numerical_answer:
        return False
    # Normalise the answer text
    normalised = answer.replace(",", "").replace("$", "")
    return numerical_answer in normalised


# ── API Client ───────────────────────────────────────────────────────────────


async def query_api(
    client: httpx.AsyncClient,
    prompt: str,
    endpoint: str,
) -> dict[str, Any]:
    """
    POST a query to the framework and return the JSON response.

    Returns a safe fallback dict on any error.
    """
    body = {
        "prompt": prompt,
        "model": "llama3.1:8b",
        "include_audit": True,
    }
    try:
        resp = await client.post(f"{API_BASE}{endpoint}", json=body)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("API {} failed: {}", endpoint, exc)
        return {"answer": "[ERROR]", "pipeline_result": None}


def _extract_fields(resp: dict[str, Any]) -> dict[str, Any]:
    """Extract auditing fields from a framework response."""
    pr = resp.get("pipeline_result") or {}
    triage = pr.get("triage_result") or {}
    ec = triage.get("error_classification") or {}
    return {
        "answer": resp.get("answer", ""),
        "total_latency_ms": pr.get("total_latency_ms", 0.0),
        "tokens_generated": pr.get("tokens_generated", 0),
        "triage_decision": triage.get("decision", ""),
        "classified_error_type": ec.get("error_type", ""),
        "semantic_entropy": triage.get("semantic_entropy", 0.0),
        "layers_invoked": pr.get("layers_invoked", ""),
    }


# ── Benchmark Core ───────────────────────────────────────────────────────────


async def benchmark_dataset(
    dataset_name: str,
    samples: list[dict[str, Any]],
    evaluator,
) -> pd.DataFrame:
    """
    Run all three configurations for every sample and return a DataFrame.

    Configurations:
      1. baseline   → /api/v1/query/layer1
      2. naive_rag  → /api/v1/query/layer2
      3. full_framework → /api/v1/query
    """
    configs = [
        ("baseline", "/api/v1/query/layer1"),
        ("naive_rag", "/api/v1/query/layer2"),
        ("full_framework", "/api/v1/query"),
    ]

    rows: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for idx, sample in enumerate(samples):
            prompt = sample["prompt"]
            correct = sample["correct_answers"]
            logger.info(
                "[{}/{}] {}: {:.60}…",
                idx + 1,
                len(samples),
                dataset_name,
                prompt,
            )

            for config_name, endpoint in configs:
                t0 = time.perf_counter()
                resp = await query_api(client, prompt, endpoint)
                wall_ms = (time.perf_counter() - t0) * 1000

                fields = _extract_fields(resp)
                answer_text = fields["answer"]

                # Evaluate accuracy
                if dataset_name == "gsm8k":
                    correct_flag = evaluator(
                        answer_text, sample.get("numerical_answer", "")
                    )
                else:
                    correct_flag = evaluator(answer_text, correct)

                rows.append({
                    "dataset": dataset_name,
                    "sample_id": sample["id"],
                    "prompt": prompt[:200],
                    "config": config_name,
                    "answer": answer_text[:500],
                    "is_correct": correct_flag,
                    "total_latency_ms": fields["total_latency_ms"],
                    "wall_latency_ms": round(wall_ms, 2),
                    "tokens_generated": fields["tokens_generated"],
                    "triage_decision": fields["triage_decision"],
                    "classified_error_type": fields["classified_error_type"],
                    "semantic_entropy": fields["semantic_entropy"],
                    "layers_invoked": fields["layers_invoked"],
                    "reference_answer": (
                        correct[0][:200] if correct else ""
                    ),
                })

            # Brief pause between samples
            await asyncio.sleep(0.3)

    return pd.DataFrame(rows)


# ── Summary Table Builder ────────────────────────────────────────────────────


def build_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Aggregate per-config summary from the raw results DataFrame.

    Columns: Config | Accuracy(%) | Avg Latency(ms) | Avg Tokens
    """
    agg = (
        df.groupby("config")
        .agg(
            accuracy=("is_correct", "mean"),
            avg_latency_ms=("wall_latency_ms", "mean"),
            avg_tokens=("tokens_generated", "mean"),
            n_samples=("sample_id", "count"),
        )
        .reset_index()
    )
    agg["accuracy_pct"] = (agg["accuracy"] * 100).round(2)
    agg["avg_latency_ms"] = agg["avg_latency_ms"].round(1)
    agg["avg_tokens"] = agg["avg_tokens"].round(1)
    agg["dataset"] = dataset_name

    # Ensure ordering: baseline → naive_rag → full_framework
    order = {"baseline": 0, "naive_rag": 1, "full_framework": 2}
    agg["_order"] = agg["config"].map(order)
    agg = agg.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    return agg


def print_latex(summary: pd.DataFrame, dataset_name: str) -> str:
    """Print a LaTeX booktabs table to stdout and return the string."""
    display_names = {
        "baseline": "Raw LLM (Baseline)",
        "naive_rag": "Naive RAG (Layer 2)",
        "full_framework": "Defense-in-Depth (Proposed)",
    }
    tbl = summary[["config", "accuracy_pct", "avg_latency_ms", "avg_tokens"]].copy()
    tbl["config"] = tbl["config"].map(display_names)
    tbl.columns = ["Configuration", "Accuracy (%)", "Avg Latency (ms)", "Avg Tokens"]

    latex = tbl.to_latex(
        index=False,
        caption=f"Benchmark Results on {dataset_name} (N={int(summary['n_samples'].iloc[0])})",
        label=f"tab:{dataset_name.lower().replace(' ', '_')}_benchmark",
        column_format="lccc",
        escape=False,
    )
    print("\n" + "=" * 72)
    print(f"  LaTeX Table — {dataset_name}")
    print("=" * 72)
    print(latex)
    return latex


# ── Main Entry ───────────────────────────────────────────────────────────────


async def run() -> None:
    """Full benchmark pipeline: load → query → evaluate → report."""

    # 0. Health check
    logger.info("Pre-flight: checking API health…")
    try:
        async with httpx.AsyncClient(timeout=10) as hc:
            r = await hc.get(f"{API_BASE}/api/v1/health")
            r.raise_for_status()
            logger.info("API healthy: {}", r.json().get("status"))
    except Exception as exc:
        logger.error(
            "API not reachable at {} — is the server running? Error: {}",
            API_BASE,
            exc,
        )
        sys.exit(1)

    all_summaries: list[pd.DataFrame] = []

    # ── TruthfulQA ───────────────────────────────────────────────────────
    tqa_csv_path = OUTPUT_DIR / "results_truthfulqa.csv"
    if tqa_csv_path.exists() and tqa_csv_path.stat().st_size > 1000:
        logger.info("TruthfulQA results already exist at {} — loading existing data", tqa_csv_path)
        df_tqa = pd.read_csv(tqa_csv_path)
        summary_tqa = build_summary(df_tqa, "TruthfulQA")
        all_summaries.append(summary_tqa)
        print_latex(summary_tqa, "TruthfulQA")
    else:
        try:
            tqa_samples = load_truthfulqa(N_SAMPLES)
        except Exception as exc:
            logger.error("Failed to load TruthfulQA: {}", exc)
            tqa_samples = []

        if tqa_samples:
            logger.info("▶ Starting TruthfulQA benchmark ({} samples)…", len(tqa_samples))
            df_tqa = await benchmark_dataset("truthfulqa", tqa_samples, eval_truthfulqa)
            df_tqa.to_csv(tqa_csv_path, index=False)
            logger.info("Saved → {}", tqa_csv_path)

            summary_tqa = build_summary(df_tqa, "TruthfulQA")
            all_summaries.append(summary_tqa)
            print_latex(summary_tqa, "TruthfulQA")

    # ── GSM8K ────────────────────────────────────────────────────────────
    try:
        gsm_samples = load_gsm8k(N_SAMPLES)
    except Exception as exc:
        logger.error("Failed to load GSM8K: {}", exc)
        gsm_samples = []

    if gsm_samples:
        logger.info("▶ Starting GSM8K benchmark ({} samples)…", len(gsm_samples))
        df_gsm = await benchmark_dataset("gsm8k", gsm_samples, eval_gsm8k)
        csv_path = OUTPUT_DIR / "results_gsm8k.csv"
        df_gsm.to_csv(csv_path, index=False)
        logger.info("Saved → {}", csv_path)

        summary_gsm = build_summary(df_gsm, "GSM8K")
        all_summaries.append(summary_gsm)
        print_latex(summary_gsm, "GSM8K")

    # ── Combined Table ───────────────────────────────────────────────────
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        print("\n" + "=" * 72)
        print("  Combined Summary")
        print("=" * 72)
        print(
            combined[
                ["dataset", "config", "accuracy_pct", "avg_latency_ms", "avg_tokens"]
            ].to_string(index=False)
        )

    logger.info("✅ All benchmarks complete.")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
