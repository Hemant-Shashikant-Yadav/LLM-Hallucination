"""
Benchmark Runner for the Defense-in-Depth Framework.

Downloads HaluEval and TruthfulQA via HuggingFace datasets,
runs batch inference through the framework API, and compares:
- Zero-shot baseline (no framework)
- Layer 1 only
- Full pipeline

Outputs structured results as CSV and LaTeX tables.
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from app.config import settings
from app.models.schemas import BenchmarkRecord, PipelineLayer
from evaluation.metrics import generate_metrics_report, report_to_latex_table


# =============================================================================
# Dataset Loaders
# =============================================================================


def load_truthfulqa(max_samples: int | None = None) -> list[dict[str, Any]]:
    """
    Load TruthfulQA dataset from HuggingFace.

    Returns list of {"question": ..., "best_answer": ..., "category": ...}.
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

        samples = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            samples.append({
                "question_id": f"tqa_{i}",
                "question": item["question"],
                "best_answer": item.get("best_answer", ""),
                "category": item.get("category", "unknown"),
            })

        logger.info("Loaded {} TruthfulQA samples", len(samples))
        return samples

    except Exception as exc:
        logger.error("Failed to load TruthfulQA: {}", exc)
        return []


def load_halueval(max_samples: int | None = None) -> list[dict[str, Any]]:
    """
    Load HaluEval dataset from HuggingFace.

    Returns list of {"question": ..., "answer": ..., "hallucinated": ...}.
    """
    try:
        from datasets import load_dataset

        # HaluEval has multiple subsets; we use the QA subset
        dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

        samples = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            samples.append({
                "question_id": f"halu_{i}",
                "question": item.get("question", ""),
                "answer": item.get("hallucinated_answer", ""),
                "correct_answer": item.get("right_answer", ""),
                "hallucinated": True,
            })

        logger.info("Loaded {} HaluEval samples", len(samples))
        return samples

    except Exception as exc:
        logger.error("Failed to load HaluEval: {}", exc)
        return []


# =============================================================================
# Benchmark Engine
# =============================================================================


class BenchmarkRunner:
    """
    Batch benchmark runner that evaluates the framework
    against baseline LLM outputs.
    """

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        output_dir: str | None = None,
    ) -> None:
        self._api_base = api_base_url
        self._output_dir = Path(output_dir or settings.eval_output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.AsyncClient(timeout=1200.0)
        logger.info(
            "BenchmarkRunner initialized | api={} | output={}",
            self._api_base,
            self._output_dir,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    # -------------------------------------------------------------------------
    # Run Benchmarks
    # -------------------------------------------------------------------------

    async def run_benchmark(
        self,
        dataset_name: str,
        samples: list[dict[str, Any]],
    ) -> list[BenchmarkRecord]:
        """
        Run the full benchmark for a dataset.

        For each sample:
        1. Get zero-shot baseline from Layer 1 only
        2. Run through full pipeline
        3. Compare results

        Args:
            dataset_name: Name of the dataset.
            samples: List of sample dicts with 'question' key.

        Returns:
            List of BenchmarkRecord objects.
        """
        records: list[BenchmarkRecord] = []

        for i, sample in enumerate(samples):
            logger.info(
                "Benchmarking sample {}/{}: {:.50}...",
                i + 1,
                len(samples),
                sample["question"],
            )

            try:
                record = await self._benchmark_single(dataset_name, sample)
                records.append(record)
            except Exception as exc:
                logger.error("Benchmark failed for sample {}: {}", i, exc)
                continue

            # Small delay to avoid overwhelming the server
            await asyncio.sleep(0.5)

        logger.info(
            "Benchmark complete | dataset={} | samples={}",
            dataset_name,
            len(records),
        )

        return records

    async def _benchmark_single(
        self,
        dataset_name: str,
        sample: dict[str, Any],
    ) -> BenchmarkRecord:
        """Benchmark a single sample."""

        # 1. Baseline: Layer 1 only (zero-shot, no verification)
        baseline_start = time.perf_counter()
        baseline_response = await self._query_api(
            sample["question"], endpoint="/api/v1/query/layer1"
        )
        baseline_latency = (time.perf_counter() - baseline_start) * 1000

        # 2. Framework: Full pipeline
        framework_start = time.perf_counter()
        framework_response = await self._query_api(
            sample["question"], endpoint="/api/v1/query"
        )
        framework_latency = (time.perf_counter() - framework_start) * 1000

        # Determine hallucination status
        reference = sample.get("best_answer") or sample.get("correct_answer", "")

        return BenchmarkRecord(
            dataset=dataset_name,
            question_id=sample.get("question_id", "unknown"),
            question=sample["question"],
            reference_answer=reference,
            baseline_answer=baseline_response.get("answer", ""),
            framework_answer=framework_response.get("answer", ""),
            is_hallucinated_baseline=sample.get("hallucinated", False),
            is_hallucinated_framework=False,  # Will be evaluated by metrics
            layers_invoked=PipelineLayer(
                framework_response
                .get("pipeline_result", {})
                .get("layers_invoked", "layer1_only")
            ),
            baseline_latency_ms=baseline_latency,
            framework_latency_ms=framework_latency,
            baseline_tokens=baseline_response
                .get("pipeline_result", {})
                .get("tokens_generated", 0),
            framework_tokens=framework_response
                .get("pipeline_result", {})
                .get("tokens_generated", 0),
        )

    async def _query_api(
        self,
        prompt: str,
        endpoint: str = "/api/v1/query",
    ) -> dict[str, Any]:
        """Send a query to the framework API."""
        try:
            response = await self._client.post(
                f"{self._api_base}{endpoint}",
                json={"prompt": prompt, "include_audit": True},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error("API query failed: {}", exc)
            return {"answer": "[Error]", "pipeline_result": {}}

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    def save_records_csv(
        self,
        records: list[BenchmarkRecord],
        filename: str,
    ) -> Path:
        """Save benchmark records to CSV."""
        filepath = self._output_dir / filename

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset", "question_id", "question",
                    "reference_answer", "baseline_answer", "framework_answer",
                    "is_hallucinated_baseline", "is_hallucinated_framework",
                    "layers_invoked",
                    "baseline_latency_ms", "framework_latency_ms",
                    "baseline_tokens", "framework_tokens",
                ],
            )
            writer.writeheader()
            for r in records:
                writer.writerow(r.model_dump())

        logger.info("Saved {} records to {}", len(records), filepath)
        return filepath

    def save_latex_table(self, report, filename: str) -> Path:
        """Save LaTeX table to file."""
        filepath = self._output_dir / filename
        latex = report_to_latex_table(report)
        filepath.write_text(latex, encoding="utf-8")
        logger.info("Saved LaTeX table to {}", filepath)
        return filepath

    def save_report_json(self, report, filename: str) -> Path:
        """Save metrics report as JSON."""
        filepath = self._output_dir / filename
        filepath.write_text(
            json.dumps(report.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Saved metrics report to {}", filepath)
        return filepath


# =============================================================================
# CLI Entry Point
# =============================================================================


async def run_all_benchmarks(
    max_samples: int = 50,
    api_url: str = "http://localhost:8000",
) -> None:
    """Run all benchmarks and generate reports."""
    runner = BenchmarkRunner(api_base_url=api_url)

    try:
        # TruthfulQA
        logger.info("Loading TruthfulQA dataset...")
        tqa_samples = load_truthfulqa(max_samples)
        if tqa_samples:
            tqa_records = await runner.run_benchmark("TruthfulQA", tqa_samples)
            runner.save_records_csv(tqa_records, "truthfulqa_results.csv")

            if tqa_records:
                tqa_report = generate_metrics_report(tqa_records, "TruthfulQA")
                runner.save_report_json(tqa_report, "truthfulqa_metrics.json")
                runner.save_latex_table(tqa_report, "truthfulqa_table.tex")

        # HaluEval
        logger.info("Loading HaluEval dataset...")
        halu_samples = load_halueval(max_samples)
        if halu_samples:
            halu_records = await runner.run_benchmark("HaluEval", halu_samples)
            runner.save_records_csv(halu_records, "halueval_results.csv")

            if halu_records:
                halu_report = generate_metrics_report(halu_records, "HaluEval")
                runner.save_report_json(halu_report, "halueval_metrics.json")
                runner.save_latex_table(halu_report, "halueval_table.tex")

    finally:
        await runner.close()

    logger.info("All benchmarks complete!")


def main() -> None:
    """CLI entry point for the benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Defense-in-Depth Framework Benchmarks"
    )
    parser.add_argument(
        "--max-samples", type=int, default=50,
        help="Maximum samples per dataset (default: 50)"
    )
    parser.add_argument(
        "--api-url", type=str, default="http://localhost:8000",
        help="Framework API URL (default: http://localhost:8000)"
    )
    args = parser.parse_args()

    asyncio.run(run_all_benchmarks(args.max_samples, args.api_url))


if __name__ == "__main__":
    main()
