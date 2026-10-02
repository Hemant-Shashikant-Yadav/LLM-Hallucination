"""
Evaluation Metrics for the Defense-in-Depth Framework.

Computes:
- Factual Accuracy (Precision, Recall, F1)
- AUROC for probe classification
- Latency Delta (wall-clock improvement)
- Token Expenditure (reduction ratio)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from app.models.schemas import BenchmarkRecord, MetricsReport, PipelineLayer


def compute_factual_accuracy(
    predictions: list[bool],
    ground_truth: list[bool],
) -> dict[str, float]:
    """
    Compute factual accuracy metrics.

    Args:
        predictions: Predicted hallucination labels (True = hallucinated).
        ground_truth: Ground truth labels.

    Returns:
        Dict with precision, recall, f1, and accuracy.
    """
    if not predictions or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average="binary", zero_division=0.0
    )
    accuracy = accuracy_score(ground_truth, predictions)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def compute_auroc(
    scores: list[float],
    labels: list[int],
) -> float:
    """
    Compute Area Under the ROC Curve for probe classification.

    Args:
        scores: Predicted hallucination probabilities.
        labels: Binary ground truth (1 = hallucinated, 0 = factual).

    Returns:
        AUROC score.
    """
    if not scores or not labels or len(set(labels)) < 2:
        return 0.5  # Random baseline

    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def compute_latency_delta(
    baseline_times_ms: list[float],
    framework_times_ms: list[float],
) -> dict[str, float]:
    """
    Compute latency improvement statistics.

    Args:
        baseline_times_ms: Baseline query latencies in ms.
        framework_times_ms: Framework query latencies in ms.

    Returns:
        Dict with mean, median, and percentage change.
    """
    if not baseline_times_ms or not framework_times_ms:
        return {"mean_baseline": 0.0, "mean_framework": 0.0, "delta_pct": 0.0}

    baseline = np.array(baseline_times_ms)
    framework = np.array(framework_times_ms)

    mean_b = float(np.mean(baseline))
    mean_f = float(np.mean(framework))

    delta_pct = ((mean_f - mean_b) / mean_b * 100) if mean_b > 0 else 0.0

    return {
        "mean_baseline": mean_b,
        "mean_framework": mean_f,
        "median_baseline": float(np.median(baseline)),
        "median_framework": float(np.median(framework)),
        "delta_pct": delta_pct,
    }


def compute_token_expenditure(
    baseline_tokens: list[int],
    framework_tokens: list[int],
) -> dict[str, float]:
    """
    Compute token expenditure reduction.

    Args:
        baseline_tokens: Token counts per query for baseline.
        framework_tokens: Token counts per query for framework.

    Returns:
        Dict with totals and percentage reduction.
    """
    total_b = sum(baseline_tokens)
    total_f = sum(framework_tokens)

    reduction_pct = (
        ((total_b - total_f) / total_b * 100) if total_b > 0 else 0.0
    )

    return {
        "total_baseline": total_b,
        "total_framework": total_f,
        "reduction_pct": reduction_pct,
        "mean_baseline": float(np.mean(baseline_tokens)) if baseline_tokens else 0.0,
        "mean_framework": float(np.mean(framework_tokens)) if framework_tokens else 0.0,
    }


def generate_metrics_report(
    records: list[BenchmarkRecord],
    dataset_name: str,
) -> MetricsReport:
    """
    Generate a comprehensive metrics report from benchmark records.

    Args:
        records: List of benchmark evaluation records.
        dataset_name: Name of the dataset.

    Returns:
        MetricsReport with all computed metrics.
    """
    if not records:
        raise ValueError("No benchmark records to generate report from")

    # Factual accuracy
    baseline_correct = [not r.is_hallucinated_baseline for r in records]
    framework_correct = [not r.is_hallucinated_framework for r in records]

    acc_baseline = sum(baseline_correct) / len(baseline_correct)
    acc_framework = sum(framework_correct) / len(framework_correct)

    # AUROC (using framework's hallucination probability if available)
    auroc = 0.5  # Default
    try:
        baseline_labels = [int(r.is_hallucinated_baseline) for r in records]
        framework_labels = [int(r.is_hallucinated_framework) for r in records]
        if len(set(baseline_labels)) >= 2:
            # Use framework correctness as a proxy score
            scores = [1.0 if not r.is_hallucinated_framework else 0.0 for r in records]
            auroc = compute_auroc(scores, baseline_labels)
    except Exception:
        pass

    # Latency
    latency = compute_latency_delta(
        [r.baseline_latency_ms for r in records],
        [r.framework_latency_ms for r in records],
    )

    # Tokens
    tokens = compute_token_expenditure(
        [r.baseline_tokens for r in records],
        [r.framework_tokens for r in records],
    )

    # Layer 1 bypass rate
    layer1_only = sum(
        1 for r in records
        if r.layers_invoked == PipelineLayer.LAYER1_ONLY
    )
    bypass_rate = layer1_only / len(records) if records else 0.0

    report = MetricsReport(
        dataset=dataset_name,
        total_samples=len(records),
        factual_accuracy_baseline=acc_baseline,
        factual_accuracy_framework=acc_framework,
        accuracy_improvement=acc_framework - acc_baseline,
        auroc=auroc,
        mean_latency_baseline_ms=latency["mean_baseline"],
        mean_latency_framework_ms=latency["mean_framework"],
        latency_delta_pct=latency["delta_pct"],
        total_tokens_baseline=tokens["total_baseline"],
        total_tokens_framework=tokens["total_framework"],
        token_reduction_pct=tokens["reduction_pct"],
        layer1_bypass_rate=bypass_rate,
    )

    logger.info(
        "Metrics report generated for {} | samples={} | "
        "acc_baseline={:.2%} | acc_framework={:.2%} | Δacc={:+.2%}",
        dataset_name,
        len(records),
        acc_baseline,
        acc_framework,
        acc_framework - acc_baseline,
    )

    return report


def report_to_latex_table(report: MetricsReport) -> str:
    """
    Generate a LaTeX table from a MetricsReport.

    Produces a publication-ready booktabs table for thesis inclusion.
    """
    return rf"""
\begin{{table}}[htbp]
\centering
\caption{{Evaluation Results on {report.dataset} (N={report.total_samples})}}
\label{{tab:{report.dataset.lower().replace(' ', '_')}_results}}
\begin{{tabular}}{{@{{}}lcc@{{}}}}
\toprule
\textbf{{Metric}} & \textbf{{Baseline}} & \textbf{{Framework}} \\
\midrule
Factual Accuracy & {report.factual_accuracy_baseline:.2%} & {report.factual_accuracy_framework:.2%} \\
Accuracy Improvement & \multicolumn{{2}}{{c}}{{{report.accuracy_improvement:+.2%}}} \\
AUROC & --- & {report.auroc:.4f} \\
\midrule
Mean Latency (ms) & {report.mean_latency_baseline_ms:.1f} & {report.mean_latency_framework_ms:.1f} \\
Latency Delta & \multicolumn{{2}}{{c}}{{{report.latency_delta_pct:+.1f}\%}} \\
\midrule
Total Tokens & {report.total_tokens_baseline:,} & {report.total_tokens_framework:,} \\
Token Reduction & \multicolumn{{2}}{{c}}{{{report.token_reduction_pct:.1f}\%}} \\
\midrule
Layer 1 Bypass Rate & \multicolumn{{2}}{{c}}{{{report.layer1_bypass_rate:.2%}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
