"""
Publication-Ready Plot Generator for the Defense-in-Depth Framework.

Generates Matplotlib charts suitable for direct thesis inclusion:
- Bar chart: Factual accuracy comparison
- Line chart: Latency vs. accuracy trade-off
- ROC curve: Probe classifier performance
- Box plot: Token expenditure distribution
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from app.config import settings
from app.models.schemas import BenchmarkRecord, MetricsReport

# Use non-interactive backend for server-side rendering
matplotlib.use("Agg")

# Publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette
COLORS = {
    "baseline": "#5C768D",   # SteelBlue
    "framework": "#1B365D",  # DeepNavy
    "layer1": "#4A90D9",
    "layer2": "#D4A574",
    "layer3": "#7CB342",
    "accent": "#E53935",
}


class PlotGenerator:
    """Generates publication-ready charts for thesis inclusion."""

    def __init__(self, output_dir: str | None = None) -> None:
        self._output_dir = Path(output_dir or settings.eval_output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("PlotGenerator initialized | output={}", self._output_dir)

    # -------------------------------------------------------------------------
    # 1. Factual Accuracy Comparison Bar Chart
    # -------------------------------------------------------------------------

    def plot_accuracy_comparison(
        self,
        reports: list[MetricsReport],
        filename: str = "accuracy_comparison",
    ) -> Path:
        """
        Bar chart comparing factual accuracy across datasets.

        Shows baseline vs. framework accuracy side by side.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        datasets = [r.dataset for r in reports]
        baseline_acc = [r.factual_accuracy_baseline * 100 for r in reports]
        framework_acc = [r.factual_accuracy_framework * 100 for r in reports]

        x = np.arange(len(datasets))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, baseline_acc, width,
            label="Zero-Shot Baseline",
            color=COLORS["baseline"],
            edgecolor="white",
        )
        bars2 = ax.bar(
            x + width / 2, framework_acc, width,
            label="Defense-in-Depth Framework",
            color=COLORS["framework"],
            edgecolor="white",
        )

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9,
            )
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_ylabel("Factual Accuracy (%)")
        ax.set_title("Factual Accuracy: Baseline vs. Defense-in-Depth Framework")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend(loc="lower right")
        ax.set_ylim(0, 105)

        filepath = self._save_figure(fig, filename)
        logger.info("Accuracy comparison chart saved to {}", filepath)
        return filepath

    # -------------------------------------------------------------------------
    # 2. Latency vs. Accuracy Trade-off Curve
    # -------------------------------------------------------------------------

    def plot_latency_accuracy_tradeoff(
        self,
        records: list[BenchmarkRecord],
        filename: str = "latency_accuracy_tradeoff",
    ) -> Path:
        """
        Scatter plot showing the latency-accuracy trade-off.

        Each point is a query; color indicates which layer processed it.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        # Group by layer
        groups = {}
        for r in records:
            layer = r.layers_invoked.value
            if layer not in groups:
                groups[layer] = {"latency": [], "correct": []}
            groups[layer]["latency"].append(r.framework_latency_ms)
            groups[layer]["correct"].append(0 if r.is_hallucinated_framework else 1)

        layer_colors = {
            "layer1_only": COLORS["layer1"],
            "layer1_layer2": COLORS["layer2"],
            "layer1_layer3": COLORS["layer3"],
            "full_pipeline": COLORS["accent"],
        }

        layer_labels = {
            "layer1_only": "Layer 1 (Fast Path)",
            "layer1_layer2": "Layer 1 + Layer 2 (NLI-RAG)",
            "layer1_layer3": "Layer 1 + Layer 3 (HalluClean)",
            "full_pipeline": "Full Pipeline",
        }

        for layer_name, data in groups.items():
            ax.scatter(
                data["latency"],
                data["correct"],
                c=layer_colors.get(layer_name, "#999999"),
                label=layer_labels.get(layer_name, layer_name),
                alpha=0.6,
                s=50,
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Correct (1) / Hallucinated (0)")
        ax.set_title("Latency vs. Accuracy Trade-off by Pipeline Layer")
        ax.legend(loc="lower right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Hallucinated", "Correct"])

        filepath = self._save_figure(fig, filename)
        logger.info("Latency-accuracy tradeoff chart saved to {}", filepath)
        return filepath

    # -------------------------------------------------------------------------
    # 3. ROC Curve
    # -------------------------------------------------------------------------

    def plot_roc_curve(
        self,
        scores: list[float],
        labels: list[int],
        filename: str = "roc_curve",
    ) -> Path:
        """
        ROC curve for the Semantic Entropy Probe classifier.

        Args:
            scores: Predicted hallucination probabilities.
            labels: Ground truth binary labels.
        """
        from sklearn.metrics import roc_curve, auc

        fig, ax = plt.subplots(figsize=(6, 6))

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr, tpr,
            color=COLORS["framework"],
            lw=2,
            label=f"SEP Probe (AUROC = {roc_auc:.4f})",
        )
        ax.plot(
            [0, 1], [0, 1],
            color="#999999",
            lw=1,
            linestyle="--",
            label="Random Baseline",
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Semantic Entropy Probe")
        ax.legend(loc="lower right")
        ax.set_aspect("equal")

        filepath = self._save_figure(fig, filename)
        logger.info("ROC curve saved to {}", filepath)
        return filepath

    # -------------------------------------------------------------------------
    # 4. Token Expenditure Box Plot
    # -------------------------------------------------------------------------

    def plot_token_expenditure(
        self,
        records: list[BenchmarkRecord],
        filename: str = "token_expenditure",
    ) -> Path:
        """
        Box plot comparing token expenditure between baseline and framework.
        """
        fig, ax = plt.subplots(figsize=(7, 5))

        baseline_tokens = [r.baseline_tokens for r in records if r.baseline_tokens > 0]
        framework_tokens = [r.framework_tokens for r in records if r.framework_tokens > 0]

        if not baseline_tokens or not framework_tokens:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return self._save_figure(fig, filename)

        bp = ax.boxplot(
            [baseline_tokens, framework_tokens],
            labels=["Zero-Shot Baseline", "Defense-in-Depth"],
            patch_artist=True,
            widths=0.5,
        )

        # Style boxes
        bp["boxes"][0].set_facecolor(COLORS["baseline"])
        bp["boxes"][1].set_facecolor(COLORS["framework"])
        for box in bp["boxes"]:
            box.set_alpha(0.7)

        for median in bp["medians"]:
            median.set_color("white")
            median.set_linewidth(2)

        ax.set_ylabel("Token Count per Query")
        ax.set_title("Token Expenditure Distribution")

        # Add mean markers
        means = [np.mean(baseline_tokens), np.mean(framework_tokens)]
        ax.scatter([1, 2], means, color=COLORS["accent"], marker="D", s=60, zorder=5, label="Mean")
        ax.legend()

        filepath = self._save_figure(fig, filename)
        logger.info("Token expenditure chart saved to {}", filepath)
        return filepath

    # -------------------------------------------------------------------------
    # 5. Layer Distribution Pie Chart
    # -------------------------------------------------------------------------

    def plot_layer_distribution(
        self,
        records: list[BenchmarkRecord],
        filename: str = "layer_distribution",
    ) -> Path:
        """
        Pie chart showing the distribution of queries across pipeline layers.
        """
        fig, ax = plt.subplots(figsize=(6, 6))

        layer_counts: dict[str, int] = {}
        for r in records:
            label = r.layers_invoked.value
            layer_counts[label] = layer_counts.get(label, 0) + 1

        labels = []
        sizes = []
        colors = []

        label_map = {
            "layer1_only": ("Layer 1 (Fast Path)", COLORS["layer1"]),
            "layer1_layer2": ("Layer 1 → 2 (NLI-RAG)", COLORS["layer2"]),
            "layer1_layer3": ("Layer 1 → 3 (HalluClean)", COLORS["layer3"]),
            "full_pipeline": ("Full Pipeline", COLORS["accent"]),
        }

        for layer_key, count in sorted(layer_counts.items()):
            name, color = label_map.get(layer_key, (layer_key, "#999999"))
            labels.append(f"{name}\n({count})")
            sizes.append(count)
            colors.append(color)

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.85,
        )

        # Style
        for autotext in autotexts:
            autotext.set_fontweight("bold")
            autotext.set_color("white")

        ax.set_title("Query Distribution Across Pipeline Layers")

        filepath = self._save_figure(fig, filename)
        logger.info("Layer distribution chart saved to {}", filepath)
        return filepath

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _save_figure(self, fig: plt.Figure, name: str) -> Path:
        """Save figure as both PDF and PNG."""
        pdf_path = self._output_dir / f"{name}.pdf"
        png_path = self._output_dir / f"{name}.png"

        fig.savefig(pdf_path, format="pdf")
        fig.savefig(png_path, format="png")
        plt.close(fig)

        return png_path
