"""
M.Tech Thesis Plot Generator — Publication-Ready Visualisations.

Reads the benchmark CSVs produced by benchmark_runner.py and generates
two 300-DPI charts:

  1. evaluation/plots/pareto_efficiency.png
     Scatter/line comparing Avg Latency vs. Accuracy across three configs.

  2. evaluation/plots/token_overhead.png
     Grouped bar chart comparing average token usage per query.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────────

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

# ── Paths ────────────────────────────────────────────────────────────────────

EVAL_DIR = Path(__file__).resolve().parent
PLOTS_DIR = EVAL_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_TRUTHFULQA = EVAL_DIR / "results_truthfulqa.csv"
CSV_GSM8K = EVAL_DIR / "results_gsm8k.csv"

# ── Matplotlib Config (publication-quality) ──────────────────────────────────

matplotlib.use("Agg")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Colours & Labels ─────────────────────────────────────────────────────────

CONFIG_COLOURS = {
    "baseline": "#5C768D",
    "naive_rag": "#D4A574",
    "full_framework": "#1B365D",
}

CONFIG_DISPLAY = {
    "baseline": "Raw LLM (Baseline)",
    "naive_rag": "Naive RAG",
    "full_framework": "Defense-in-Depth (Proposed)",
}

CONFIG_MARKERS = {
    "baseline": "o",
    "naive_rag": "s",
    "full_framework": "D",
}

DATASET_DISPLAY = {
    "truthfulqa": "TruthfulQA",
    "gsm8k": "GSM8K",
}


# ── Data Helpers ─────────────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    """Load and concatenate all available result CSVs."""
    frames = []
    for csv_path in (CSV_TRUTHFULQA, CSV_GSM8K):
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            logger.info("Loaded {} rows from {}", len(df), csv_path.name)
            frames.append(df)
        else:
            logger.warning("CSV not found: {}", csv_path)
    if not frames:
        logger.error("No result CSVs found.  Run benchmark_runner.py first.")
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(dataset, config) aggregates."""
    agg = (
        df.groupby(["dataset", "config"])
        .agg(
            accuracy=("is_correct", "mean"),
            avg_latency_ms=("wall_latency_ms", "mean"),
            avg_tokens=("tokens_generated", "mean"),
            n=("sample_id", "count"),
        )
        .reset_index()
    )
    agg["accuracy_pct"] = (agg["accuracy"] * 100).round(2)
    agg["avg_latency_ms"] = agg["avg_latency_ms"].round(1)
    agg["avg_tokens"] = agg["avg_tokens"].round(1)
    return agg


# ── Plot 1: Pareto Efficiency ────────────────────────────────────────────────


def plot_pareto(agg: pd.DataFrame) -> Path:
    """
    Scatter/line plot: Avg Latency (x) vs. Accuracy (y) for each config,
    with separate series per dataset.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    datasets_present = agg["dataset"].unique()

    for ds in datasets_present:
        ds_data = agg[agg["dataset"] == ds]
        ds_label = DATASET_DISPLAY.get(ds, ds)

        for _, row in ds_data.iterrows():
            cfg = row["config"]
            ax.scatter(
                row["avg_latency_ms"],
                row["accuracy_pct"],
                c=CONFIG_COLOURS.get(cfg, "#999"),
                marker=CONFIG_MARKERS.get(cfg, "o"),
                s=140,
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )
            # Annotate point
            ax.annotate(
                f" {CONFIG_DISPLAY.get(cfg, cfg)}",
                (row["avg_latency_ms"], row["accuracy_pct"]),
                fontsize=8,
                va="center",
            )

        # Draw connecting line across configs for this dataset
        order = ["baseline", "naive_rag", "full_framework"]
        ordered = ds_data.set_index("config").reindex(order).dropna()
        if len(ordered) > 1:
            ax.plot(
                ordered["avg_latency_ms"],
                ordered["accuracy_pct"],
                linestyle="--",
                linewidth=1.2,
                color="#888",
                alpha=0.6,
                label=f"_{ds_label}",  # hidden from legend
            )

    # Legend from colours (configs)
    from matplotlib.lines import Line2D
    handles = []
    for cfg_key in ["baseline", "naive_rag", "full_framework"]:
        handles.append(
            Line2D(
                [0], [0],
                marker=CONFIG_MARKERS[cfg_key],
                color="w",
                markerfacecolor=CONFIG_COLOURS[cfg_key],
                markersize=10,
                label=CONFIG_DISPLAY[cfg_key],
            )
        )
    # Dataset annotations
    for ds in datasets_present:
        handles.append(
            Line2D(
                [0], [0],
                marker="",
                linestyle="--",
                color="#888",
                label=f"  ({DATASET_DISPLAY.get(ds, ds)})",
            )
        )
    ax.legend(handles=handles, loc="best", framealpha=0.9)

    ax.set_xlabel("Average Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Pareto Efficiency: Latency vs. Accuracy")

    path = PLOTS_DIR / "pareto_efficiency.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved → {}", path)
    return path


# ── Plot 2: Token Overhead Bar Chart ─────────────────────────────────────────


def plot_token_overhead(agg: pd.DataFrame) -> Path:
    """
    Grouped bar chart: average tokens per query across configs,
    grouped by dataset.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    datasets = agg["dataset"].unique()
    configs = ["baseline", "naive_rag", "full_framework"]
    n_ds = len(datasets)
    n_cfg = len(configs)
    bar_width = 0.22
    x = np.arange(n_ds)

    for ci, cfg in enumerate(configs):
        vals = []
        for ds in datasets:
            row = agg[(agg["dataset"] == ds) & (agg["config"] == cfg)]
            vals.append(row["avg_tokens"].values[0] if len(row) else 0)
        bars = ax.bar(
            x + ci * bar_width,
            vals,
            bar_width,
            label=CONFIG_DISPLAY.get(cfg, cfg),
            color=CONFIG_COLOURS.get(cfg, "#999"),
            edgecolor="white",
            linewidth=0.6,
        )
        # Value labels
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(
        [DATASET_DISPLAY.get(d, d) for d in datasets]
    )
    ax.set_ylabel("Average Tokens per Query")
    ax.set_title("Token Overhead Comparison")
    ax.legend(loc="upper right", framealpha=0.9)

    path = PLOTS_DIR / "token_overhead.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved → {}", path)
    return path


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Loading benchmark data…")
    df = load_data()
    agg = aggregate(df)

    logger.info("Generating pareto_efficiency.png…")
    p1 = plot_pareto(agg)

    logger.info("Generating token_overhead.png…")
    p2 = plot_token_overhead(agg)

    print("\n" + "=" * 56)
    print("  Plot Generation Complete")
    print("=" * 56)
    print(f"  • {p1}")
    print(f"  • {p2}")
    print()


if __name__ == "__main__":
    main()
