"""
Thesis Plot Generator — Extended Visualisations.

Run this script to generate multiple publication-ready graphs for your M.Tech thesis.
It reads the benchmark CSVs and produces graphs proving your objectives:
1. Accuracy Comparison (Objective 4)
2. Semantic Entropy Distribution (Objective 1)
3. Triage Routing Breakdown (Objective 2)
4. Latency Distribution (Objective 4)
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from loguru import logger

# ── Paths ────────────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).resolve().parent
PLOTS_DIR = EVAL_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Look for CSVs
def find_csvs():
    paths = []
    for f in ["results_truthfulqa.csv", "results_gsm8k.csv"]:
        # Check current dir, eval dir, and result-2 dir
        p1 = EVAL_DIR / f
        p2 = Path(f"result-2/{f}")
        p3 = Path(f)
        
        if p2.exists():
            paths.append(p2)
        elif p1.exists():
            paths.append(p1)
        elif p3.exists():
            paths.append(p3)
    return list(set(paths))

# ── Matplotlib Config (Publication-quality) ──────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

CONFIG_COLOURS = {
    "baseline": "#5C768D",
    "naive_rag": "#D4A574",
    "full_framework": "#1B365D",
}

CONFIG_LABELS = {
    "baseline": "Baseline LLM",
    "naive_rag": "Naive RAG",
    "full_framework": "Defense-in-Depth",
}

# ── Plot 1: Accuracy Comparison (Objective 4) ────────────────────────────────
def plot_accuracy(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    agg = df.groupby(["dataset", "config"])["is_correct"].mean().unstack() * 100
    
    # Reorder columns
    agg = agg[["baseline", "naive_rag", "full_framework"]]
    agg.rename(columns=CONFIG_LABELS, inplace=True)
    
    agg.plot(kind="bar", ax=ax, color=[CONFIG_COLOURS[c] for c in ["baseline", "naive_rag", "full_framework"]], edgecolor="white")
    
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Dataset")
    ax.set_title("Framework Accuracy vs. Baseline & Naive RAG")
    plt.xticks(rotation=0)
    ax.legend(title="Configuration")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    path = PLOTS_DIR / "thesis_accuracy_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")

# ── Plot 2: Semantic Entropy Distribution (Objective 1) ──────────────────────
def plot_entropy_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Filter out NaNs
    ent_df = df.dropna(subset=['semantic_entropy'])
    
    if ent_df.empty:
        logger.warning("No semantic entropy data found for plotting.")
        return
        
    sns.histplot(data=ent_df, x="semantic_entropy", hue="dataset", kde=True, bins=15, ax=ax, palette="Set2")
    
    # Add T_safe threshold line
    plt.axvline(x=0.35, color='red', linestyle='--', linewidth=2, label="T_safe Threshold (0.35)")
    
    ax.set_xlabel("Semantic Entropy (Uncertainty Score)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Model Uncertainty (Objective 1)")
    ax.legend()
    
    path = PLOTS_DIR / "thesis_entropy_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")

# ── Plot 3: Triage Routing Breakdown (Objective 2) ───────────────────────────
def plot_triage_routing(df: pd.DataFrame):
    # Only look at the full framework rows
    fw_df = df[df["config"] == "full_framework"].copy()
    if fw_df.empty:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    datasets = fw_df["dataset"].unique()
    colors = sns.color_palette("pastel")[0:3]
    
    for i, ds in enumerate(datasets):
        if i > 1: break
        ds_df = fw_df[fw_df["dataset"] == ds]
        counts = ds_df["triage_decision"].value_counts()
        
        axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops={'edgecolor': 'white'})
        axes[i].set_title(f"{ds.upper()} Routing Decision")
        
    fig.suptitle("Uncertainty Classification & Layer Routing (Objective 2)", fontsize=16)
    
    path = PLOTS_DIR / "thesis_routing_piecharts.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")

# ── Plot 4: Latency Distribution Boxplot (Objective 4) ───────────────────────
def plot_latency_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Convert ms to seconds
    df["latency_sec"] = df["wall_latency_ms"] / 1000.0
    
    sns.boxplot(x="dataset", y="latency_sec", hue="config", data=df, ax=ax, palette=CONFIG_COLOURS)
    
    ax.set_ylabel("Latency (Seconds)")
    ax.set_xlabel("Dataset")
    ax.set_title("Computational Cost: Latency Distribution (Objective 4)")
    ax.set_yscale("log")
    
    # Rename legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[CONFIG_LABELS.get(l, l) for l in labels], title="Configuration")
    
    path = PLOTS_DIR / "thesis_latency_boxplot.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")

# ── Plot 5: Token Generation Overhead ────────────────────────────────────────
def plot_token_overhead(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    agg = df.groupby(["dataset", "config"])["tokens_generated"].mean().unstack()
    
    # Reorder columns
    agg = agg[["baseline", "naive_rag", "full_framework"]]
    agg.rename(columns=CONFIG_LABELS, inplace=True)
    
    agg.plot(kind="bar", ax=ax, color=[CONFIG_COLOURS[c] for c in ["baseline", "naive_rag", "full_framework"]], edgecolor="white")
    
    ax.set_ylabel("Average Tokens Generated")
    ax.set_xlabel("Dataset")
    ax.set_title("Analysis: Token Generation Overhead")
    plt.xticks(rotation=0)
    ax.legend(title="Configuration")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    path = PLOTS_DIR / "thesis_token_overhead.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")

# ── Plot 6: Layer Invocation Flow ────────────────────────────────────────────
def plot_layer_invocations(df: pd.DataFrame):
    fw_df = df[df["config"] == "full_framework"].copy()
    if fw_df.empty: return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = fw_df.groupby(['dataset', 'layers_invoked']).size().unstack(fill_value=0)
    
    if not counts.empty:
        perc = counts.div(counts.sum(axis=1), axis=0) * 100
        perc.plot(kind='bar', stacked=True, ax=ax, colormap="viridis", edgecolor="white")
        
        ax.set_ylabel("Percentage of Queries (%)")
        ax.set_xlabel("Dataset")
        ax.set_title("Analysis: Dynamic Layer Invocation Flow")
        plt.xticks(rotation=0)
        
        handles, labels = ax.get_legend_handles_labels()
        pretty_labels = {"layer1_only": "Layer 1 (Safe)", "layer1_layer2": "Layer 2 (NLI-RAG)", "layer1_layer3": "Layer 3 (HalluClean)"}
        ax.legend(handles, [pretty_labels.get(l, l) for l in labels], title="Resolution Stage", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        path = PLOTS_DIR / "thesis_layer_invocations.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info(f"Saved: {path}")

# ── Plot 7: Error Type Classification ────────────────────────────────────────
def plot_error_types(df: pd.DataFrame):
    fw_df = df.dropna(subset=['classified_error_type']).copy()
    if fw_df.empty: return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=fw_df, x="dataset", hue="classified_error_type", ax=ax, palette="Set3", edgecolor="white")
    
    ax.set_ylabel("Number of Queries")
    ax.set_xlabel("Dataset")
    ax.set_title("Analysis: Identified Error Profiles per Dataset")
    ax.legend(title="Error Classification", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    path = PLOTS_DIR / "thesis_error_types.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")

# ── Plot 8: Latency vs. Token Scaling ────────────────────────────────────────
def plot_latency_vs_tokens(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    df["latency_sec"] = df["wall_latency_ms"] / 1000.0
    
    sns.scatterplot(data=df, x="tokens_generated", y="latency_sec", hue="config", style="dataset", 
                    palette=CONFIG_COLOURS, s=120, alpha=0.7, ax=ax)
    
    ax.set_ylabel("Latency (Seconds)")
    ax.set_xlabel("Tokens Generated")
    ax.set_title("System Scalability: Latency vs. Token Count")
    ax.set_yscale("log")
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [CONFIG_LABELS.get(l, l) for l in labels], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    path = PLOTS_DIR / "thesis_latency_scaling.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")

def main():
    csv_paths = find_csvs()
    if not csv_paths:
        logger.error("Could not find CSV result files. Please ensure you have run the benchmarks.")
        sys.exit(1)
        
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    
    logger.info(f"Loaded {len(df)} total evaluation rows.")
    
    # Core Objective Plots
    plot_accuracy(df)
    plot_entropy_distribution(df)
    plot_triage_routing(df)
    plot_latency_distribution(df)
    
    # Deep Analysis Plots
    plot_token_overhead(df)
    plot_layer_invocations(df)
    plot_error_types(df)
    plot_latency_vs_tokens(df)
    
    print("\n✅ All 8 thesis graphs generated successfully in the 'evaluation/plots/' directory!")

if __name__ == "__main__":
    main()
