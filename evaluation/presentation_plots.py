"""
Presentation figures for the Phase 1 review deck.

Reads the scored CSVs written by `evaluation.benchmark_runner` and writes
300-DPI PNGs:

  fig_accuracy.png          accuracy per configuration with 95% bootstrap CIs
  fig_routing.png           Layer 1 routing distribution + per-route accuracy
  fig_cost.png              latency and token cost per configuration
  fig_entropy.png           semantic entropy of correct vs. wrong drafts (+ AUROC)
  fig_metric_audit.png      legacy (lenient) vs. strict metrics on the same answers
  fig_architecture.png      Phase 1 pipeline diagram with the models used

Usage:
  python -m evaluation.presentation_plots --results evaluation/results/phase1_llama2_all_configs --out evaluation/results/phase1_llama2_all_configs/figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Validated categorical palette (fixed order, see dataviz validator run)
COLORS = {
    "baseline": "#D97706",
    "naive_rag": "#0D9488",
    "full_framework": "#4F46E5",
    "reference_verifier": "#BE185D",
}
LABELS = {
    "baseline": "Baseline\n(llama2:7b)",
    "naive_rag": "Naive RAG",
    "full_framework": "Defense-in-Depth\n(proposed)",
    "reference_verifier": "Reference",
}
ORDER = ["baseline", "naive_rag", "full_framework", "reference_verifier"]
# Configurations shown in the presentation figures (baseline vs. framework)
SHOWN = {"baseline", "full_framework"}
INK, MUTED, GRID = "#0F172A", "#475569", "#E2E8F0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


def _bootstrap(values: np.ndarray, n: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _configs(df: pd.DataFrame) -> list[str]:
    return [c for c in ORDER if c in df["config"].unique()]


# ── Figures ──────────────────────────────────────────────────────────────────


def fig_accuracy(tqa: pd.DataFrame | None, gsm: pd.DataFrame | None, out: Path) -> None:
    panels = []
    if gsm is not None:
        panels.append(("GSM8K — exact final answer", gsm, "is_correct"))
    if tqa is not None:
        panels.append(("TruthfulQA — % truthful (judge)", tqa, "judge_truthful"))
        panels.append(("TruthfulQA — % truthful & informative", tqa, "truthful_informative"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 5.6), squeeze=False)
    for ax, (title, df, metric) in zip(axes[0], panels):
        configs = _configs(df)
        means, errs = [], []
        for c in configs:
            v = df[df["config"] == c][metric].astype(bool).to_numpy().astype(float)
            lo, hi = _bootstrap(v)
            means.append(100 * v.mean())
            errs.append([100 * (v.mean() - lo), 100 * (hi - v.mean())])
        x = np.arange(len(configs))
        ax.bar(x, means, width=0.62, color=[COLORS[c] for c in configs],
               edgecolor="white", linewidth=2)
        ax.errorbar(x, means, yerr=np.array(errs).T, fmt="none", ecolor=INK,
                    elinewidth=1.2, capsize=4)
        for xi, m, e in zip(x, means, errs):
            ax.text(xi, m + e[1] + 2, f"{m:.0f}%", ha="center", va="bottom",
                    color=INK, fontsize=11, fontweight="bold")
        ax.set_xticks(x, [LABELS[c] for c in configs], fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title, loc="left")
    fig.text(0.01, -0.02, "Error bars: 95% bootstrap confidence interval. Same questions for every configuration.",
             color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_accuracy.png")
    plt.close(fig)


def fig_routing(frames: dict[str, pd.DataFrame], out: Path) -> None:
    decisions = ["safe", "knowledge_uncertain", "reasoning_uncertain"]
    dec_labels = {"safe": "Safe (fast path)", "knowledge_uncertain": "Knowledge → Layer 2",
                  "reasoning_uncertain": "Reasoning → Layer 3"}
    dec_colors = {"safe": "#94A3B8", "knowledge_uncertain": "#0D9488", "reasoning_uncertain": "#4F46E5"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    # Left: routing distribution
    ax = axes[0]
    names = list(frames)
    left = np.zeros(len(names))
    for d in decisions:
        vals = []
        for n in names:
            fw = frames[n][frames[n]["config"] == "full_framework"]
            vals.append(100 * (fw["triage_decision"] == d).mean())
        vals = np.array(vals)
        ax.barh(names, vals, left=left, color=dec_colors[d], edgecolor="white",
                linewidth=2, label=dec_labels[d], height=0.55)
        for i, v in enumerate(vals):
            if v >= 8:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
        left += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of queries (%)")
    ax.set_title("Layer 1 routing decisions", loc="left")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5)

    # Right: accuracy per route, baseline draft vs framework answer
    ax = axes[1]
    labels, base_acc, fw_acc = [], [], []
    for n, df in frames.items():
        fw = df[df["config"] == "full_framework"].set_index("sample_id")
        base = df[df["config"] == "baseline"].set_index("sample_id")
        metric = "is_correct"
        for d in decisions:
            ids = fw.index[fw["triage_decision"] == d]
            if len(ids) == 0:
                continue
            labels.append(f"{n}\n{dec_labels[d].split(' ')[0]}\n(n={len(ids)})")
            base_acc.append(100 * base.loc[ids, metric].astype(bool).mean())
            fw_acc.append(100 * fw.loc[ids, metric].astype(bool).mean())
    x = np.arange(len(labels))
    ax.bar(x - 0.18, base_acc, 0.34, color=COLORS["baseline"], edgecolor="white", linewidth=2, label="Baseline draft")
    ax.bar(x + 0.18, fw_acc, 0.34, color=COLORS["full_framework"], edgecolor="white", linewidth=2, label="After framework")
    for xi, b, f in zip(x, base_acc, fw_acc):
        ax.text(xi - 0.18, b + 1.5, f"{b:.0f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi + 0.18, f + 1.5, f"{f:.0f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy per route", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_routing.png")
    plt.close(fig)


def fig_cost(frames: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (col, scale, title, unit) in zip(axes, [
        ("wall_latency_ms", 1000.0, "Mean latency per query", "seconds"),
        ("tokens_generated", 1.0, "Mean tokens per query (all LLM calls)", "tokens"),
    ]):
        names = list(frames)
        configs = [c for c in ORDER if any(c in f["config"].unique() for f in frames.values())]
        width = 0.8 / len(configs)
        for k, c in enumerate(configs):
            vals = [frames[n][frames[n]["config"] == c][col].mean() / scale for n in names]
            xs = np.arange(len(names)) + (k - (len(configs) - 1) / 2) * width
            ax.bar(xs, vals, width * 0.92, color=COLORS[c], edgecolor="white", linewidth=2,
                   label=LABELS[c].replace("\n", " "))
            for xv, v in zip(xs, vals):
                ax.text(xv, v * 1.02, f"{v:,.0f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(np.arange(len(names)), names)
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left")
    axes[0].legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_cost.png")
    plt.close(fig)


def fig_entropy(frames: dict[str, pd.DataFrame], out: Path) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    aurocs: dict[str, float] = {}
    fig, axes = plt.subplots(1, len(frames), figsize=(5.5 * len(frames), 4.2), squeeze=False)
    for ax, (n, df) in zip(axes[0], frames.items()):
        fw = df[df["config"] == "full_framework"].set_index("sample_id")
        base = df[df["config"] == "baseline"].set_index("sample_id")
        se = fw["semantic_entropy"].astype(float)
        wrong = ~base.loc[fw.index, "is_correct"].astype(bool)
        bins = np.linspace(0, 1, 11)
        ax.hist([se[~wrong], se[wrong]], bins=bins, color=[COLORS["naive_rag"], "#BE185D"],
                edgecolor="white", linewidth=1.5, label=["Draft correct", "Draft wrong (hallucination)"])
        thr = fw["semantic_entropy"].notna()
        if wrong.nunique() == 2:
            aurocs[n] = float(roc_auc_score(wrong.astype(int), se))
        ax.set_xlabel("Semantic entropy  $H_{sem}$ (normalised)")
        ax.set_ylabel("Number of queries")
        ax.set_title(f"{n}: AUROC = {aurocs.get(n, float('nan')):.2f}", loc="left")
        ax.legend(fontsize=9)
    fig.text(0.01, -0.03, "AUROC: how well H_sem separates wrong drafts from correct drafts (0.5 = chance).",
             color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_entropy.png")
    plt.close(fig)
    return aurocs


def _legacy_tqa_overlap(answer: str, correct: list[str]) -> bool:
    ans = answer.lower()
    toks = set(re.findall(r"\b\w+\b", ans))
    for ca in correct:
        ca = ca.lower().strip()
        if ca and ca in ans:
            return True
        ct = set(re.findall(r"\b\w+\b", ca))
        if ct and len(ct & toks) / len(ct) >= 0.5:
            return True
    return False


def fig_metric_audit(tqa: pd.DataFrame | None, gsm: pd.DataFrame | None, out: Path) -> None:
    rows = []
    if tqa is not None:
        b = tqa[tqa["config"] == "baseline"]
        legacy = np.mean([_legacy_tqa_overlap(str(a), json.loads(c))
                          for a, c in zip(b["answer"], b["correct_answers"])])
        rows.append(("TruthfulQA baseline", 100 * legacy, 100 * b["judge_truthful"].astype(bool).mean(),
                     "token overlap ≥ 50%", "independent LLM judge"))
    if gsm is not None:
        b = gsm[gsm["config"] == "baseline"]
        rows.append(("GSM8K baseline", 100 * b["is_correct_legacy"].astype(bool).mean(),
                     100 * b["is_correct"].astype(bool).mean(), "number appears anywhere", "exact final answer"))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(rows))
    ax.barh(y + 0.18, [r[1] for r in rows], 0.34, color="#94A3B8", edgecolor="white", label="Old metric (lenient)")
    ax.barh(y - 0.18, [r[2] for r in rows], 0.34, color=COLORS["full_framework"], edgecolor="white", label="New metric (strict)")
    for yi, r in zip(y, rows):
        ax.text(r[1] + 1, yi + 0.18, f"{r[1]:.0f}%  ({r[3]})", va="center", fontsize=9, color=INK)
        ax.text(r[2] + 1, yi - 0.18, f"{r[2]:.0f}%  ({r[4]})", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Measured accuracy of the SAME baseline answers (%)")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_title("Metric audit: the old metrics over-counted correct answers", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_metric_audit.png")
    plt.close(fig)


def fig_architecture(out: Path, setup: str = "asym") -> None:
    verifier = "llama3.1:8b" if setup == "asym" else "llama2:7b"
    layer3 = ("plan → execute → judge → refine" if setup == "asym"
              else "plan → programs → verified vote")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color, text_color="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=color, edgecolor="white", linewidth=2))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", color=text_color,
                fontsize=11, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", color=text_color, fontsize=8.5)

    def arrow(x1, y1, x2, y2, label=""):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     color=MUTED, linewidth=1.4))
        if label:
            ax.text((x1 + x2) / 2 + 0.08, (y1 + y2) / 2, label, fontsize=8.5, color=MUTED, va="center")

    box(3.5, 5.6, 3.0, 0.8, "User query → Draft", "generator: llama2:7b (greedy)", "#1F2A44")
    box(2.6, 4.15, 4.8, 0.95, "Layer 1: Semantic entropy triage",
        "K = 4 extra samples · cluster by meaning\n(NLI for facts, final number for maths)", "#3657C8")
    box(0.2, 2.35, 2.6, 0.95, "Fast path", "H_sem < T_safe and draft\nin majority → return draft", "#94A3B8")
    box(3.4, 2.35, 3.0, 0.95, "Layer 2: NLI-RAG", "claims → evidence → DeBERTa NLI\n→ rewrite + CoVe check", "#0D9488")
    box(6.8, 2.35, 3.0, 0.95, "Layer 3: HalluClean", f"{layer3}\n(verifier: {verifier})", "#4F46E5")
    box(3.0, 0.55, 4.0, 0.9, "Verified response + audit trail", "tokens, latency, route, entropy, NLI verdicts", "#1F2A44")

    arrow(5.0, 5.6, 5.0, 5.1)
    arrow(3.2, 4.15, 1.6, 3.3)
    arrow(5.0, 4.15, 4.9, 3.3)
    arrow(6.8, 4.15, 8.2, 3.3)
    ax.text(1.55, 3.82, "low entropy", fontsize=9, color=MUTED, ha="center")
    ax.text(5.08, 3.72, "factual,\nuncertain", fontsize=9, color=MUTED, ha="left")
    ax.text(8.45, 3.82, "quantitative,\nuncertain", fontsize=9, color=MUTED, ha="center")
    arrow(1.6, 2.35, 3.8, 1.45)
    arrow(4.9, 2.35, 5.0, 1.45)
    arrow(8.2, 2.35, 6.2, 1.45)
    ax.text(5.0, 0.15, "Layer 3 keeps the draft when its verified final answer matches; otherwise the verified answer replaces it.",
            ha="center", fontsize=8.5, color=MUTED, style="italic")
    fig.savefig(out / "fig_architecture.png")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="evaluation/results/phase1_llama2_all_configs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--setup", choices=["asym", "llama2_only"], default="asym")
    args = ap.parse_args()

    res = Path(args.results)
    out = Path(args.out) if args.out else res / "figures"
    out.mkdir(parents=True, exist_ok=True)

    def _load(name: str) -> pd.DataFrame | None:
        path = res / f"results_{name}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        return df[df["config"].isin(SHOWN)].reset_index(drop=True)

    tqa = _load("truthfulqa")
    gsm = _load("gsm8k")
    frames = {k: v for k, v in (("GSM8K", gsm), ("TruthfulQA", tqa)) if v is not None}

    fig_accuracy(tqa, gsm, out)
    fig_routing(frames, out)
    fig_cost(frames, out)
    aurocs = fig_entropy(frames, out)
    fig_metric_audit(tqa, gsm, out)
    fig_architecture(out, args.setup)
    (out / "aurocs.json").write_text(json.dumps(aurocs, indent=2))
    print(f"Figures written to {out} | AUROC: {aurocs}")


if __name__ == "__main__":
    main()
