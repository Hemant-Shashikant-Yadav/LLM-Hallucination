"""
Objective 1 report: how unsure each raw model is, and how well each published method
predicts its mistakes (PLAN §8.2-8.3).

Reads results/<model>/scores.csv (test split for every number, dev only for thresholds)
and writes results/reports/obj1/: CSV tables, LaTeX, figures, summary.json and obj1.md.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from uncertainty import metrics, thresholds
from uncertainty.label.correctness import load_labels
from uncertainty.methods.registry import BY_NAME, METHODS, PRIMARY
from uncertainty.report import style
from uncertainty.roster import PIPELINE_MODEL, spec
from uncertainty.score import K_CURVE, load_scores, scores_path
from uncertainty.sets.load import SETS
from uncertainty.settings import PROJECT_ROOT, settings
from uncertainty.store import model_dir, write_json

KEY_METHODS = ["se_discrete", "p_true", "nll_norm", "verbal"]
CACHE_VERSION = 1


def _f(x) -> np.ndarray:
    return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)


def _groups(test: pd.DataFrame):
    for model, by_model in test.groupby("model", sort=False):
        yield model, "ALL", by_model
        for set_name in [s for s in SETS if s in set(by_model["set"])]:
            yield model, set_name, by_model[by_model["set"] == set_name]


def method_table(test: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for model, set_name, g in _groups(test):
        w = _f(g["wrong"])
        for m in METHODS:
            if m.name not in g:
                continue
            u = _f(g[m.name])
            ok = ~np.isnan(u)
            lo, hi = metrics.bootstrap_auroc(u[ok], w[ok], n_boot=n_boot)
            rows.append({
                "model": model, "set": set_name, "method": m.name, "label": m.label, "family": m.family,
                "n": int(ok.sum()), "n_wrong": int(w[ok].sum()), "auroc": metrics.auroc(u[ok], w[ok]),
                "auroc_lo": lo, "auroc_hi": hi, "auprc": metrics.auprc(u[ok], w[ok]),
                "aurac": metrics.aurac(u[ok], w[ok]), "prr": metrics.prr(u[ok], w[ok]),
                "acc_at_80": metrics.accuracy_at_coverage(u[ok], w[ok], 0.8),
                "acc_at_90": metrics.accuracy_at_coverage(u[ok], w[ok], 0.9),
            })
    return pd.DataFrame(rows)


def delong_table(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, set_name, g in _groups(test):
        w = _f(g["wrong"])
        group = []
        for m in METHODS:
            if m.name == PRIMARY or m.name not in g:
                continue
            d = metrics.delong(_f(g[PRIMARY]), _f(g[m.name]), w)
            if d:
                group.append({"model": model, "set": set_name, "method": m.name, **d})
        for row, p_adj in zip(group, metrics.holm([r["p"] for r in group])):
            row["p_holm"] = p_adj
        rows += group
    return pd.DataFrame(rows)


def profile_table(df: pd.DataFrame, test: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    rows = []
    for model, set_name, g in _groups(test):
        t = thresholds.load(results_dir, model)
        if not t.get("tuned"):
            t = thresholds.tune_model(results_dir, model, df[df["model"] == model])
        se = _f(g["se_norm"])
        wrong = _f(g["wrong"])
        sure = se <= t["low"] + 1e-9
        rows.append({
            "model": model, "set": set_name, "n": len(g), "accuracy": float(np.nanmean(1 - wrong)),
            "mean_se_norm": float(np.nanmean(se)) if np.isfinite(se).any() else None,
            "share_high_uncertainty": float(np.nanmean(se >= t["high"] - 1e-9)),
            "share_sure": float(np.nanmean(sure)),
            "blind_spot": float(sure[wrong == 1].mean()) if (wrong == 1).any() else None,
            "confident_error_rate": float((sure & (wrong == 1)).mean()),
            "threshold_low": t["low"], "threshold_high": t["high"],
        })
    return pd.DataFrame(rows)


def calibration_table(test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, bins = [], {}
    for model, set_name, g in _groups(test):
        correct = 1 - _f(g["wrong"])
        for col, name in (("p_true_conf", "p_true"), ("verbal_conf", "verbal")):
            c = _f(g[col])
            ok = ~np.isnan(c)
            rows.append({"model": model, "set": set_name, "method": name, "n": int(ok.sum()),
                         "ece": metrics.ece(c[ok], correct[ok], settings.ece_bins),
                         "brier": metrics.brier(c[ok], correct[ok]),
                         "mean_confidence": float(c[ok].mean()) if ok.any() else None,
                         "accuracy": float(correct[ok].mean()) if ok.any() else None})
            if set_name == "ALL":
                bins[f"{model}|{name}"] = metrics.reliability(c[ok], correct[ok])
    return pd.DataFrame(rows), bins


def k_curve_table(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in test.groupby("model", sort=False):
        w = _f(g["wrong"])
        for k in K_CURVE:
            rows.append({"model": model, "k": k, "auroc": metrics.auroc(_f(g[f"se_k{k}"]), w)})
        rows.append({"model": model, "k": int(g["n_samples"].max() or 10), "auroc": metrics.auroc(_f(g[PRIMARY]), w)})
        rows.append({"model": model, "k": "pipeline_k4", "auroc": metrics.auroc(_f(g["se_pipeline_k4"]), w)})
    return pd.DataFrame(rows)


def size_table(table: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in profile["model"].unique():
        s = spec(model)
        p = profile[(profile["model"] == model) & (profile["set"] == "ALL")].iloc[0]
        a = table[(table["model"] == model) & (table["set"] == "ALL") & (table["method"] == PRIMARY)]
        rows.append({"model": model, "family": s.family, "size_b": s.size_b, "generation": s.generation,
                     "accuracy": p["accuracy"], "mean_se_norm": p["mean_se_norm"], "blind_spot": p["blind_spot"],
                     "auroc_se": a["auroc"].iloc[0] if len(a) else None})
    return pd.DataFrame(rows)


def cost_table(test: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in test.groupby("model", sort=False):
        k = float(g["n_samples"].mean())
        sample_tok = float(g["sample_tokens"].mean())
        self_tok = float(g["self_eval_tokens"].mean()) / 2
        for m in METHODS:
            if "samples" in m.needs:
                calls, tokens = k, sample_tok
            elif m.family == "self-eval":
                calls, tokens = 1, self_tok
            else:
                calls, tokens = 0, 0.0
            a = table[(table["model"] == model) & (table["set"] == "ALL") & (table["method"] == m.name)]
            rows.append({"model": model, "method": m.name, "extra_calls": calls, "extra_tokens": tokens,
                         "auroc": a["auroc"].iloc[0] if len(a) else None})
    return pd.DataFrame(rows)


def judge_agreement(results_dir: Path, models: list[str]) -> pd.DataFrame:
    from sklearn.metrics import cohen_kappa_score

    rows = []
    for model in models:
        first, second = load_labels(results_dir, model), load_labels(results_dir, model, second=True)
        pairs = [(first[i]["correct"], r["correct"]) for i, r in second.items()
                 if i in first and first[i]["correct"] is not None and r["correct"] is not None]
        alias = [(r["alias_match"], r["correct"]) for r in first.values()
                 if r.get("alias_match") is not None and r["correct"] is not None]
        row = {"model": model, "second_judge_n": len(pairs)}
        if pairs:
            a, b = zip(*pairs)
            row["second_judge_agreement"] = float(np.mean(np.array(a) == np.array(b)))
            row["second_judge_kappa"] = float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else None
        if alias:
            a, b = zip(*alias)
            row["alias_n"] = len(alias)
            row["alias_agreement"] = float(np.mean(np.array(a) == np.array(b)))
        rows.append(row)
    return pd.DataFrame(rows)


def layer1_pipeline() -> pd.DataFrame:
    """AUROC of the pipeline's stored Layer 1 entropy on its own test runs (no model needed)."""
    runs = {"gsm8k": "evaluation/results/test_qwen3b_v6/results_gsm8k.csv",
            "truthfulqa": "evaluation/results/test_qwen3b_v5/results_truthfulqa.csv"}
    rows = []
    for name, rel in runs.items():
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        d = pd.read_csv(path)
        raw = d[d["config"] == "baseline"][["sample_id", "is_correct"]]
        fw = d[d["config"] == "full_framework"][["sample_id", "semantic_entropy"]]
        j = raw.merge(fw, on="sample_id").dropna()
        wrong = 1 - j["is_correct"].astype(float)
        lo, hi = metrics.bootstrap_auroc(j["semantic_entropy"], wrong, n_boot=settings.bootstrap)
        rows.append({"set": name, "run": rel, "n": len(j), "accuracy": float(1 - wrong.mean()),
                     "auroc_layer1_hsem": metrics.auroc(j["semantic_entropy"], wrong), "auroc_lo": lo, "auroc_hi": hi,
                     "note": "qwen2.5:3b, K = 4 at T = 0.7, pipeline prompt; computed from stored CSVs"})
    return pd.DataFrame(rows)


# ============================================================================ figures


def _main_model(models: list[str]) -> str:
    return PIPELINE_MODEL if PIPELINE_MODEL in models else models[0]


def figures(test: pd.DataFrame, table: pd.DataFrame, cal_bins: dict, kcurve: pd.DataFrame,
            sizes: pd.DataFrame, cost: pd.DataFrame, out: Path) -> list[str]:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    style.setup()
    made = []
    models = list(dict.fromkeys(test["model"]))
    main = _main_model(models)
    g = test[test["model"] == main]
    w = _f(g["wrong"])
    slug = spec(main).slug

    # 1. AUROC heatmap: methods × models (pooled test questions)
    pivot = table[table["set"] == "ALL"].pivot(index="method", columns="model", values="auroc")
    pivot = pivot.reindex([m.name for m in METHODS if m.name in pivot.index])[models]
    # Width: room for the method labels (~3.5 in), then ~0.9 in per model column, then the colour bar
    fig, ax = plt.subplots(figsize=(max(8.0, 4.6 + 0.9 * len(models)), 0.42 * len(pivot) + 1.8))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap=style.BLUES, vmin=0.5, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(models)), [spec(m).tag for m in models], rotation=35, ha="right")
    ax.set_yticks(range(len(pivot)), [BY_NAME[m].label for m in pivot.index])
    ax.grid(False)
    for (i, j), v in np.ndenumerate(pivot.to_numpy(dtype=float)):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.75 else style.TEXT)
    fig.colorbar(im, ax=ax, fraction=0.03, label="AUROC (wrong answer = positive)")
    ax.set_title("How well each method predicts a wrong answer (test questions, all sets)")
    made.append(str(style.save(fig, out / "fig_auroc_heatmap.png")))

    # 2-3. ROC and rejection curves for the main model's key methods
    fig_roc, ax_roc = plt.subplots(figsize=(5.2, 4.6))
    fig_rej, ax_rej = plt.subplots(figsize=(5.6, 4.2))
    for i, name in enumerate(KEY_METHODS):
        u = _f(g[name])
        ok = ~np.isnan(u)
        if ok.sum() < 5 or len(set(w[ok])) < 2:
            continue
        fpr, tpr, _ = roc_curve(w[ok], u[ok])
        auc = metrics.auroc(u[ok], w[ok])
        ax_roc.plot(fpr, tpr, color=style.color(i), label=f"{BY_NAME[name].label} ({auc:.2f})")
        acc = metrics.accuracy_curve(u[ok], 1 - w[ok])
        ax_rej.plot(np.arange(1, len(acc) + 1) / len(acc), acc, color=style.color(i), label=BY_NAME[name].label)
    ax_roc.plot([0, 1], [0, 1], color=style.GRID, linewidth=1)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC, {main}")
    ax_roc.legend(loc="lower right", fontsize=8)
    made.append(str(style.save(fig_roc, out / f"fig_roc_{slug}.png")))
    ax_rej.axhline(1 - np.nanmean(w), color=style.GRID, linewidth=1)
    ax_rej.set(xlabel="Share of answers kept (most certain first)", ylabel="Accuracy of kept answers",
               title=f"Refusing the least certain answers, {main}")
    ax_rej.legend(fontsize=8)
    made.append(str(style.save(fig_rej, out / f"fig_rejection_{slug}.png")))

    # 4. Reliability: P(True) and verbalised confidence
    fig, ax = plt.subplots(figsize=(5, 4.6))
    for i, name in enumerate(("p_true", "verbal")):
        b = cal_bins.get(f"{main}|{name}") or []
        if b:
            ax.plot([x["confidence"] for x in b], [x["accuracy"] for x in b], marker="o", markersize=5,
                    color=style.color(i), label=BY_NAME[name].label)
    ax.plot([0, 1], [0, 1], color=style.GRID, linewidth=1)
    ax.set(xlabel="Stated confidence", ylabel="Actual accuracy", xlim=(0, 1), ylim=(0, 1),
           title=f"Calibration, {main}")
    ax.legend(fontsize=8)
    made.append(str(style.save(fig, out / f"fig_reliability_{slug}.png")))

    # 5. Entropy of right vs wrong answers
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    se = _f(g["se_norm"])
    bins = np.linspace(0, 1, 11)
    for i, (label, sel) in enumerate((("Right answers", w == 0), ("Wrong answers", w == 1))):
        vals = se[sel & ~np.isnan(se)]
        if len(vals):
            ax.hist(vals, bins=bins, histtype="step", linewidth=2, color=style.color(i), label=label,
                    weights=np.full(len(vals), 1 / len(vals)))
    ax.set(xlabel="Semantic entropy (0 = all samples agree, 1 = all differ)", ylabel="Share of answers",
           title=f"Uncertainty of right and wrong answers, {main}")
    ax.legend(fontsize=8)
    made.append(str(style.save(fig, out / f"fig_entropy_{slug}.png")))

    # 6. AUROC against the number of samples (up to four models: legend + palette stay readable)
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for i, model in enumerate(models[:4]):
        kc = kcurve[(kcurve["model"] == model) & (kcurve["k"] != "pipeline_k4")]
        ax.plot(kc["k"].astype(int), kc["auroc"], marker="o", markersize=5, color=style.color(i), label=model)
    ax.set(xlabel="Samples per question (K)", ylabel="AUROC of semantic entropy", title="How many samples are needed")
    ax.legend(fontsize=8)
    made.append(str(style.save(fig, out / "fig_auroc_vs_k.png")))

    # 7. AUROC against cost, main model (one series, labelled directly)
    c = cost[(cost["model"] == main)].dropna(subset=["auroc"])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(c["extra_tokens"], c["auroc"], s=40, color=style.color(0), edgecolor=style.SURFACE, linewidth=2)
    for _, r in c.iterrows():
        ax.annotate(BY_NAME[r["method"]].label, (r["extra_tokens"], r["auroc"]), fontsize=7, color=style.TEXT_2,
                    xytext=(4, 3), textcoords="offset points")
    ax.set(xlabel="Extra generated tokens per question", ylabel="AUROC", title=f"Accuracy of the signal vs its cost, {main}")
    made.append(str(style.save(fig, out / f"fig_auroc_vs_cost_{slug}.png")))

    # 8. Size effects (small multiples, one axis each, models labelled directly)
    if len(sizes) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
        for ax, (col, title) in zip(axes, (("accuracy", "Accuracy"), ("auroc_se", "AUROC of semantic entropy"),
                                           ("blind_spot", "Blind spot (errors made while sure)"))):
            ax.scatter(sizes["size_b"], sizes[col], s=40, color=style.color(0), edgecolor=style.SURFACE, linewidth=2)
            for _, r in sizes.iterrows():
                ax.annotate(r["model"], (r["size_b"], r[col]), fontsize=7, color=style.TEXT_2,
                            xytext=(4, 3), textcoords="offset points")
            ax.set(title=title, xlabel="Parameters (billions)")
        made.append(str(style.save(fig, out / "fig_size_effects.png")))
    return made


# ============================================================================ build


def _latex(table: pd.DataFrame, models: list[str]) -> str:
    t = table[table["set"] == "ALL"]
    lines = [r"\begin{tabular}{l" + "c" * len(models) + "}", r"\toprule",
             "Method & " + " & ".join(m.replace("_", r"\_") for m in models) + r" \\", r"\midrule"]
    for m in METHODS:
        cells = []
        for model in models:
            r = t[(t["model"] == model) & (t["method"] == m.name)]
            v = r["auroc"].iloc[0] if len(r) else None
            cells.append("--" if v is None or pd.isna(v) else f"{v:.3f}")
        lines.append(f"{m.label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _md_table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append("–" if v is None or (isinstance(v, float) and np.isnan(v)) else
                         (fmt[c].format(v) if c in fmt else str(v)))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def _per_model(results_dir: Path, model: str, df: pd.DataFrame, test: pd.DataFrame, split: str) -> dict:
    """
    One model's tables, cached beside its scores: reports are rebuilt after every model, and
    without the cache each rebuild would redo every earlier model's bootstraps.
    """
    path = model_dir(results_dir, model) / "obj1_cache.pkl"
    src = scores_path(results_dir, model)
    key = {"scores_mtime": src.stat().st_mtime if src.exists() else None, "split": split,
           "n_boot": settings.bootstrap, "version": CACHE_VERSION}
    if path.exists():
        try:
            cached = pickle.loads(path.read_bytes())
            if cached.get("key") == key:
                return cached
        except Exception:
            pass
    thresholds.tune_model(results_dir, model, df)
    calibration, bins = calibration_table(test)
    part = {"key": key, "table": method_table(test, settings.bootstrap), "delong": delong_table(test),
            "profile": profile_table(df, test, results_dir), "calibration": calibration, "bins": bins,
            "kcurve": k_curve_table(test)}
    path.write_bytes(pickle.dumps(part))
    return part


def build(results_dir: Path, models: list[str] | None = None, split: str = "test") -> dict:
    """`split='all'` reports every question (the pilot, which uses dev questions only)."""
    out = results_dir / "reports" / "obj1"
    out.mkdir(parents=True, exist_ok=True)
    df = load_scores(results_dir, models)
    df = df[df["wrong"].notna()] if not df.empty else df
    test = df if split == "all" else df[df["split"] == split] if not df.empty else df
    if test.empty or test["wrong"].nunique() < 2:
        logger.warning("Objective 1 report: not enough scored answers yet")
        return {}
    models = list(dict.fromkeys(test["model"]))
    parts = [_per_model(results_dir, m, df[df["model"] == m], test[test["model"] == m], split) for m in models]
    table = pd.concat([p["table"] for p in parts], ignore_index=True)
    delong = pd.concat([p["delong"] for p in parts], ignore_index=True)
    profile = pd.concat([p["profile"] for p in parts], ignore_index=True)
    calibration = pd.concat([p["calibration"] for p in parts], ignore_index=True)
    cal_bins = {k: v for p in parts for k, v in p["bins"].items()}
    kcurve = pd.concat([p["kcurve"] for p in parts], ignore_index=True)
    sizes = size_table(table, profile)
    cost = cost_table(test, table)
    judges = judge_agreement(results_dir, models)
    layer1 = layer1_pipeline()
    for name, frame in (("method_table", table), ("delong", delong), ("profile", profile),
                        ("calibration", calibration), ("k_curve", kcurve), ("size_effects", sizes),
                        ("cost", cost), ("judge_agreement", judges), ("layer1_pipeline", layer1)):
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "reliability.json").write_text(json.dumps(cal_bins, indent=1), encoding="utf-8")
    (out / "method_table_all.tex").write_text(_latex(table, models), encoding="utf-8")
    figs = figures(test, table, cal_bins, kcurve, sizes, cost, out / "figures")

    all_rows = table[(table["set"] == "ALL")]
    headline = []
    for model in models:
        t = all_rows[all_rows["model"] == model].dropna(subset=["auroc"])
        se = t[t["method"] == PRIMARY]
        best = t.sort_values("auroc", ascending=False).head(1)
        p = profile[(profile["model"] == model) & (profile["set"] == "ALL")].iloc[0]
        headline.append({
            "model": model, "n_test": int(p["n"]), "accuracy": p["accuracy"],
            "auroc_se": se["auroc"].iloc[0] if len(se) else None,
            "auroc_se_ci": [se["auroc_lo"].iloc[0], se["auroc_hi"].iloc[0]] if len(se) else None,
            "best_method": best["method"].iloc[0] if len(best) else None,
            "best_auroc": best["auroc"].iloc[0] if len(best) else None,
            "blind_spot": p["blind_spot"], "mean_se_norm": p["mean_se_norm"],
        })
    summary = {"models": models, "n_questions_test": int(len(test)), "headline": headline,
               "layer1_pipeline": layer1.to_dict("records"), "figures": figs}
    write_json(out / "summary.json", summary)

    h = pd.DataFrame(headline)
    md = ["# Objective 1: baseline uncertainty of raw models\n",
          f"Test questions: {len(test)} across {len(models)} models. Primary method: semantic entropy "
          "(Farquhar et al., 2024). AUROC: a wrong answer is the positive class; 0.5 = chance.\n",
          "## Headline per model\n",
          _md_table(h.assign(ci=h["auroc_se_ci"].map(lambda c: "–" if not c or c[0] is None else f"{c[0]:.2f}–{c[1]:.2f}")),
                    ["model", "n_test", "accuracy", "auroc_se", "ci", "best_method", "best_auroc", "blind_spot"],
                    {"accuracy": "{:.3f}", "auroc_se": "{:.3f}", "best_auroc": "{:.3f}", "blind_spot": "{:.3f}"}),
          "\n`blind_spot`: share of wrong answers given while the samples agreed (semantic entropy at or below "
          "the dev-tuned 'sure' threshold): confident hallucinations that sampling cannot catch.\n",
          "## Pipeline Layer 1 on its own stored test runs\n",
          _md_table(layer1, ["set", "n", "accuracy", "auroc_layer1_hsem", "auroc_lo", "auroc_hi"],
                    {"accuracy": "{:.2f}", "auroc_layer1_hsem": "{:.3f}", "auroc_lo": "{:.2f}", "auroc_hi": "{:.2f}"})
          if len(layer1) else "(stored pipeline runs not found)\n",
          "\n## Files\n", "Tables: `method_table.csv` (every model × set × method), `delong.csv`, `profile.csv`, "
          "`calibration.csv`, `k_curve.csv`, `size_effects.csv`, `cost.csv`, `judge_agreement.csv`, "
          "`layer1_pipeline.csv`, `method_table_all.tex`. Figures: `figures/`.\n"]
    (out / "obj1.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Objective 1 report → {}", out)
    return summary
