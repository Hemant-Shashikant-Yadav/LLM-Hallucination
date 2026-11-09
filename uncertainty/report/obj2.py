"""
Objective 2: train and evaluate the framework, save it for the analyze tool, and write the
report (PLAN §9.5).

Reads scores, thresholds, interventions and (when imported) human labels; writes
results/models/*.joblib and results/reports/obj2/ (JSON and CSV tables, rules, figures,
an example gallery and obj2.md). Rebuilding takes seconds and needs no model, so it is
re-run after the human labels are imported.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from uncertainty import thresholds
from uncertainty.framework import classifier, detector, taxonomy
from uncertainty.framework.features import pipeline_task_type
from uncertainty.label.interventions import load_interventions
from uncertainty.report import style
from uncertainty.roster import spec
from uncertainty.score import load_scores
from uncertainty.sets.load import Item
from uncertainty.settings import settings
from uncertainty.store import load_generations, write_json


def models_dir(results_dir: Path) -> Path:
    return results_dir / "models"


def human_labels_path(results_dir: Path) -> Path:
    return results_dir / "human_labels.csv"


def load_human(results_dir: Path, annotator: str | None = None) -> dict[tuple[str, str], dict]:
    path = human_labels_path(results_dir)
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    first = annotator or (sorted(df["annotator"].unique())[0] if len(df) else None)
    df = df[df["annotator"] == first]
    return {(r.model, r.id): {"type": r.type.strip().lower(), "subtype": r.subtype.strip().lower() or None}
            for r in df.itertuples(index=False) if r.type.strip()}


def assemble_all(results_dir: Path, items: dict[str, Item], scores: pd.DataFrame,
                 use_human: bool = True) -> pd.DataFrame:
    models = list(dict.fromkeys(scores["model"]))
    thr = {m: thresholds.load(results_dir, m) for m in models}
    iv = {(m, i): r for m in models for i, r in load_interventions(results_dir, m).items()}
    human = load_human(results_dir) if use_human else {}
    return taxonomy.assemble(scores, items, thr, iv, human)


def _actionability(results_dir: Path, scored: pd.DataFrame, oof: dict, bundle: dict, n_boot: int) -> dict:
    """Does the intervention matching the predicted type help more than the other one?"""
    rows = []
    for (model, item_id), g in scored.groupby(["model", "id"]):
        iv = load_interventions_cached(results_dir, model).get(item_id)
        if not iv or "evidence" not in iv or "hint" not in iv:
            continue
        pred = oof.get((model, item_id))
        if pred is None and bundle:
            pred, _, _ = classifier.predict(bundle, g.iloc[0].to_dict())
        if pred not in ("knowledge", "reasoning"):
            continue
        ev, hint = iv["evidence"]["resolved"], iv["hint"]["resolved"]
        matched, other = (ev, hint) if pred == "knowledge" else (hint, ev)
        rows.append((float(matched), float(other)))
    if not rows:
        return {"n": 0}
    a = np.array(rows)
    rng = np.random.default_rng(0)
    diffs = [a[i, 0].mean() - a[i, 1].mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    return {"n": len(a), "matched_resolve_rate": float(a[:, 0].mean()), "other_resolve_rate": float(a[:, 1].mean()),
            "diff": float(a[:, 0].mean() - a[:, 1].mean()),
            "diff_lo": float(np.quantile(diffs, 0.025)), "diff_hi": float(np.quantile(diffs, 0.975))}


_IV_CACHE: dict[tuple[str, str], dict] = {}


def load_interventions_cached(results_dir: Path, model: str) -> dict:
    key = (str(results_dir), model)
    if key not in _IV_CACHE:
        _IV_CACHE[key] = load_interventions(results_dir, model)
    return _IV_CACHE[key]


def _human_agreement(results_dir: Path, weak: pd.DataFrame, oof: dict) -> dict:
    from sklearn.metrics import cohen_kappa_score

    path = human_labels_path(results_dir)
    if not path.exists():
        return {"available": False}
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df["type"].str.strip() != ""]
    out: dict = {"available": True, "n_labels": len(df), "annotators": sorted(df["annotator"].unique())}
    if len(out["annotators"]) >= 2:
        a, b = out["annotators"][:2]
        pa = df[df["annotator"] == a].set_index("uid")["type"]
        pb = df[df["annotator"] == b].set_index("uid")["type"]
        both = pa.index.intersection(pb.index)
        if len(both) >= 5:
            out["inter_annotator"] = {"n": len(both), "agreement": float((pa[both] == pb[both]).mean()),
                                      "kappa": float(cohen_kappa_score(pa[both], pb[both]))}
    human = load_human(results_dir)
    w = {(r.model, r.id): r.type for r in weak.itertuples(index=False) if isinstance(r.type, str)}
    pairs = [(h["type"], w[k]) for k, h in human.items() if k in w]
    if len(pairs) >= 5:
        x, y = zip(*pairs)
        out["human_vs_weak_labels"] = {"n": len(pairs), "agreement": float(np.mean(np.array(x) == np.array(y))),
                                       "kappa": float(cohen_kappa_score(x, y))}
    pairs = [(h["type"], oof[k]) for k, h in human.items() if k in oof]
    if len(pairs) >= 5:
        x, y = zip(*pairs)
        out["human_vs_classifier"] = {"n": len(pairs), "agreement": float(np.mean(np.array(x) == np.array(y))),
                                      "kappa": float(cohen_kappa_score(x, y))}
    return out


def _figures(res: dict, out: Path) -> list[str]:
    import matplotlib.pyplot as plt

    style.setup()
    made = []
    best = res["best"]
    cm = np.array(res["classifiers"][best]["confusion"], dtype=float)
    rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(rates, cmap=style.BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(taxonomy.TYPES)), taxonomy.TYPES, rotation=25, ha="right")
    ax.set_yticks(range(len(taxonomy.TYPES)), taxonomy.TYPES)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(rates):
        ax.text(j, i, f"{v:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=8,
                color="white" if v > 0.55 else style.TEXT)
    ax.set(xlabel="Predicted type", ylabel="Labelled type", title=f"Confusion ({best}, rows sum to 1)")
    made.append(str(style.save(fig, out / "fig_confusion.png")))

    names = ["baseline", "rules", best]
    width = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, name in enumerate(names):
        f1s = [c["f1"] for c in res["classifiers"][name]["per_class"]]
        ax.bar(np.arange(len(taxonomy.TYPES)) + (i - 1) * width, f1s, width * 0.9, color=style.color(i),
               label={"baseline": "Pipeline rule (baseline)", "rules": "Decision rules"}.get(name, f"Learned ({name})"))
    ax.set_xticks(range(len(taxonomy.TYPES)), taxonomy.TYPES)
    ax.set(ylabel="F1", ylim=(0, 1), title="Per-type F1 of the three classifiers")
    ax.legend(fontsize=8)
    made.append(str(style.save(fig, out / "fig_per_class_f1.png")))

    imp = res.get("importance") or []
    if imp:
        top = imp[:12][::-1]
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(top) + 1.2))
        ax.barh([t["feature"] for t in top], [t["importance"] for t in top], height=0.6, color=style.color(0))
        ax.set(xlabel="Drop in macro-F1 when shuffled", title="Which signals the classifier relies on")
        made.append(str(style.save(fig, out / "fig_importance.png")))
    return made


def _examples(results_dir: Path, data: pd.DataFrame, oof: list, per_type: int = 3) -> str:
    md = ["# Objective 2: examples of each uncertainty type\n",
          "Answers whose labelled type the classifier also predicted (out-of-fold).\n"]
    d = data.assign(pred=oof)
    for t in taxonomy.TYPES:
        hits = d[(d["type"] == t) & (d["pred"] == t)].head(per_type)
        md.append(f"\n## {t}\n\n{taxonomy.DESCRIPTIONS[t]}\n")
        for r in hits.itertuples(index=False):
            recs = {x["id"]: x for x in load_generations(results_dir, r.model, r.set)}
            rec = recs.get(r.id)
            if rec is None:
                continue
            answer = " ".join(rec["greedy"]["text"].split())[:300]
            md.append(f"- **{r.model} · {r.set}** — {rec['question'][:200]}\n  - Answer: {answer}\n"
                      f"  - Semantic entropy (normalised): {r.se_norm:.2f} · label from: {r.label_source}"
                      + (f" · subtype: {r.subtype}" if isinstance(r.subtype, str) else "") + "\n")
    return "\n".join(md)


def build(results_dir: Path, items: dict[str, Item], n_boot: int | None = None) -> dict:
    n_boot = min(n_boot or settings.bootstrap, 1000)
    out = results_dir / "reports" / "obj2"
    out.mkdir(parents=True, exist_ok=True)
    _IV_CACHE.clear()
    scores = load_scores(results_dir)
    if scores.empty:
        logger.warning("Objective 2: no scores yet")
        return {}
    scores = scores[scores["wrong"].notna()].copy()
    for m in scores["model"].unique():
        if not thresholds.load(results_dir, m).get("tuned"):
            thresholds.tune_model(results_dir, m, scores[scores["model"] == m])

    summary: dict = {"detector": {}, "classifier": {}}
    bundles: dict = {}
    for variant in ("full", "lite"):
        res, bundle = detector.evaluate(scores, variant, n_boot)
        summary["detector"][variant] = res
        bundles[f"detector_{variant}"] = bundle

    labels = assemble_all(results_dir, items, scores)
    labels.to_csv(out / "type_labels.csv", index=False)
    feats = scores.drop(columns=["split", "set"], errors="ignore")
    scored = labels.merge(feats, on=["model", "id"], how="left")
    scored["family"] = scored["model"].map(lambda m: spec(m).family)
    scored["baseline"] = scored["id"].map(lambda i: pipeline_task_type(items[i].question) if i in items else "knowledge")
    data = scored[scored["type"].notna()].reset_index(drop=True)
    summary["labels"] = {
        "in_scope": len(scored), "labelled": len(data),
        "by_type": data["type"].value_counts().to_dict(),
        "by_source": data["label_source"].value_counts().to_dict(),
        "unlabelled_reasons": scored[scored["type"].isna()]["unlabelled_reason"].value_counts().to_dict(),
    }

    oof_map: dict = {}
    for variant in ("full", "lite"):
        res, bundle = classifier.evaluate(data, variant, n_boot)
        if variant == "full" and "oof_best" in res:
            oof_map = dict(zip(zip(data["model"], data["id"]), res["oof_best"]))
            (out / "rules.txt").write_text(res.get("rules_text", ""), encoding="utf-8")
            pd.DataFrame(res.get("importance", [])).to_csv(out / "importance.csv", index=False)
            rows = [{"classifier": n, **pc} for n, c in res["classifiers"].items() for pc in c["per_class"]]
            pd.DataFrame(rows).to_csv(out / "per_class.csv", index=False)
            gen = [{"held_out_by": k, "group": g, **v} for k, d in res["generalisation"].items() for g, v in d.items()]
            pd.DataFrame(gen).to_csv(out / "generalisation.csv", index=False)
            summary["figures"] = _figures(res, out / "figures")
            (out / "examples.md").write_text(_examples(results_dir, data, res["oof_best"]), encoding="utf-8")
        res.pop("oof_best", None)
        summary["classifier"][variant] = res
        bundles[f"classifier_{variant}"] = bundle

    summary["actionability"] = _actionability(results_dir, scored, oof_map, bundles.get("classifier_full"), n_boot)
    weak = assemble_all(results_dir, items, scores, use_human=False)
    summary["human"] = _human_agreement(results_dir, weak, oof_map)

    mdir = models_dir(results_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    for name, bundle in bundles.items():
        if bundle:
            joblib.dump(bundle, mdir / f"{name}.joblib")
    write_json(out / "summary.json", summary)
    (out / "obj2.md").write_text(_markdown(summary), encoding="utf-8")
    logger.info("Objective 2 report → {}", out)
    return summary


def _fmt(v, f="{:.3f}") -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def _markdown(s: dict) -> str:
    det = s["detector"].get("full", {})
    cls = s["classifier"].get("full", {})
    md = ["# Objective 2: identifying and classifying uncertainty\n",
          "## Identify: how unsure is the answer?\n",
          "| Variant | AUROC (95% CI) | AURAC | ECE | Best single method (AUROC) | Semantic entropy AUROC |",
          "|---|---|---|---|---|---|"]
    for v, d in s["detector"].items():
        md.append(f"| {v} | {_fmt(d.get('auroc'))} ({_fmt(d.get('auroc_lo'), '{:.2f}')}–{_fmt(d.get('auroc_hi'), '{:.2f}')}) "
                  f"| {_fmt(d.get('aurac'))} | {_fmt(d.get('ece'))} | {d.get('best_single')} ({_fmt(d.get('best_single_auroc'))}) "
                  f"| {_fmt(d.get('se_auroc'))} |")
    if det.get("delong_vs_se"):
        md.append(f"\nDetector vs semantic entropy (DeLong, test): Δ = {_fmt(det['delong_vs_se']['diff'])}, "
                  f"p = {_fmt(det['delong_vs_se']['p'], '{:.4f}')}.\n")
    md += ["\n## Classify: what kind of uncertainty?\n",
           f"Labelled answers: {s['labels']['labelled']} of {s['labels']['in_scope']} in scope. "
           f"By type: {s['labels']['by_type']}. By source: {s['labels']['by_source']}.\n"]
    if cls.get("classifiers"):
        md += ["| Classifier | Macro-F1 (95% CI) | Accuracy |", "|---|---|---|"]
        for name, c in cls["classifiers"].items():
            md.append(f"| {name} | {_fmt(c['macro_f1'])} ({_fmt(c['macro_f1_lo'], '{:.2f}')}–{_fmt(c['macro_f1_hi'], '{:.2f}')}) "
                      f"| {_fmt(c['accuracy'])} |")
        b = cls["best_vs_baseline"]
        md.append(f"\nBest learned ({cls['best']}) minus the pipeline's rule: {_fmt(b['diff'])} "
                  f"(95% CI {_fmt(b['diff_lo'])} to {_fmt(b['diff_hi'])}); "
                  f"**{'beats' if b['beats_baseline'] else 'does not beat'} the baseline**.\n")
        if cls.get("subtype"):
            md.append(f"Aleatoric subtype (ambiguous / unanswerable / false premise): macro-F1 "
                      f"{_fmt(cls['subtype']['macro_f1'])} on {cls['subtype']['n']} answers.\n")
        md.append("Generalisation (macro-F1 on the held-out group): see `generalisation.csv`.\n")
    a = s.get("actionability", {})
    if a.get("n"):
        md.append(f"\n## Actionability\n\nOn {a['n']} answers where both interventions were run, the intervention "
                  f"matching the predicted type resolved {_fmt(a['matched_resolve_rate'])} vs "
                  f"{_fmt(a['other_resolve_rate'])} for the other (difference {_fmt(a['diff'])}, "
                  f"95% CI {_fmt(a['diff_lo'])} to {_fmt(a['diff_hi'])}).\n")
    h = s.get("human", {})
    md.append("\n## Human labels\n\n" + ("Not imported yet: `python -m uncertainty annotate export`, label, then "
                                         "`annotate import` and `report obj2`.\n" if not h.get("available")
                                         else json.dumps(h, indent=1) + "\n"))
    md.append("\nFiles: `summary.json`, `type_labels.csv`, `per_class.csv`, `generalisation.csv`, `importance.csv`, "
              "`rules.txt`, `examples.md`, `figures/`.\n")
    return "\n".join(md)
