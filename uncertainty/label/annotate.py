"""
Human labels, the one manual step (PLAN §9.3), done after the Mac run on either machine.

- `export`: ~300 in-scope answers, stratified by set and model, to a CSV you fill in
  (columns `type`, `subtype`, `notes`), plus the labelling guide. No prediction is shown,
  so labels are not biased by the classifier.
- `import`: reads the filled CSVs (one per annotator) into results/human_labels.csv.
- `export-audit` / `import-audit`: 100 judge-graded answers, to check the judge.
Then `python -m uncertainty report obj2` (and `report obj1`) add the agreement figures.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd

from uncertainty.framework.taxonomy import TYPES
from uncertainty.label.correctness import load_labels
from uncertainty.report.obj2 import assemble_all, human_labels_path
from uncertainty.score import load_scores
from uncertainty.sets.load import ALEATORIC_SUBTYPES, Item
from uncertainty.store import load_generations

GUIDE = """# Labelling guide: what kind of uncertainty?

For each row, read the question, the reference answers, the model's answer and three of its
samples. Decide why the model is (or should be) unsure, using this decision tree, and write
the type in the `type` column (and `subtype` for aleatoric).

1. Does the question itself have no single true answer?
   - several reasonable readings with different answers → `aleatoric`, subtype `ambiguous`
   - unknown, unknowable, or a matter of opinion → `aleatoric`, subtype `unanswerable`
   - built on something false ("Why did X…" when X never happened) → `aleatoric`, subtype `false_premise`
2. Does the true answer depend on when it is asked (office holders, records, prices, "latest")? → `temporal`
3. Are the needed facts given in the question (or common knowledge), with the samples going
   wrong or disagreeing in the calculation or deduction? → `reasoning`
4. Otherwise: the model lacks or misremembers a fact (names, dates, places disagree or are
   wrong) → `knowledge`

If two apply, pick the one whose fix would help most (evidence for knowledge, a method for
reasoning). Leave `type` empty if you cannot decide; use `notes` freely. Write your initials
in `annotator` (the same on every row).
"""


def _sample_rows(results_dir: Path, items: dict[str, Item], n: int, seed: int) -> pd.DataFrame:
    scores = load_scores(results_dir)
    scores = scores[scores["wrong"].notna()]
    labels = assemble_all(results_dir, items, scores, use_human=False)
    rng = random.Random(seed)
    groups = {k: g.index.tolist() for k, g in labels.groupby(["set", "model"])}
    for idx in groups.values():
        rng.shuffle(idx)
    picked: list[int] = []
    while len(picked) < n and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(picked) < n:
                picked.append(groups[key].pop())
    return labels.loc[picked]


def export(results_dir: Path, items: dict[str, Item], out: Path, n: int = 300, seed: int = 11) -> Path:
    rows = _sample_rows(results_dir, items, n, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "model", "id", "set", "question", "reference_answers", "model_answer",
                    "sample_1", "sample_2", "sample_3", "annotator", "type", "subtype", "notes"])
        for r in rows.itertuples(index=False):
            key = (r.model, r.set)
            if key not in cache:
                cache[key] = {x["id"]: x for x in load_generations(results_dir, r.model, r.set)}
            rec = cache[key].get(r.id)
            if rec is None:
                continue
            item = items[r.id]
            samples = [s["text"] for s in rec["samples"] if s["text"].strip()][:3] + ["", "", ""]
            w.writerow([f"{r.model}|{r.id}", r.model, r.id, r.set, item.question, " | ".join(item.answers[:6]),
                        rec["greedy"]["text"][:1500], *[s[:600] for s in samples[:3]], "", "", "", ""])
    out.with_name("annotation_guide.md").write_text(GUIDE, encoding="utf-8")
    return out


def import_labels(results_dir: Path, files: list[Path]) -> Path:
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str).fillna("")
        df["type"] = df["type"].str.strip().str.lower()
        df["subtype"] = df["subtype"].str.strip().str.lower()
        bad = df[(df["type"] != "") & ~df["type"].isin(TYPES)]
        if len(bad):
            raise ValueError(f"{f}: unknown type(s) {sorted(bad['type'].unique())}; use one of {TYPES}")
        bad = df[(df["subtype"] != "") & ~df["subtype"].isin(ALEATORIC_SUBTYPES)]
        if len(bad):
            raise ValueError(f"{f}: unknown subtype(s) {sorted(bad['subtype'].unique())}")
        if (df["annotator"].str.strip() == "").any():
            df["annotator"] = df["annotator"].replace("", f.stem)
        frames.append(df[["uid", "model", "id", "set", "annotator", "type", "subtype", "notes"]])
    out = human_labels_path(results_dir)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    return out


def export_audit(results_dir: Path, items: dict[str, Item], out: Path, n: int = 100, seed: int = 13) -> Path:
    """Judge-graded answers for a blind human check (the judge's verdict is not shown)."""
    scores = load_scores(results_dir)
    pool = []
    for model in scores["model"].unique():
        for i, lab in load_labels(results_dir, model).items():
            if lab["source"].startswith("judge_") and lab["correct"] is not None and i in items:
                pool.append((model, i, lab["set"]))
    random.Random(seed).shuffle(pool)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "model", "id", "question", "reference_answers", "model_answer", "correct", "notes"])
        for model, i, set_name in pool[:n]:
            if (model, set_name) not in cache:
                cache[(model, set_name)] = {x["id"]: x for x in load_generations(results_dir, model, set_name)}
            rec = cache[(model, set_name)].get(i)
            if rec:
                item = items[i]
                refs = " | ".join(item.answers[:6]) or "(no single true answer: correct = says it cannot be known)"
                w.writerow([f"{model}|{i}", model, i, item.question, refs, rec["greedy"]["text"][:1500], "", ""])
    return out


def import_audit(results_dir: Path, file: Path) -> dict:
    df = pd.read_csv(file, dtype=str).fillna("")
    df = df[df["correct"].str.strip().str.lower().isin(["yes", "no", "true", "false", "1", "0"])]
    human = df["correct"].str.strip().str.lower().isin(["yes", "true", "1"])
    agree = []
    for (model, i), h in zip(zip(df["model"], df["id"]), human):
        lab = load_labels(results_dir, model).get(i)
        if lab and lab["correct"] is not None:
            agree.append(bool(lab["correct"]) == bool(h))
    result = {"n": len(agree), "judge_agreement_with_human": sum(agree) / len(agree) if agree else None}
    pd.DataFrame([result]).to_csv(results_dir / "judge_audit.csv", index=False)
    return result
