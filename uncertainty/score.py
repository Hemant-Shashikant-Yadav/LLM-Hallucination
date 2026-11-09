"""
Objective 1 scores: every method on every answered question → results/<model>/scores.csv.

Offline and CPU-only: reads the stored generations, NLI matrices and correctness labels.
Re-running it after a method changes costs no model time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from uncertainty.framework.features import question_features, structure_features
from uncertainty.label.correctness import load_labels
from uncertainty.methods.nli import load_matrices, nli_path
from uncertainty.methods.registry import METHODS, score_view
from uncertainty.methods.semantic_entropy import se_discrete, se_normalized, se_pipeline
from uncertainty.methods.view import View, view_from_record
from uncertainty.roster import spec
from uncertainty.sets.load import SETS
from uncertainty.store import load_generations, model_dir


K_CURVE = (2, 4, 6, 8)
LITE_K = 4
# Signals the lite mode has: everything except the self-evaluation calls
LITE_METHODS = [m.name for m in METHODS if m.family != "self-eval"]


def scores_path(results_dir: Path, model: str) -> Path:
    return model_dir(results_dir, model) / "scores.csv"


def record_features(rec: dict, view: View) -> dict:
    """Every score and feature of one answered question (also used by the analyze tool)."""
    lite = view.subset(LITE_K)
    return {
        "n_samples": view.m,
        "n_truncated": sum(x.get("done_reason") == "length" for x in rec["samples"]),
        "greedy_truncated": rec["greedy"].get("done_reason") == "length",
        "se_norm": se_normalized(view),
        "p_true_conf": view.p_true, "verbal_conf": None if view.verbal is None else view.verbal / 100,
        **{f"probe_{k}": v.get("p") for k, v in rec.get("probes", {}).items()},
        "calls": rec["cost"]["calls"], "greedy_tokens": rec["greedy"].get("eval_count", 0),
        "sample_tokens": rec["cost"].get("sample_tokens", 0),
        "self_eval_tokens": rec["cost"].get("self_eval_tokens", 0),
        "seconds": rec["cost"].get("seconds", 0.0),
        **score_view(view),
        # Fewer samples (AUROC-vs-K) and the pipeline's own operating point (K = 4)
        **{f"se_k{k}": se_discrete(view.subset(k)) for k in K_CURVE},
        "se_pipeline_k4": se_pipeline(lite),
        # Objective 2 features, and the same signals at the 'lite' budget (K = 4, no extra calls)
        **structure_features(view),
        **question_features(rec["kind"], rec["question"], rec["greedy"]["text"]),
        **{f"lite__{k}": v for k, v in score_view(lite).items() if k in LITE_METHODS},
        "lite__se_norm": se_normalized(lite),
        **{f"lite__{k}": v for k, v in structure_features(lite).items()},
    }


def score_model(results_dir: Path, model: str, sets: list[str] | None = None) -> pd.DataFrame:
    labels = load_labels(results_dir, model)
    s = spec(model)
    rows = []
    for set_name in sets or list(SETS):
        records = load_generations(results_dir, model, set_name)
        if not records:
            continue
        mats = load_matrices(nli_path(results_dir, s.slug, set_name))
        for rec in records:
            view = view_from_record(rec, nli=mats.get(rec["id"]))
            lab = labels.get(rec["id"], {})
            correct = lab.get("correct")
            rows.append({
                "model": model, "family": s.family, "size_b": s.size_b, "generation": s.generation,
                "set": set_name, "id": rec["id"], "split": rec["split"], "core": rec["core"], "kind": rec["kind"],
                "correct": None if correct is None else int(bool(correct)),
                "wrong": None if correct is None else int(not correct),
                "correct_source": lab.get("source"), "alias_match": lab.get("alias_match"),
                **record_features(rec, view),
            })
    df = pd.DataFrame(rows)
    path = scores_path(results_dir, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("{} | scores for {} questions → {}", model, len(df), path)
    return df


def load_scores(results_dir: Path, models: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for d in sorted(p for p in results_dir.iterdir() if p.is_dir()) if results_dir.exists() else []:
        f = d / "scores.csv"
        if f.exists():
            df = pd.read_csv(f)
            if models is None or (len(df) and df["model"].iloc[0] in models):
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["model", "set", "id"] + [m.name for m in METHODS])
    return pd.concat(frames, ignore_index=True)
