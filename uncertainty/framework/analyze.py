"""
The Objective 2 tool (PLAN §9.6): one question and one model in, an uncertainty report out.

    report = await analyze("Who was the first person to walk on the Moon?", model="qwen2.5:3b")

Generates the answer and its samples, computes every signal, then applies the trained
detector (how unsure) and classifier (what kind), and recommends an action. Modes:
'full' (K = 10, P(True), verbalised confidence, probes) and 'lite' (K = 4, no extra calls,
the pipeline's budget). Needs the models trained by `run-all` or `python -m uncertainty train`.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib

from app.answer_utils import classify_task_heuristic, is_yes_no_question, parse_options
from uncertainty.framework import classifier, detector
from uncertainty.framework.taxonomy import recommended_action
from uncertainty.generate.backends import Backend, make_backend
from uncertainty.generate.runner import answer_record
from uncertainty.methods.nli import FakeNLI, NLIScorer, nli_matrix, nli_texts, valid_samples
from uncertainty.methods.view import view_from_record
from uncertainty.score import LITE_K, record_features
from uncertainty.sets.load import Item
from uncertainty.settings import settings


def infer_kind(question: str) -> str:
    if parse_options(question):
        return "choice"
    if is_yes_no_question(question):
        return "yesno"
    if classify_task_heuristic(question)[0] == "math":
        return "number"
    return "short"


def load_bundles(results_dir: Path, mode: str) -> tuple[dict, dict]:
    mdir = results_dir / "models"
    det, cls = mdir / f"detector_{mode}.joblib", mdir / f"classifier_{mode}.joblib"
    if not det.exists() or not cls.exists():
        raise FileNotFoundError(
            f"No trained framework in {mdir}. Run `python -m uncertainty run-all` (or `train`) first.")
    return joblib.load(det), joblib.load(cls)


async def analyze(question: str, model: str, mode: str = "full", backend: Backend | None = None,
                  nli=None, results_dir: Path | None = None, logprob_mode: str = "raw") -> dict:
    results_dir = results_dir or settings.results_dir
    det_bundle, cls_bundle = load_bundles(results_dir, mode)
    backend = backend or make_backend("ollama")
    kind = infer_kind(question)
    item = Item(id="adhoc", set="adhoc", kind=kind, question=question, answers=[])
    start = time.perf_counter()
    k = settings.k_samples if mode == "full" else LITE_K
    rec = await answer_record(backend, model, item, k=k, logprob_mode=logprob_mode, probes=mode == "full",
                              self_eval=mode == "full")
    matrix = None
    if kind == "short":
        nli = nli or (FakeNLI() if backend.name == "fake" else NLIScorer())
        texts = [rec["greedy"]["text"]] + [s["text"] for s in valid_samples(rec)]
        matrix = nli_matrix(nli, nli_texts(question, texts))
    view = view_from_record(rec, nli=matrix)
    feats = record_features(rec, view)
    p, level = detector.predict(det_bundle, feats)
    type_, probs, subtype = classifier.predict(cls_bundle, feats)
    p_conf = feats.get("p_true_conf")
    v_conf = feats.get("verbal_conf")
    # Overconfidence: the samples agree, but the model's own check doubts the answer
    overconfident = level == "low" and any(c is not None and c < 0.5 for c in (p_conf, v_conf))
    signals = {k_: feats.get(k_) for k_ in ("se_discrete", "se_norm", "p_true", "verbal", "eigv", "nll_norm",
                                             "num_clusters", "top_share", "premise_minus_conclusion")}
    signals.update({k_: v for k_, v in feats.items() if k_.startswith("probe_")})
    return {
        "question": question, "model": model, "mode": mode, "kind": kind,
        "answer": rec["greedy"]["text"],
        "identify": {"p_hallucination": round(p, 4), "level": level,
                     "semantic_entropy": None if feats.get("se_norm") is None else round(feats["se_norm"], 4)},
        "classify": {"type": type_, "subtype": subtype, "type_probs": probs, "overconfidence_risk": overconfident},
        "signals": {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in signals.items()},
        "recommended_action": recommended_action(level, type_, subtype, overconfident),
        "cost": {"calls": rec["cost"]["calls"],
                 "tokens": rec["cost"]["prompt_tokens"] + rec["cost"]["completion_tokens"],
                 "seconds": round(time.perf_counter() - start, 2)},
    }
