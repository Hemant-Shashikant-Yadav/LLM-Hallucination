"""
Objective 2 interventions (PLAN §9.3): the type of uncertainty is whichever intervention
removes it.

For each uncertain or wrong answer of a focus model, the same model is asked again with
5 samples:
- with oracle evidence (a passage that contains the answer's facts), and
- with a reasoning hint (the first half of the reference method, never the answer).
An intervention "fixes" the question when the majority answer becomes correct and the
samples agree: normalised semantic entropy at or below the model's dev-tuned 'sure'
threshold, or at most one dissenting sample (80% agreement).

Two passes, so models are loaded once each: generation with the model under test
(interventions_raw.jsonl), then NLI and the judge (interventions.jsonl). Both resumable.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import pandas as pd
from loguru import logger

from uncertainty import thresholds
from uncertainty.framework.taxonomy import in_scope, intervention_label
from uncertainty.generate.backends import Backend
from uncertainty.generate.prompts import answer_prompt, with_evidence, with_hint
from uncertainty.generate.runner import max_tokens_for
from uncertainty.label.correctness import grade_text
from uncertainty.label.judge import Judge, gather_chunked
from uncertainty.methods.nli import nli_matrix, nli_texts
from uncertainty.methods.semantic_entropy import se_normalized
from uncertainty.methods.view import View
from uncertainty.answers import cluster_keys, read_answer
from uncertainty.roster import spec
from uncertainty.sets.load import Item
from uncertainty.settings import settings
from uncertainty.store import append_jsonl, model_dir, read_jsonl


MIN_AGREEMENT = 0.8


def raw_path(results_dir: Path, model: str) -> Path:
    return model_dir(results_dir, model) / "interventions_raw.jsonl"


def final_path(results_dir: Path, model: str) -> Path:
    return model_dir(results_dir, model) / "interventions.jsonl"


def load_interventions(results_dir: Path, model: str) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(final_path(results_dir, model))}


def candidates(scores: pd.DataFrame, items: dict[str, Item], low: float) -> list[Item]:
    out = []
    for r in scores.itertuples(index=False):
        item = items.get(r.id)
        if item is None or not (item.evidence or item.hint):
            continue
        if item.gold_type is not None:
            continue  # metadata already gives these a type label
        if in_scope(r.wrong, r.se_norm, low):
            out.append(item)
    return out


async def generate_interventions(backend: Backend, model: str, todo: list[Item], results_dir: Path,
                                 n: int | None = None) -> int:
    """Pass 1: samples under each available intervention, with the model under test."""
    n = n or settings.intervention_samples
    path = raw_path(results_dir, model)
    done = {r["id"] for r in read_jsonl(path)}
    todo = [it for it in todo if it.id not in done]
    think = spec(model).think
    gate = asyncio.Semaphore(settings.item_concurrency)

    async def one(item: Item) -> dict:
        async with gate:
            _, base = answer_prompt(item.kind, item.question)
            rec = {"id": item.id, "set": item.set, "kind": item.kind}
            for name, text, wrap in (("evidence", item.evidence, with_evidence), ("hint", item.hint, with_hint)):
                if not text:
                    continue
                gens = await asyncio.gather(*[
                    backend.generate(model, wrap(base, text), temperature=settings.sample_temperature,
                                     max_tokens=max_tokens_for(item.kind), seed=settings.seed_base + 500 + i,
                                     logprobs=False, think=think)
                    for i in range(n)])
                rec[name] = [g.text for g in gens if g.text.strip() and not g.truncated]
            return rec

    logger.info("{} | interventions: {} questions to re-ask", model, len(todo))
    count = 0
    for coro in asyncio.as_completed([one(it) for it in todo]):
        append_jsonl(path, await coro)
        count += 1
        if count % 25 == 0:
            logger.info("{} | interventions {}/{}", model, count, len(todo))
    return count


def _majority(item: Item, texts: list[str], nli_scorer) -> tuple[str | None, float, float | None]:
    """(representative answer of the largest meaning cluster, its share, normalised entropy)."""
    if not texts:
        return None, 0.0, None
    greedy = {"text": texts[0], "logprobs": []}
    samples = [{"text": t} for t in texts]
    if item.kind == "short":
        nli = nli_matrix(nli_scorer, nli_texts(item.question, [texts[0], *texts]))
        view = View(kind=item.kind, question=item.question, greedy=greedy, samples=samples, nli=nli)
    else:
        keys = cluster_keys(item.kind, [read_answer(item.kind, t, item.question) for t in [texts[0], *texts]])
        view = View(kind=item.kind, question=item.question, greedy=greedy, samples=samples, keys=keys)
    ids = view.clusters()
    top, count = Counter(ids).most_common(1)[0]
    rep = texts[ids.index(top)]
    return rep, count / len(texts), se_normalized(view) if len(texts) >= 2 else 0.0


async def resolve_interventions(backend: Backend, model: str, items: dict[str, Item], results_dir: Path,
                                judge_model: str, nli_scorer) -> int:
    """Pass 2: NLI clusters, the judge on each majority answer, and the type label."""
    low = thresholds.load(results_dir, model)["low"]
    done = load_interventions(results_dir, model)
    raw = [r for r in read_jsonl(raw_path(results_dir, model)) if r["id"] not in done and r["id"] in items]
    judge = Judge(backend, judge_model, results_dir / "judge_cache" / f"{judge_model.replace(':', '_')}.json")

    async def one(rec: dict) -> dict:
        item = items[rec["id"]]
        out = {"id": item.id, "set": item.set, "kind": item.kind, "threshold_low": low}
        resolved = {}
        for name in ("evidence", "hint"):
            if name not in rec:
                resolved[name] = None
                continue
            rep, share, se = _majority(item, rec[name], nli_scorer)
            verdict = await grade_text(judge, item, rep) if rep else {"correct": False}
            # Settled: the majority is right and the samples agree (sure by the model's own
            # threshold, or at most one dissent: with 5 samples a 4-1 split is already ~0.31)
            agree = se is not None and (se <= low + 1e-9 or share >= MIN_AGREEMENT)
            ok = bool(verdict["correct"]) and agree
            out[name] = {"n": len(rec[name]), "majority_share": share, "se_norm": se,
                         "majority_correct": verdict["correct"], "resolved": ok}
            resolved[name] = ok
        out["label"] = intervention_label(resolved["evidence"], resolved["hint"])
        return out

    rows = await gather_chunked([one(r) for r in raw])
    for row in rows:
        append_jsonl(final_path(results_dir, model), row)
    judge.save()
    logger.info("{} | interventions resolved: {}", model, dict(Counter(r["label"] for r in rows)))
    return len(rows)
