"""
Is each greedy answer right? (PLAN §7)

Worked answers are graded by rules (`answers.rule_correct`, the pipeline's readers); a yes/no
answer that never says yes or no is read by the judge. Free-text answers are graded by the
cross-family judge, with a model-free alias match recorded beside it. A second judge from a
third family re-grades a fixed subset, for the agreement check.

Writes results/<model>/labels.jsonl (and labels_second.jsonl); resumable.
"""

from __future__ import annotations

import random
import re
import string
from pathlib import Path

from loguru import logger

from uncertainty.answers import rule_correct
from uncertainty.generate.backends import Backend
from uncertainty.label.judge import Judge, gather_chunked
from uncertainty.sets.load import SETS, Item
from uncertainty.store import append_jsonl, load_generations, model_dir, read_jsonl


def normalize(text: str) -> str:
    """SQuAD-style normalisation: lower case, no punctuation, articles or extra spaces."""
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def alias_match(answer: str, aliases: list[str]) -> bool | None:
    if not aliases:
        return None
    norm = f" {normalize(answer)} "
    return any(f" {normalize(a)} " in norm for a in aliases if normalize(a))


def labels_path(results_dir: Path, model: str, second: bool = False) -> Path:
    return model_dir(results_dir, model) / ("labels_second.jsonl" if second else "labels.jsonl")


def load_labels(results_dir: Path, model: str, second: bool = False) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(labels_path(results_dir, model, second))}


async def grade_text(judge: Judge, item: Item, text: str) -> dict:
    """Correctness of any answer text to an item (greedy answers and intervention answers)."""
    if item.kind != "short":
        verdict = rule_correct(item.kind, text, item.question, item.answers[0])
        if verdict is not None:
            return {"correct": verdict, "source": "rule"}
        said = await judge.yes_no(item.question, text)
        return {"correct": None if said is None else said == item.answers[0], "source": "judge_yesno"}
    v = await judge.grade(item.question, text, item.answers, item.judge_mode, item.incorrect_answers)
    return {"correct": v["correct"], "informative": v.get("informative"), "source": f"judge_{item.judge_mode}"}


async def label_model(backend: Backend, model: str, items: dict[str, list[Item]], results_dir: Path,
                      judge_model: str, second_judge_model: str | None = None, second_n: int = 300) -> dict:
    """Label every generated greedy answer of a model; returns counts and the judge failure rate."""
    out_path = labels_path(results_dir, model)
    done = load_labels(results_dir, model)
    judge = Judge(backend, judge_model, results_dir / "judge_cache" / f"{judge_model.replace(':', '_')}.json")
    todo: list[tuple[Item, dict]] = []
    for set_name in [s for s in SETS if s in items]:
        by_id = {it.id: it for it in items[set_name]}
        for rec in load_generations(results_dir, model, set_name):
            if rec["id"] in by_id and rec["id"] not in done:
                todo.append((by_id[rec["id"]], rec))
    logger.info("{} | labelling {} answers (judge {})", model, len(todo), judge_model)

    async def one(item: Item, rec: dict) -> dict:
        text = rec["greedy"]["text"]
        verdict = await grade_text(judge, item, text)
        return {"id": item.id, "set": item.set, "split": item.split, "kind": item.kind,
                **verdict, "judge": judge_model if verdict["source"] != "rule" else None,
                "alias_match": alias_match(text, item.answers) if item.kind == "short" else None}

    labelled = await gather_chunked([one(it, rec) for it, rec in todo])
    for row in labelled:
        append_jsonl(out_path, row)
    judge.save()

    stats = {"new": len(labelled), "judge_calls": judge.calls, "judge_failures": judge.failures,
             "judge_failure_rate": judge.failures / judge.calls if judge.calls else 0.0}
    if second_judge_model:
        stats["second"] = await _second_judge(backend, model, items, results_dir, second_judge_model, second_n)
    return stats


async def _second_judge(backend: Backend, model: str, items: dict[str, list[Item]], results_dir: Path,
                        judge_model: str, n: int) -> dict:
    """Re-grade a fixed, seeded subset of judge-graded answers with a judge from a third family."""
    first = load_labels(results_dir, model)
    done = load_labels(results_dir, model, second=True)
    by_id = {it.id: it for its in items.values() for it in its}
    pool = sorted(i for i, r in first.items() if r["source"].startswith("judge_") and i in by_id)
    random.Random(f"second-{model}").shuffle(pool)
    chosen = [i for i in pool[:n] if i not in done]
    if not chosen:
        return {"new": 0}
    records = {}
    for set_name in {by_id[i].set for i in chosen}:
        records.update({r["id"]: r for r in load_generations(results_dir, model, set_name)})
    judge = Judge(backend, judge_model, results_dir / "judge_cache" / f"{judge_model.replace(':', '_')}.json")

    async def one(item_id: str) -> dict:
        verdict = await grade_text(judge, by_id[item_id], records[item_id]["greedy"]["text"])
        return {"id": item_id, "set": by_id[item_id].set, **verdict, "judge": judge_model}

    rows = await gather_chunked([one(i) for i in chosen if i in records])
    for row in rows:
        append_jsonl(labels_path(results_dir, model, second=True), row)
    judge.save()
    return {"new": len(rows), "judge_failures": judge.failures}
