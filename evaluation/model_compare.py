"""
Raw-model comparison for choosing the pipeline model.

Each candidate answers the same questions the way the pipeline drafts them (the
question alone, no system prompt, greedy, 2,048 tokens), so accuracy here is the
"raw model" column the framework then has to improve on. qwen2.5:3b's drafts are
already stored, so it is scored from those files instead of being re-run.

Sets: GSM8K dev items 500-539 (40), the hard sets (74), the difficulty ladder (80).

Usage:
  python -m evaluation.model_compare collect --model qwen3:4b-instruct-2507-q4_K_M
  python -m evaluation.model_compare collect --model qwen3.5:4b-q4_K_M --no-think
  python -m evaluation.model_compare score
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import pandas as pd

from app.answer_utils import extract_final_number, numbers_match

EVAL = Path(__file__).resolve().parent
OUT_DIR = EVAL / "results" / "model_compare"


def items() -> list[dict]:
    """Every question with its scoring kind and key, in a fixed order."""
    from evaluation import hard_eval

    rows = []
    dev = pd.read_csv(EVAL / "results/dev/qwen3b_v4" / "results_gsm8k.csv")
    for r in dev[dev.config == "baseline"].itertuples():
        rows.append({"set": "gsm8k_dev", "id": r.sample_id, "category": "GSM8K dev",
                     "kind": "gsm8k", "key": str(r.numerical_answer), "question": r.prompt})
    for set_name in ("hard", "ladder"):
        hard_eval._SET = set_name
        for category, entries in hard_eval._sets().items():
            for item_id, question, kind, key in entries:
                rows.append({"set": set_name, "id": item_id, "category": category,
                             "kind": kind, "key": key, "question": question})
    return rows


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", model)


async def collect(model: str, think: bool | None) -> None:
    from ollama import AsyncClient

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{_slug(model)}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    client = AsyncClient(timeout=900)
    extra = {} if think is None else {"think": think}
    todo = [it for it in items() if it["id"] not in done]
    for n, it in enumerate(todo, start=1):
        t0 = time.perf_counter()
        resp = await client.generate(
            model=model, prompt=it["question"], stream=False,
            options={"temperature": 0.0, "num_predict": 2048, "num_ctx": 4096}, **extra,
        )
        row = {**it, "model": model, "answer": resp.response,
               "tokens": resp.eval_count or 0, "done_reason": resp.done_reason,
               "seconds": round(time.perf_counter() - t0, 1)}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{n}/{len(todo)}] {it['category']:20} {it['id']:24} {row['tokens']:5} tok {row['seconds']:5.1f}s",
              flush=True)


def _stored_qwen25() -> list[dict]:
    """qwen2.5:3b's drafts from the runs already on disk (the raw column of each)."""
    by_id = {it["id"]: it for it in items()}
    rows = []
    dev = pd.read_csv(EVAL / "results/dev/qwen3b_v4" / "results_gsm8k.csv")
    for r in dev[dev.config == "baseline"].itertuples():
        rows.append({**by_id[r.sample_id], "answer": str(r.answer), "seconds": r.wall_latency_ms / 1000,
                     "tokens": r.tokens_generated})
    for path in ("results/hard_sets_and_ladder/hard_final.jsonl", "results/hard_sets_and_ladder/ladder_v2.jsonl"):
        for line in (EVAL / path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows.append({**by_id[r["id"]], "answer": r["draft"], "seconds": None, "tokens": None})
    return rows


def _correct(row: dict, judged: dict) -> bool:
    from evaluation.hard_eval import is_correct

    if row["kind"] == "gsm8k":
        return bool(numbers_match(extract_final_number(row["answer"]), row["key"]))
    return is_correct(row, row["answer"], judged)


def score() -> None:
    from evaluation.hard_eval import _CACHE, _explicit_yes_no, _judge_yes_no

    runs = {"qwen2.5:3b (stored drafts)": _stored_qwen25()}
    for path in sorted(OUT_DIR.glob("*.jsonl")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        runs[rows[0]["model"]] = rows
    # Yes/no answers that never say yes or no go to the fixed judge (cached by text)
    unresolved = [(r["question"], r["answer"]) for rows in runs.values() for r in rows
                  if r["kind"] == "yesno" and _explicit_yes_no(r["answer"]) is None]
    judged = asyncio.run(_judge_yes_no(unresolved)) if unresolved else {}
    if not judged and _CACHE.exists():
        judged = json.loads(_CACHE.read_text(encoding="utf-8"))

    table = {}
    for model, rows in runs.items():
        cats: dict[str, list[bool]] = {}
        for r in rows:
            cats.setdefault(r["category"], []).append(_correct(r, judged))
        secs = [r["seconds"] for r in rows if r.get("seconds")]
        toks = [r["tokens"] for r in rows if r.get("tokens")]
        cut = sum(1 for r in rows if r.get("done_reason") == "length")
        table[model] = {"by_category": {c: f"{sum(v)}/{len(v)}" for c, v in cats.items()},
                        "total": f"{sum(sum(v) for v in cats.values())}/{sum(len(v) for v in cats.values())}",
                        "mean_seconds": round(sum(secs) / len(secs), 1) if secs else None,
                        "mean_tokens": round(sum(toks) / len(toks)) if toks else None,
                        "cut_off": cut}
    (OUT_DIR / "summary.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    cats = list(next(iter(table.values()))["by_category"])
    print(f"{'category':22}" + "".join(f"{m[:26]:>28}" for m in table))
    for c in cats:
        print(f"{c[:22]:22}" + "".join(f"{t['by_category'].get(c, '-'):>28}" for t in table.values()))
    for k in ("total", "mean_seconds", "mean_tokens", "cut_off"):
        print(f"{k:22}" + "".join(f"{str(t[k]):>28}" for t in table.values()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["collect", "score"])
    ap.add_argument("--model")
    ap.add_argument("--no-think", action="store_true", help="send think=false (thinking models)")
    args = ap.parse_args()
    if args.mode == "collect":
        asyncio.run(collect(args.model, False if args.no_think else None))
    else:
        score()


if __name__ == "__main__":
    main()
