"""
Hard evaluation: 12th-standard maths, JEE MCQs, real-life word problems, reasoning.

Collect answers from the running API (resumable), then score raw draft vs framework.

Usage:
  python -m evaluation.hard_eval collect --tag before
  python -m evaluation.hard_eval score --tag before
  python -m evaluation.hard_eval compare --tags before after
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

from app.answer_utils import extract_yes_no

OUT_DIR = Path(__file__).resolve().parent / "results" / "hard_sets_and_ladder"
_SET = "hard"  # 'hard' (evaluation/hard_sets.py), 'ladder', or a frozen full set below
# Full test sets frozen in data/eval_sets/full/ (python -m evaluation.export_eval_sets --full)
FULL_SETS = {"math500": "math500", "math": "math_test", "jeebench": "jeebench"}


def _sets() -> dict[str, list]:
    if _SET in FULL_SETS:
        path = Path(__file__).resolve().parent.parent / "data" / "eval_sets" / "full" / f"{FULL_SETS[_SET]}.jsonl"
        sets: dict[str, list] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            category = f"{r['subject']} {r['type']}" if _SET == "jeebench" else r["subject"]
            sets.setdefault(category, []).append((r["id"], r["question"], r["kind"], r["key"]))
        return sets
    if _SET == "ladder":
        from evaluation.ladder_sets import load
        return load()
    from evaluation.hard_sets import SETS
    return SETS


def _path(tag: str) -> Path:
    return OUT_DIR / f"{_SET}_{tag}.jsonl"


async def collect(api_base: str, tag: str, only: list[str] | None = None) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    path = _path(tag)
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    async with httpx.AsyncClient(timeout=1200) as client:
        for category, items in _sets().items():
            if only and category not in only:
                continue
            for item_id, question, kind, key in items:
                if item_id in done:
                    continue
                t0 = time.perf_counter()
                try:
                    resp = await client.post(
                        f"{api_base}/api/v1/query", json={"prompt": question, "include_audit": True}
                    )
                    resp.raise_for_status()
                    body = resp.json()
                except Exception as exc:
                    print(f"ERROR {item_id}: {exc} (not recorded; re-run to retry)")
                    continue
                pr = body.get("pipeline_result") or {}
                row = {
                    "id": item_id, "category": category, "kind": kind, "key": key,
                    "question": question,
                    "draft": (pr.get("triage_result") or {}).get("draft_response", ""),
                    "answer": body.get("answer", ""),
                    "task_type": pr.get("task_type"), "layers": pr.get("layers_invoked"),
                    "H_sem": (pr.get("triage_result") or {}).get("semantic_entropy"),
                    "degradations": pr.get("degradations"),
                    "tokens": pr.get("tokens_generated"),
                    "seconds": round(time.perf_counter() - t0, 1),
                    "traces": ((pr.get("layer3_result") or {}).get("reasoning_traces") or [])[:1],
                }
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{category:11} {item_id} {row['task_type']} → {row['layers']} {row['seconds']:.0f}s",
                      flush=True)


# ── Scoring ──────────────────────────────────────────────────────────────────


_JUDGE_MODEL = "llama3.1:8b"   # independent of the pipeline model
_CACHE = OUT_DIR / "yesno_judge_cache.json"


def _explicit_yes_no(text: str) -> str | None:
    return extract_yes_no(text)  # the pipeline's own reader: one definition of "says yes/no"


async def _judge_yes_no(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """
    Answers like 'Jane Austen was not alive when…' never say yes or no. An independent
    judge model reads each and reports which way it answers (cached by text).
    """
    from ollama import AsyncClient

    cache = json.loads(_CACHE.read_text(encoding="utf-8")) if _CACHE.exists() else {}
    client = AsyncClient(timeout=600)
    for question, answer in pairs:
        cache_key = f"{question}\n---\n{answer}"
        if cache_key in cache:
            continue
        resp = await client.generate(
            model=_JUDGE_MODEL,
            system=("You read a question that has a yes/no answer and a response to it. "
                    "Report whether the response answers YES or NO. If it does not commit, "
                    'say UNCLEAR. Respond with JSON only: {"verdict": "yes" | "no" | "unclear"}'),
            prompt=f"QUESTION: {question}\n\nRESPONSE:\n{answer[:2000]}",
            format="json",
            options={"temperature": 0, "seed": 0, "num_predict": 30},
        )
        try:
            cache[cache_key] = str(json.loads(resp.response).get("verdict", "unclear")).lower()
        except json.JSONDecodeError:
            cache[cache_key] = "unclear"
    _CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache


def _picked_letters(text: str) -> set[str]:
    r"""Letters a multi-correct answer picks: '\boxed{ABD}', 'Final answer: (A), (B) and (D)'."""
    import re

    from app.answer_utils import final_expression

    if "\\boxed{" in text:
        seg = final_expression(text) or ""  # last box, nested braces included
    else:
        marks = list(re.finditer(r"(?:correct (?:answers?|options?)|answer)\s*(?:is|are|:)", text, re.I))
        seg = text[marks[-1].end():][:120] if marks else ""
    seg = re.sub(r"\\text\{([^{}]*)\}", r"\1", seg)
    letters = re.findall(r"\(([A-Da-d])\)", seg)          # (A), (B) and (D)
    if not letters:
        run = re.search(r"(?<![A-Za-z])([A-D]{2,4})(?![A-Za-z])", seg)  # ABD
        letters = list(run.group(1)) if run else re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", seg)
    return {m.lower() for m in letters}


def is_correct(row: dict, text: str, judged: dict[str, str] | None = None) -> bool:
    from app.answer_utils import answers_equivalent, extract_choice, final_expression, parse_options

    kind, key = row["kind"], row["key"]
    if kind == "choice":
        return extract_choice(text, parse_options(row["question"])) == key
    if kind == "multi":  # JEE multi-correct: exactly the key's set of letters
        return _picked_letters(text) == set(key)
    if kind == "yesno":
        verdict = _explicit_yes_no(text) or (judged or {}).get(f"{row['question']}\n---\n{text}")
        return verdict == key
    candidate = final_expression(text) or text
    return answers_equivalent(candidate, key, symbolic=(kind == "expr"))


def score(tag: str) -> dict:
    rows = [json.loads(l) for l in _path(tag).read_text(encoding="utf-8").splitlines() if l.strip()]
    unresolved = [(r["question"], t) for r in rows if r["kind"] == "yesno"
                  for t in (r["draft"], r["answer"]) if _explicit_yes_no(t) is None]
    judged = asyncio.run(_judge_yes_no(unresolved)) if unresolved else {}
    summary: dict = {}
    for row in rows:
        row["raw_ok"] = is_correct(row, row["draft"], judged)
        row["fw_ok"] = is_correct(row, row["answer"], judged)
        s = summary.setdefault(row["category"], {"n": 0, "raw": 0, "fw": 0, "fixed": 0, "broken": 0,
                                                 "seconds": 0.0})
        s["n"] += 1
        s["raw"] += row["raw_ok"]
        s["fw"] += row["fw_ok"]
        s["fixed"] += (not row["raw_ok"]) and row["fw_ok"]
        s["broken"] += row["raw_ok"] and (not row["fw_ok"])
        s["seconds"] += row["seconds"]
    total = {k: sum(s[k] for s in summary.values()) for k in ("n", "raw", "fw", "fixed", "broken", "seconds")}
    summary["ALL"] = total
    print(f"\n{tag}: {'category':12} {'n':>3} {'raw':>7} {'framework':>10} {'fixed':>6} {'broken':>7} {'avg s':>6}")
    for cat, s in summary.items():
        print(f"{'':{len(tag) + 2}}{cat:12} {s['n']:>3} {s['raw']:>3}/{s['n']:<3} {s['fw']:>5}/{s['n']:<4} "
              f"{s['fixed']:>6} {s['broken']:>7} {s['seconds'] / max(s['n'], 1):>6.0f}")
    wrong = [r for r in rows if not r["fw_ok"]]
    if wrong:
        print("\nframework wrong:", ", ".join(f"{r['id']}" for r in wrong))
    (OUT_DIR / f"{_SET}_{tag}_scored.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["collect", "score", "compare"])
    ap.add_argument("--tag", default="before")
    ap.add_argument("--tags", nargs="+", default=["before", "after"])
    ap.add_argument("--api-base", default="http://127.0.0.1:8000")
    ap.add_argument("--set", default="hard", choices=["hard", "ladder", *FULL_SETS])
    ap.add_argument("--only", nargs="+", help="collect only these categories, e.g. 'MATH L5'")
    args = ap.parse_args()
    global _SET
    _SET = args.set
    if args.mode == "collect":
        asyncio.run(collect(args.api_base, args.tag, args.only))
    elif args.mode == "score":
        score(args.tag)
    else:
        for tag in args.tags:
            score(tag)


if __name__ == "__main__":
    sys.exit(main())
