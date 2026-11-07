"""
Frozen copy of every test set used in the results, for reviewers and reruns.

Writes data/eval_sets/<set>.jsonl (one question per line: id, question, answer key,
source fields) and data/eval_sets/MANIFEST.json (source dataset, pinned version,
split, item range, count, SHA-256 of each file). GSM8K and TruthfulQA are read at
the pinned Hugging Face revisions also used by benchmark_runner; the other sets come
from the repo files that define them.

Usage:
  python -m evaluation.export_eval_sets
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.benchmark_runner import load_gsm8k, load_truthfulqa
from evaluation.ladder_sets import REVISIONS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "eval_sets"


def _write(name: str, rows: list[dict], meta: dict, manifest: dict) -> None:
    path = OUT / f"{name}.jsonl"
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    manifest[name] = {**meta, "file": f"data/eval_sets/{path.name}", "count": len(rows),
                      "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def main() -> None:
    from evaluation.benchmark_runner import GSM8K_REVISION, TRUTHFULQA_REVISION
    from evaluation.hard_sets import SETS
    from evaluation.mcq_10th import MCQS
    from evaluation.smoke_general import SMOKE

    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    gsm = {"source": "openai/gsm8k", "config": "main", "split": "test",
           "revision": GSM8K_REVISION, "license": "MIT"}
    tqa = {"source": "truthfulqa/truthful_qa", "config": "generation", "split": "validation",
           "revision": TRUTHFULQA_REVISION, "license": "Apache-2.0"}
    for name, loader, meta, n, offset in [
        ("gsm8k_test_0-99", load_gsm8k, gsm, 100, 0),
        ("gsm8k_dev_500-539", load_gsm8k, gsm, 40, 500),
        ("truthfulqa_test_0-99", load_truthfulqa, tqa, 100, 0),
        ("truthfulqa_dev_500-539", load_truthfulqa, tqa, 40, 500),
    ]:
        _write(name, loader(n, offset), {**meta, "items": f"{offset}-{offset + n - 1}"}, manifest)

    ladder = json.loads((ROOT / "evaluation" / "ladder_items.json").read_text(encoding="utf-8"))
    _write("difficulty_ladder",
           [{"id": i, "rung": rung, "question": q, "kind": k, "key": key}
            for rung, entries in ladder.items() for i, q, k, key in entries],
           {"source": "Hendrycks MATH, ARC-Easy/Challenge, StrategyQA, MMLU (logical_fallacies, "
                      "formal_logic); 8 per rung, seed 7 (evaluation/ladder_sets.py)",
            "revision": REVISIONS, "license": "MIT; ARC CC-BY-SA-4.0"}, manifest)
    _write("hard_sets",
           [{"id": i, "category": cat, "question": q, "kind": k, "key": key}
            for cat, entries in SETS.items() for i, q, k, key in entries],
           {"source": "written for this project; keys verified with sympy or brute force "
                      "(evaluation/hard_sets.py verify_keys)"}, manifest)
    _write("mcq_10th", [{"topic": t, "question": q, "key": k, "worked": w} for t, q, k, w in MCQS],
           {"source": "written for this project (evaluation/mcq_10th.py)"}, manifest)
    _write("smoke_general",
           [{"question": q, "expected_route": route, "answer_check": check} for q, route, check in SMOKE],
           {"source": "written for this project (evaluation/smoke_general.py)"}, manifest)

    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    for name, m in manifest.items():
        print(f"{name:24} {m['count']:4}  {m['file']}")


if __name__ == "__main__":
    main()
