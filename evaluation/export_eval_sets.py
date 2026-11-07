"""
Frozen copies of the full test sets, for reviewers and reruns.

Writes data/eval_sets/full/<set>.jsonl (one question per line: id, question, answer
key, source fields) and MANIFEST.json (source dataset, pinned Hugging Face revision,
split, licence, count, SHA-256 of each file, and which items the reported results
used). The small sets written for this project are defined in code and listed there.

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
OUT = ROOT / "data" / "eval_sets" / "full"

MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
JEEBENCH_REVISION = "d070d1113ce365df05903763a2017ec2d0d56118"
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]
JEE_KIND = {"MCQ": "choice", "MCQ(multiple)": "multi", "Integer": "value", "Numeric": "value"}

# Sets written for this project: defined in code, not exported (see their files)
OTHER_SETS = {
    "difficulty_ladder": "evaluation/ladder_items.json (80: MATH L1-5, ARC, StrategyQA, MMLU; "
                         "8 per rung, seed 7, pinned revisions in evaluation/ladder_sets.py)",
    "hard_sets": "evaluation/hard_sets.py (74; keys verified with sympy or brute force)",
    "mcq_10th": "evaluation/mcq_10th.py (15)",
    "smoke_general": "evaluation/smoke_general.py (16)",
}


def _write(name: str, rows: list[dict], meta: dict, manifest: dict) -> None:
    path = OUT / f"{name}.jsonl"
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    manifest[name] = {**meta, "file": path.relative_to(ROOT).as_posix(), "count": len(rows),
                      "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def main() -> None:
    """Whole test splits: GSM8K, TruthfulQA, MATH (5,000), MATH-500, JEEBench (JEE Advanced)."""
    from datasets import load_dataset

    from app.answer_utils import final_expression
    from evaluation.benchmark_runner import GSM8K_REVISION, TRUTHFULQA_REVISION

    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    _write("gsm8k_test", load_gsm8k(10_000, 0),
           {"source": "openai/gsm8k", "config": "main", "split": "test",
            "revision": GSM8K_REVISION, "license": "MIT",
            "reported_results_used": "items 0-99 (test), 500-539 (dev)"}, manifest)
    _write("truthfulqa", load_truthfulqa(10_000, 0),
           {"source": "truthfulqa/truthful_qa", "config": "generation", "split": "validation",
            "revision": TRUTHFULQA_REVISION, "license": "Apache-2.0",
            "reported_results_used": "items 0-99 (test), 500-539 (dev)"}, manifest)
    math_rows = []
    for subject in MATH_SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test",
                          revision=REVISIONS["EleutherAI/hendrycks_math"])
        for i, r in enumerate(ds):
            math_rows.append({"id": f"math-{subject}-{i}", "subject": subject, "level": r["level"],
                              "question": r["problem"], "kind": "expr",
                              "key": final_expression(r["solution"]), "solution": r["solution"]})
    _write("math_test", math_rows,
           {"source": "EleutherAI/hendrycks_math (all 7 subjects)", "split": "test",
            "revision": REVISIONS["EleutherAI/hendrycks_math"], "license": "MIT",
            "note": "key = last boxed answer of the reference solution"}, manifest)
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=MATH500_REVISION)
    _write("math500", [{"id": f"m500-{r['unique_id']}", "subject": r["subject"], "level": r["level"],
                        "question": r["problem"], "kind": "expr", "key": r["answer"]} for r in m500],
           {"source": "HuggingFaceH4/MATH-500", "split": "test", "revision": MATH500_REVISION,
            "license": "MIT (from MATH)"}, manifest)
    jee = load_dataset("daman1209arora/jeebench", split="test", revision=JEEBENCH_REVISION)
    rows = []
    for r in jee:
        kind = JEE_KIND[r["type"]]
        rows.append({"id": f"jee-{r['subject']}-{r['index']}", "subject": r["subject"], "type": r["type"],
                     "exam": r["description"], "question": r["question"], "kind": kind,
                     "key": r["gold"].lower() if kind in ("choice", "multi") else r["gold"]})
    _write("jeebench", rows,
           {"source": "daman1209arora/jeebench (JEE Advanced)", "split": "test",
            "revision": JEEBENCH_REVISION, "license": "MIT"}, manifest)
    manifest["other_sets_defined_in_code"] = OTHER_SETS
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    for name, m in manifest.items():
        if "count" in m:
            print(f"{name:24} {m['count']:5}  {m['file']}")


if __name__ == "__main__":
    main()
