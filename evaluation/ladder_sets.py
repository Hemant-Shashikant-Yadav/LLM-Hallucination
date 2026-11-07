"""
Difficulty ladders: how the framework's gain over the raw model changes as questions
get harder.

  maths (Layer 3)            MATH benchmark (Hendrycks et al., 2021), official Levels 1-5
  critical thinking (Layer 2) ARC-Easy → ARC-Challenge → StrategyQA → MMLU logical
                             fallacies → MMLU formal logic

Items are sampled once with a fixed seed and cached in ladder_items.json, so every run
(and every later model) answers exactly the same questions.

Usage:
  python -m evaluation.ladder_sets            # build / show the cache
"""

from __future__ import annotations

import json
import random
from pathlib import Path

CACHE = Path(__file__).resolve().with_name("ladder_items.json")
PER_RUNG = 8
# Pinned Hugging Face dataset versions (the frozen items are in ladder_items.json)
REVISIONS = {
    "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "ChilleD/StrategyQA": "705562638fe1d8ca6bb98c66fc8f94d45fda8c83",
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
}
SEED = 7
MATH_SUBJECTS = ["algebra", "counting_and_probability", "number_theory", "prealgebra"]
LETTERS = "abcdefgh"


def _boxed(solution: str) -> str | None:
    start = solution.rfind("\\boxed{")
    if start < 0:
        return None
    pos, depth = start + len("\\boxed"), 0
    for end in range(pos, len(solution)):
        depth += {"{": 1, "}": -1}.get(solution[end], 0)
        if depth == 0:
            return solution[pos + 1:end]
    return None


def _mcq(stem: str, choices: list[str]) -> str:
    return stem.strip() + "\n" + "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(choices))


def build() -> dict[str, list]:
    import sympy as sp
    from datasets import load_dataset

    from app.answer_utils import to_sympy

    rng = random.Random(SEED)
    sets: dict[str, list] = {}

    # ── Maths ladder: MATH Levels 1-5, numeric answers only (exactly scorable) ──
    pool = []
    for subject in MATH_SUBJECTS:
        for i, row in enumerate(load_dataset("EleutherAI/hendrycks_math", subject, split="test", revision=REVISIONS["EleutherAI/hendrycks_math"])):
            if "[asy]" in row["problem"] or len(row["problem"]) > 600:
                continue  # diagrams cannot be read by a text model
            expr = to_sympy(_boxed(row["solution"]))
            if expr is None or not expr.is_number:
                continue
            pool.append((f"{subject[:4]}{i}", row["problem"], row["level"], sp.sstr(sp.nsimplify(expr))))
    for level in range(1, 6):
        rung = [p for p in pool if p[2] == f"Level {level}"]
        picked = rng.sample(rung, PER_RUNG)
        sets[f"MATH L{level}"] = [(f"M{level}-{pid}", q, "value", key) for pid, q, _, key in picked]

    # ── Critical-thinking ladder ────────────────────────────────────────────────
    def arc(cfg: str, tag: str):
        ds = load_dataset("allenai/ai2_arc", cfg, split="test", revision=REVISIONS["allenai/ai2_arc"])
        rows = [r for r in ds if len(r["choices"]["label"]) == 4]
        out = []
        for r in rng.sample(rows, PER_RUNG):
            idx = r["choices"]["label"].index(r["answerKey"])
            out.append((f"{tag}-{r['id']}", _mcq(r["question"], r["choices"]["text"]), "choice", LETTERS[idx]))
        return out

    sets["T1 ARC-Easy"] = arc("ARC-Easy", "ae")
    sets["T2 ARC-Challenge"] = arc("ARC-Challenge", "ac")
    sqa = load_dataset("ChilleD/StrategyQA", split="test", revision=REVISIONS["ChilleD/StrategyQA"])
    sets["T3 StrategyQA"] = [
        (f"sq-{r['qid']}", r["question"], "yesno", "yes" if str(r["answer"]) == "True" else "no")
        for r in rng.sample(list(sqa), PER_RUNG)
    ]
    for tag, cfg in [("T4 logical fallacies", "logical_fallacies"), ("T5 formal logic", "formal_logic")]:
        ds = load_dataset("cais/mmlu", cfg, split="test", revision=REVISIONS["cais/mmlu"])
        sets[tag] = [
            (f"{cfg[:3]}-{i}", _mcq(ds[i]["question"], ds[i]["choices"]), "choice", LETTERS[ds[i]["answer"]])
            for i in rng.sample(range(len(ds)), PER_RUNG)
        ]
    CACHE.write_text(json.dumps(sets, indent=1, ensure_ascii=False), encoding="utf-8")
    return sets


def load() -> dict[str, list]:
    if not CACHE.exists():
        return build()
    return {k: [tuple(item) for item in v] for k, v in json.loads(CACHE.read_text(encoding="utf-8")).items()}


if __name__ == "__main__":
    sets = load()
    for name, items in sets.items():
        print(f"{name:22} {len(items)} items | e.g. {items[0][1][:90]!r} → {items[0][3]}")
