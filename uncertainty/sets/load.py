"""
Question sets: the Item schema, reading the frozen files, tiers and the dev/test split.

The files in data/uq_sets/ are written by sets/build.py. Each holds the extended tier of
one set; `core` marks the subset every model answers (PLAN §5).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from uncertainty.settings import settings

# Set name → (domain, answer kind). Order = run order and report order.
SETS: dict[str, tuple[str, str]] = {
    "triviaqa": ("Popular facts", "short"),
    "popqa": ("Long-tail facts", "short"),
    "halueval": ("Multi-hop facts", "short"),
    "truthfulqa": ("Misconceptions", "short"),
    "gsm8k": ("Arithmetic word problems", "number"),
    "math": ("Competition maths", "expr"),
    "strategyqa": ("Multi-step commonsense", "yesno"),
    "arc": ("Science MCQ", "choice"),
    "ambigqa": ("Ambiguous questions", "short"),
    "selfaware": ("Unanswerable questions", "short"),
    "freshqa": ("Changing facts, false premises", "short"),
}

KINDS = ("short", "number", "expr", "choice", "yesno")
# Kinds whose answer is read by rules (worked answers); 'short' answers need NLI and a judge
STRUCTURED = ("number", "expr", "choice", "yesno")
# Metadata uncertainty types (PLAN §9.3, source 2)
ALEATORIC_SUBTYPES = ("ambiguous", "unanswerable", "false_premise")


@dataclass
class Item:
    id: str
    set: str
    kind: str
    question: str                      # as shown to the model (options included for 'choice')
    answers: list[str]                 # gold: aliases | [key] | [letter] | ['yes' | 'no']
    split: str = "test"                # dev | test
    core: bool = True                  # in the core tier; the file itself is the extended tier
    judge_mode: str = "match"          # match | truthfulqa | abstain (how a short answer is graded)
    incorrect_answers: list[str] = field(default_factory=list)
    gold_type: str | None = None       # ambiguous | unanswerable | false_premise | temporal
    evidence: str | None = None        # oracle evidence (knowledge intervention)
    hint: str | None = None            # partial method (reasoning intervention)
    meta: dict = field(default_factory=dict)

    @property
    def structured(self) -> bool:
        return self.kind in STRUCTURED

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def set_path(name: str, sets_dir: Path | None = None) -> Path:
    return (sets_dir or settings.sets_dir) / f"{name}.jsonl"


def load_set(name: str, tier: str = "extended", sets_dir: Path | None = None) -> list[Item]:
    path = set_path(name, sets_dir)
    items = [Item(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [it for it in items if tier == "extended" or it.core]


def spread(items: list[Item], n: int) -> list[Item]:
    """n items evenly spaced through the list: files are ordered by id, and FreshQA, for one,
    lists every false-premise question first, so the first n would miss whole strata."""
    if n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def load_all(tier: str = "extended", sets: list[str] | None = None,
             sets_dir: Path | None = None, limit: int | None = None) -> dict[str, list[Item]]:
    """Every set (or the named ones) at a tier. `limit` keeps n per set, spread evenly (dry runs)."""
    out = {}
    for name in sets or SETS:
        items = load_set(name, tier, sets_dir)
        out[name] = spread(items, limit) if limit else items
    return out


def write_set(name: str, items: list[Item], sets_dir: Path | None = None) -> Path:
    path = set_path(name, sets_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(it.to_json() + "\n" for it in items), encoding="utf-8")
    return path
