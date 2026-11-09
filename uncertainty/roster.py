"""
Model roster (PLAN §4): which raw models are measured, which judge grades each, and which
get the larger question set and the Objective 2 interventions.

Design: several families, a size ladder inside each family, old vs new generations. The
roster is picked from the Mac's memory; `--models` overrides it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    family: str
    size_b: float
    generation: str
    min_memory_gb: int = 16       # unified memory the Mac needs to run it with parallel slots
    think: bool | None = None     # False for thinking models, so samples stay short
    optional: bool = False        # dropped first when --max-hours has to trim

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9.]+", "_", self.tag)


ROSTER: list[ModelSpec] = [
    ModelSpec("qwen2.5:3b", "qwen", 3, "qwen2.5"),
    ModelSpec("qwen2.5:7b", "qwen", 7, "qwen2.5"),
    ModelSpec("qwen2.5:14b", "qwen", 14, "qwen2.5", min_memory_gb=24, optional=True),
    ModelSpec("qwen3:4b-instruct-2507-q4_K_M", "qwen", 4, "qwen3"),
    ModelSpec("llama3.2:3b", "llama", 3, "llama3"),
    ModelSpec("llama3.1:8b", "llama", 8, "llama3"),
    ModelSpec("llama2:7b", "llama", 7, "llama2"),
    ModelSpec("gemma3:4b", "gemma", 4, "gemma3"),
    ModelSpec("gemma3:12b", "gemma", 12, "gemma3", min_memory_gb=24, optional=True),
    ModelSpec("phi4-mini", "phi", 3.8, "phi4"),
    ModelSpec("phi4", "phi", 14, "phi4", min_memory_gb=24, optional=True),
    ModelSpec("mistral:7b", "mistral", 7, "mistral", optional=True),
]

# The current pipeline model: always measured first, so Objectives 1-2 link to 3-4.
PIPELINE_MODEL = "qwen2.5:3b"
# Focus models (extended tier + interventions): the pipeline model, another family's
# largest small model, and a third family unless the pipeline gets a new model on the Mac.
FOCUS_DEFAULT = ["qwen2.5:3b", "llama3.1:8b", "gemma3:4b"]

_FAMILIES = ("qwen", "llama", "gemma", "phi", "mistral", "deepseek")


def spec(tag: str) -> ModelSpec:
    """The roster entry for a tag; an unknown tag gets family and size from its name."""
    for s in ROSTER:
        if s.tag == tag:
            return s
    low = tag.lower()
    family = next((f for f in _FAMILIES if low.startswith(f)), low.split(":")[0])
    size = re.search(r"(\d+(?:\.\d+)?)b\b", low)
    return ModelSpec(tag, family, float(size.group(1)) if size else 0.0, low.split(":")[0])


def pick_roster(memory_gb: float) -> list[ModelSpec]:
    """Every roster model that fits this much unified memory."""
    return [s for s in ROSTER if s.min_memory_gb <= memory_gb]


def focus_models(tags: list[str], pipeline_model: str | None = None) -> list[str]:
    """Focus models in run order, limited to the models actually being run."""
    focus = FOCUS_DEFAULT[:2] + [pipeline_model or FOCUS_DEFAULT[2]]
    if pipeline_model and pipeline_model not in tags:
        tags = [*tags, pipeline_model]
    return [m for m in dict.fromkeys(focus) if m in tags]


def judge_for(model: str, memory_gb: float) -> str:
    """Primary judge: never the model under test, always another family."""
    if spec(model).family == "llama":
        return "qwen2.5:14b" if memory_gb >= 24 else "qwen2.5:7b"
    return "llama3.1:8b"


def second_judge_for(model: str, memory_gb: float) -> str:
    """Second judge for the agreement check: a third family (neither the model's nor the judge's)."""
    taken = {spec(model).family, spec(judge_for(model, memory_gb)).family}
    candidates = (["gemma3:12b"] if memory_gb >= 24 else []) + ["mistral:7b", "qwen2.5:7b", "llama3.1:8b"]
    return next(c for c in candidates if spec(c).family not in taken)


def judges_needed(tags: list[str], memory_gb: float) -> list[str]:
    out: list[str] = []
    for t in tags:
        out += [judge_for(t, memory_gb), second_judge_for(t, memory_gb)]
    return list(dict.fromkeys(out))
