"""
Features for Objective 2, all available at inference time (no gold data, no dataset name).

Computed in the scoring pass (score.py) from the same View the Objective 1 methods use:
- cluster structure: how many meanings, how dominant the largest, how confident each cluster
  is (confident-but-different clusters suggest aleatoric; scattered unlikely answers suggest
  epistemic uncertainty);
- premise vs conclusion agreement: samples that share their premises (names, intermediate
  numbers) but reach different conclusions suggest a reasoning problem;
- question kind, the pipeline's task type, and text cues (time words, hedges, refusals).
The self-diagnosis probes and Objective 1 scores are added from the record and score columns.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

import numpy as np

from app.answer_utils import classify_task_heuristic, strip_options
from uncertainty.methods.semantic_entropy import mean_logprob
from uncertainty.methods.view import View
from uncertainty.sets.load import KINDS

_FINAL_LINE = re.compile(r"(final answer|answer)\s*[:=].*$", re.IGNORECASE | re.MULTILINE)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_ENTITY = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b|\b\d+(?:\.\d+)?\b")
_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
_TIME_WORDS = re.compile(r"\b(current(?:ly)?|latest|now|today|recent(?:ly)?|this year|as of|so far|still|yet)\b", re.I)
_HEDGE = re.compile(r"\b(i'?m not sure|not certain|it depends|might|may be|possibly|probably|unclear|"
                    r"as of my (?:knowledge|last)|i believe|i think)\b", re.I)
_REFUSAL = re.compile(r"\b(cannot be (?:known|determined|answered)|impossible to (?:know|say|determine)|"
                      r"no one knows|unknown|i don'?t know|not possible to|no definitive|there is no (?:single|"
                      r"definitive|correct) answer|subjective|matter of opinion)\b", re.I)
_CORRECTION = re.compile(r"\b(there (?:is|was|are|were) no|has never|have never|never (?:been|happened)|"
                         r"actually|false premise|incorrect assumption|did not|didn'?t|is not true)\b", re.I)

STRUCTURE = ["top_share", "second_share", "greedy_in_majority", "lp_top1", "lp_top2", "confident_split",
             "lp_spread", "premise_agreement", "premise_minus_conclusion"]
CUES = ["q_has_year", "q_time_words", "a_hedge", "a_refusal", "a_correction", "q_length"]
KIND_COLS = [f"kind_{k}" for k in KINDS]
TASK_COLS = ["task_math", "task_factual", "task_time_sensitive"]
PROBE_COLS = ["probe_ambiguous", "probe_time_dependent", "probe_calculation", "probe_false_premise",
              "probe_unanswerable"]


def _premise_tokens(kind: str, text: str) -> set[str]:
    body = _FINAL_LINE.sub("", text)
    pattern = _NUMBER if kind in ("number", "expr") else _ENTITY
    return {m.group(0).lower() for m in pattern.finditer(body)}


def structure_features(v: View) -> dict[str, float]:
    nan = float("nan")
    out = dict.fromkeys(STRUCTURE, nan)
    if v.m < 1 or (v.keys is None and v.nli is None):
        return out
    ids = v.clusters()
    counts = Counter(ids).most_common()
    out["top_share"] = counts[0][1] / v.m
    out["second_share"] = counts[1][1] / v.m if len(counts) > 1 else 0.0
    eq = v.equivalent()
    members = [i for i, c in enumerate(ids) if c == counts[0][0]]
    out["greedy_in_majority"] = float(any(eq[0, 1 + i] for i in members))
    lps = [mean_logprob(s) for s in v.samples]
    if all(lp is not None for lp in lps):
        cluster_lp = {c: float(np.mean([lps[i] for i, x in enumerate(ids) if x == c])) for c, _ in counts}
        out["lp_top1"] = cluster_lp[counts[0][0]]
        out["lp_top2"] = cluster_lp[counts[1][0]] if len(counts) > 1 else out["lp_top1"]
        out["confident_split"] = min(out["lp_top1"], out["lp_top2"]) if len(counts) > 1 else nan
        out["lp_spread"] = float(np.std(lps))
    sets = [_premise_tokens(v.kind, s["text"]) for s in v.samples]
    pairs = [(a, b) for a, b in combinations(sets, 2) if a or b]
    if pairs:
        out["premise_agreement"] = float(np.mean([len(a & b) / len(a | b) for a, b in pairs]))
        out["premise_minus_conclusion"] = out["premise_agreement"] - out["top_share"]
    return out


def question_features(kind: str, question: str, greedy_text: str) -> dict[str, float]:
    task, _ = classify_task_heuristic(question)
    stem = strip_options(question)
    return {
        **{f"kind_{k}": float(kind == k) for k in KINDS},
        **{f"task_{t}": float(task == t) for t in ("math", "factual", "time_sensitive")},
        "q_has_year": float(bool(_YEAR.search(stem))),
        "q_time_words": float(bool(_TIME_WORDS.search(stem))),
        "a_hedge": float(bool(_HEDGE.search(greedy_text or ""))),
        "a_refusal": float(bool(_REFUSAL.search(greedy_text or ""))),
        "a_correction": float(bool(_CORRECTION.search(greedy_text or ""))),
        "q_length": float(len(stem.split())),
    }


def pipeline_task_type(question: str) -> str:
    """The pipeline's own routing rule as an uncertainty type: the Objective 2 baseline."""
    task, _ = classify_task_heuristic(question)
    return {"math": "reasoning", "time_sensitive": "temporal"}.get(task, "knowledge")
