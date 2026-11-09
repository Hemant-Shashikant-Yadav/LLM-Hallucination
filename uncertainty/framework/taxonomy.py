"""
The uncertainty taxonomy (PLAN §9.2) and where each answer's type label comes from (§9.3).

Two axes:
- Identify: how unsure (a calibrated hallucination probability and a low/medium/high level).
- Classify: what kind of uncertainty, which says what would fix it.

Type labels, in order of precedence: a human label; the dataset's own metadata (AmbigQA →
ambiguous, SelfAware unanswerable → unanswerable, FreshQA → false premise or temporal); the
interventions ("the type is whichever intervention removes the uncertainty"). Answers that
are right and sure have no uncertainty to classify.
"""

from __future__ import annotations

import pandas as pd

from uncertainty.sets.load import ALEATORIC_SUBTYPES, Item

TYPES = ["knowledge", "reasoning", "aleatoric", "temporal"]

DESCRIPTIONS = {
    "knowledge": "The model lacks or misremembers a fact; evidence settles it.",
    "reasoning": "The facts are known or given, but the derivation is unstable; a method settles it.",
    "aleatoric": "The question has no single true answer (ambiguous, unanswerable, false premise).",
    "temporal": "The true answer changes over time; the model's knowledge may be out of date.",
}

ACTIONS = {
    "knowledge": "retrieve_evidence",          # what Layer 2 does
    "reasoning": "structured_reasoning",       # what Layer 3 does
    "aleatoric": "clarify_or_abstain",
    "temporal": "fresh_retrieval",
}
SUBTYPE_ACTIONS = {
    "ambiguous": "ask_to_clarify_or_list_readings",
    "unanswerable": "abstain",
    "false_premise": "correct_the_premise",
}


def recommended_action(level: str, type_: str, subtype: str | None, overconfident: bool) -> str:
    if level == "low" and not overconfident:
        return "accept"
    if type_ == "aleatoric" and subtype in SUBTYPE_ACTIONS:
        return SUBTYPE_ACTIONS[subtype]
    return ACTIONS[type_]


def intervention_label(evidence: bool | None, hint: bool | None) -> str:
    """Resolved by evidence only → knowledge; by the hint only → reasoning. None = not available.
    When a set offers only one intervention, it decides alone."""
    if evidence is None and hint is None:
        return "not_applicable"
    if evidence and hint:
        return "mixed"
    if evidence:
        return "knowledge"
    if hint:
        return "reasoning"
    return "unresolved"


def in_scope(wrong, se_norm, low: float) -> bool:
    """Is there uncertainty to classify? A wrong answer, or samples that disagree."""
    return bool(wrong == 1) or (se_norm is not None and not pd.isna(se_norm) and se_norm > low + 1e-9)


def assemble(scores: pd.DataFrame, items: dict[str, Item], thresholds: dict[str, dict],
             interventions: dict[tuple[str, str], dict], human: dict[tuple[str, str], dict] | None = None
             ) -> pd.DataFrame:
    """
    One row per (model, question) in scope, with its type label (or None) and the label's source.
    `interventions` and `human` are keyed by (model, item id).
    """
    human = human or {}
    rows = []
    for r in scores.itertuples(index=False):
        low = thresholds.get(r.model, {}).get("low", 0.0)
        if not in_scope(r.wrong, r.se_norm, low):
            continue
        item = items.get(r.id)
        type_, subtype, source, note = None, None, None, None
        iv = interventions.get((r.model, r.id))
        h = human.get((r.model, r.id))
        if h and h.get("type") in TYPES:
            type_, subtype, source = h["type"], h.get("subtype") or None, "human"
        elif item is not None and item.gold_type in ALEATORIC_SUBTYPES:
            type_, subtype, source = "aleatoric", item.gold_type, "metadata"
        elif item is not None and item.gold_type == "temporal":
            type_, source = "temporal", "metadata"
        elif iv is not None and iv["label"] in ("knowledge", "reasoning"):
            type_, source = iv["label"], "intervention"
        else:
            note = iv["label"] if iv is not None else "no_intervention"
        rows.append({"model": r.model, "id": r.id, "set": r.set, "split": r.split, "type": type_,
                     "subtype": subtype, "label_source": source, "unlabelled_reason": note})
    return pd.DataFrame(rows, columns=["model", "id", "set", "split", "type", "subtype", "label_source",
                                       "unlabelled_reason"])
