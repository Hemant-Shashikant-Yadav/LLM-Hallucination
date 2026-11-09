"""
Reading answers, by answer kind.

A thin layer over the pipeline's `app.answer_utils` (imported read-only): answer reading
caused the pipeline's worst bugs, and that code is already debugged on these benchmarks.
"""

from __future__ import annotations

import re

from app.answer_utils import (
    answers_equivalent,
    canonicalize,
    extract_choice,
    extract_final_answer,
    extract_yes_no,
    final_expression,
    leading_sentences,
    parse_options,
)

_LAST_YES_NO = re.compile(r"answer\s*(?:is|:)\s*\**\s*(yes|no)\b", re.IGNORECASE)


def read_yes_no(text: str) -> str | None:
    """The verdict a worked answer ends on ('… Answer: no'), else the pipeline's reader."""
    hits = _LAST_YES_NO.findall(text or "")
    return hits[-1].lower() if hits else extract_yes_no(text)


def read_answer(kind: str, text: str, question: str) -> str | None:
    """The comparable answer key of a response; None for free text or an unreadable answer."""
    if not text:
        return None
    if kind in ("number", "expr"):
        return extract_final_answer(text)
    if kind == "choice":
        return extract_choice(text, parse_options(question))
    if kind == "yesno":
        return read_yes_no(text)
    return None


def cluster_keys(kind: str, keys: list[str | None]) -> list[str | None]:
    """Keys that mean the same answer share one value (0.5 = 1/2); None stays None."""
    return canonicalize(keys) if kind in ("number", "expr") else list(keys)


def rule_correct(kind: str, text: str, question: str, key: str) -> bool | None:
    """
    Correctness of a worked answer by rules (as evaluation/hard_eval.py scores).
    None when it cannot be decided by rules: a yes/no answer that never says yes or no.
    """
    if kind in ("number", "expr"):
        candidate = final_expression(text) or text
        return answers_equivalent(candidate, key, symbolic=(kind == "expr"))
    if kind == "choice":
        return extract_choice(text, parse_options(question)) == key
    if kind == "yesno":
        verdict = read_yes_no(text)
        return None if verdict is None else verdict == key
    raise ValueError(f"rule_correct does not grade kind {kind!r}")


def brief(kind: str, text: str, question: str) -> str:
    """A short form of an answer, for prompts that list several answers (P(True))."""
    key = read_answer(kind, text, question)
    if kind in ("number", "expr"):
        return key or leading_sentences(text, 1, 160)
    if kind == "choice":
        return f"({key})" if key else leading_sentences(text, 1, 160)
    if kind == "yesno":
        return key or leading_sentences(text, 1, 160)
    return leading_sentences(text, 2, 300)
