"""
Shared answer-parsing helpers.

Used by Layer 1 (clustering numeric answers), Layer 3 (comparing the draft
with the verified chain) and the evaluation harness (GSM8K scoring).
"""

from __future__ import annotations

import re

_NUMBER = r"-?\$?\s?\d[\d,]*(?:\.\d+)?"

# Explicit answer markers, strongest first
_MARKER_PATTERNS = [
    re.compile(r"####\s*(" + _NUMBER + ")"),
    re.compile(r"final answer\s*(?:is|:)?\s*[:=]?\s*\**\s*(" + _NUMBER + ")", re.IGNORECASE),
    re.compile(r"answer\s*(?:is|:)\s*\**\s*(" + _NUMBER + ")", re.IGNORECASE),
]
_ANY_NUMBER = re.compile(_NUMBER)


def normalize_number(raw: str) -> str | None:
    """Normalize '$1,234.50' → '1234.5', '18.0' → '18'. Returns None if not numeric."""
    cleaned = raw.replace(",", "").replace("$", "").replace(" ", "").rstrip(".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def extract_final_number(text: str) -> str | None:
    """
    Extract the final numeric answer from a free-form response.

    Prefers explicit markers ("#### 18", "Final answer: 18", "the answer is 18");
    otherwise falls back to the last number in the text.
    """
    if not text:
        return None
    for pattern in _MARKER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            value = normalize_number(matches[-1])
            if value is not None:
                return value
    numbers = _ANY_NUMBER.findall(text)
    for raw in reversed(numbers):
        value = normalize_number(raw)
        if value is not None:
            return value
    return None


_REASONING_CUES = re.compile(
    r"\b(how many|how much|how long|how far|how old|what is the total|calculate|"
    r"what percentage|what fraction)\b",
    re.IGNORECASE,
)


def is_numeric_reasoning_query(prompt: str) -> bool:
    """
    Task-type router: a query is treated as multi-step quantitative reasoning
    when it contains at least two numeric quantities, or one quantity plus a
    quantitative question cue.
    """
    quantities = re.findall(r"\d[\d,]*(?:\.\d+)?", prompt)
    if len(quantities) >= 2:
        return True
    return len(quantities) >= 1 and bool(_REASONING_CUES.search(prompt))


def leading_sentences(text: str, max_sentences: int = 2, max_chars: int = 400) -> str:
    """Return the first sentences of a response (the core claim) for NLI comparison."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    head = " ".join(sentences[:max_sentences])
    return head[:max_chars]
