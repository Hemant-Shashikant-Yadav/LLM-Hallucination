"""
Meaning equivalence for free-text answers: DeBERTa-v3 NLI, the model Layer 1 uses.

One full matrix per question over [greedy, valid samples...]: M[i, j] = NLI probabilities
(entailment, neutral, contradiction) with answer i as premise and answer j as hypothesis,
each answer prefixed with the question (Kuhn et al., 2023). Computed once, saved per set,
and shared by every NLI-based method.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from loguru import logger

from app.answer_utils import leading_sentences
from uncertainty.settings import settings

ENTAIL, NEUTRAL, CONTRA = 0, 1, 2


def valid_samples(record: dict) -> list[dict]:
    """Samples that count as answers: not empty and not cut off at the token limit."""
    return [s for s in record["samples"] if s["text"].strip() and s.get("done_reason") != "length"]


def nli_texts(question: str, texts: list[str]) -> list[str]:
    return [f"{question} {leading_sentences(t)}" for t in texts]


def _device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NLIScorer:
    """Batched DeBERTa-v3 cross-encoder → (n, 3) probabilities [entail, neutral, contradiction]."""

    name = "deberta"

    def __init__(self, model_name: str | None = None, device: str | None = None, batch: int | None = None) -> None:
        self.model_name = model_name or settings.nli_model
        self.device = _device(device or settings.nli_device)
        self.batch = batch or settings.nli_batch
        self._model = None
        self._tok = None
        self._order: list[int] = []

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading NLI model {} on {}", self.model_name, self.device)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device).eval()
        labels = {i: str(l).lower() for i, l in self._model.config.id2label.items()}
        index = {name: i for i, name in labels.items()}
        self._order = [index["entailment"], index["neutral"], index["contradiction"]]
        if self.device != "cpu":
            try:  # some operators are missing on some GPU back ends (e.g. older MPS)
                self.score_pairs([("A cat sat.", "A cat sat.")])
            except Exception as exc:
                logger.warning("NLI on {} failed ({}); using the CPU", self.device, exc)
                self.device = "cpu"
                self._model = self._model.to("cpu")

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        import torch

        self.load()
        out = []
        for start in range(0, len(pairs), self.batch):
            chunk = pairs[start:start + self.batch]
            enc = self._tok([p for p, _ in chunk], [h for _, h in chunk], padding=True, truncation=True,
                            max_length=256, return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self._model(**enc).logits.float(), dim=-1).cpu().numpy()
            out.append(probs[:, self._order])
        return np.concatenate(out) if out else np.zeros((0, 3))


class FakeNLI:
    """For dry runs: answers mean the same when the simulated model's stated answers match."""

    name = "fake"
    _ANSWER = re.compile(r"(?:answer is|It is)\s+(.+?)(?:[.,](?:\s|$)|$)", re.IGNORECASE)

    def load(self) -> None:
        return None

    def _key(self, text: str) -> str:
        m = self._ANSWER.search(text)
        return (m.group(1) if m else text).strip().lower()

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        rows = [[0.92, 0.05, 0.03] if self._key(p) == self._key(h) else [0.04, 0.16, 0.80] for p, h in pairs]
        return np.array(rows, dtype=float).reshape(-1, 3)


def nli_matrix(scorer, texts: list[str]) -> np.ndarray:
    n = len(texts)
    m = np.zeros((n, n, 3), dtype=np.float32)
    m[np.arange(n), np.arange(n)] = [1.0, 0.0, 0.0]
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    if pairs:
        probs = scorer.score_pairs([(texts[i], texts[j]) for i, j in pairs])
        for (i, j), p in zip(pairs, probs):
            m[i, j] = p
    return m


def nli_path(results_dir: Path, model_slug: str, set_name: str) -> Path:
    return results_dir / model_slug / "nli" / f"{set_name}.npz"


def save_matrices(path: Path, matrices: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(matrices)
    arrays = {f"m{i}": matrices[k].astype(np.float16) for i, k in enumerate(ids)}
    np.savez_compressed(path, ids=np.array(json.dumps(ids)), **arrays)


def load_matrices(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path) as data:
        ids = json.loads(str(data["ids"]))
        return {k: data[f"m{i}"].astype(np.float32) for i, k in enumerate(ids)}


def build_set_matrices(scorer, records: list[dict], path: Path) -> int:
    """NLI matrices for the free-text questions of one set; returns how many were computed."""
    have = load_matrices(path)
    todo = [r for r in records if r["kind"] == "short" and r["id"] not in have]
    if not todo:
        return 0
    for n, rec in enumerate(todo, start=1):
        texts = [rec["greedy"]["text"]] + [s["text"] for s in valid_samples(rec)]
        have[rec["id"]] = nli_matrix(scorer, nli_texts(rec["question"], texts))
        if n % 100 == 0:
            logger.info("NLI {}: {}/{}", path.stem, n, len(todo))
    save_matrices(path, have)
    return len(todo)
