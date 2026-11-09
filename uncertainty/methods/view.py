"""
One question's answers, ready for the uncertainty methods.

Meaning equivalence follows Layer 1: worked answers (numbers, expressions, options, yes/no)
mean the same when their read answers match; free-text answers when they entail each other
both ways under NLI. Clusters are formed greedily: each answer joins the first cluster whose
representative it is equivalent to (Kuhn et al., 2023).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uncertainty.answers import cluster_keys, read_answer
from uncertainty.methods.nli import CONTRA, ENTAIL, valid_samples


@dataclass
class View:
    kind: str
    question: str
    greedy: dict
    samples: list[dict]                   # valid samples only
    keys: list[str | None] | None = None  # read answers of [greedy, *samples] (worked answers)
    nli: np.ndarray | None = None         # (n, n, 3) over [greedy, *samples] (free text)
    p_true: float | None = None
    verbal: float | None = None

    @property
    def m(self) -> int:
        return len(self.samples)

    @property
    def n(self) -> int:
        return 1 + len(self.samples)

    def equivalent(self) -> np.ndarray:
        """(n, n) bool: answers i and j mean the same."""
        n = self.n
        if self.keys is not None:
            k = self.keys
            eq = np.array([[i == j or (k[i] is not None and k[i] == k[j]) for j in range(n)] for i in range(n)])
            return eq
        if self.nli is not None:
            p = self.nli
            entails = p[..., ENTAIL] >= np.maximum(p[..., 1], p[..., CONTRA])
            return entails & entails.T
        return np.eye(n, dtype=bool)

    def similarity(self) -> np.ndarray:
        """(n, n) soft similarity in [0, 1]: mean two-way entailment, or 0/1 key match."""
        if self.nli is not None and self.keys is None:
            e = self.nli[..., ENTAIL]
            w = (e + e.T) / 2
            np.fill_diagonal(w, 1.0)
            return w
        return self.equivalent().astype(float)

    def clusters(self, with_greedy: bool = False) -> list[int]:
        """Cluster id of each answer: samples only (default) or greedy first then samples."""
        eq = self.equivalent()
        idx = list(range(self.n)) if with_greedy else list(range(1, self.n))
        reps: list[int] = []
        ids: list[int] = []
        for i in idx:
            for c, r in enumerate(reps):
                if eq[i, r]:
                    ids.append(c)
                    break
            else:
                reps.append(i)
                ids.append(len(reps) - 1)
        return ids

    def subset(self, k: int) -> "View":
        """The same question with only the first k samples (for the AUROC-vs-K analysis)."""
        keep = list(range(min(k, self.m) + 1))
        return View(
            kind=self.kind, question=self.question, greedy=self.greedy, samples=self.samples[:k],
            keys=None if self.keys is None else [self.keys[i] for i in keep],
            nli=None if self.nli is None else self.nli[np.ix_(keep, keep)],
            p_true=self.p_true, verbal=self.verbal,
        )


def view_from_record(rec: dict, nli: np.ndarray | None = None, keys: list | None = None) -> View:
    samples = valid_samples(rec)
    if rec["kind"] != "short" and keys is None:
        texts = [rec["greedy"]["text"]] + [s["text"] for s in samples]
        keys = cluster_keys(rec["kind"], [read_answer(rec["kind"], t, rec["question"]) for t in texts])
    verbal = rec.get("verbal", {}).get("value")
    return View(
        kind=rec["kind"], question=rec["question"], greedy=rec["greedy"], samples=samples,
        keys=keys if rec["kind"] != "short" else None,
        nli=nli if rec["kind"] == "short" else None,
        p_true=rec.get("p_true", {}).get("p"),
        verbal=None if verbal is None else min(max(verbal, 0.0), 100.0),
    )
