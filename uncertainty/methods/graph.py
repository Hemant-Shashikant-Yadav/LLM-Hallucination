"""
Spectral uncertainty over a similarity graph of the samples (Lin, Trivedi & Sun, 2024).

W is the samples' pairwise similarity (mean two-way NLI entailment for free text, 0/1
answer match for worked answers). With degree matrix D and normalised Laplacian
L = I − D^-1/2 W D^-1/2:
  EigV = Σ_k max(0, 1 − λ_k)               (a soft count of meaning clusters)
  Deg  = 1 − mean_j(D_jj) / m               (average dissimilarity)
  Ecc  = ‖V − mean(V)‖_F, V = eigenvectors of the smallest eigenvalues (λ < 0.9), one row
         per sample (how spread the answers are in the spectral embedding)
"""

from __future__ import annotations

import numpy as np

from uncertainty.methods.semantic_entropy import has_meaning
from uncertainty.methods.view import View


def _samples_w(v: View) -> np.ndarray | None:
    if v.m < 2 or not has_meaning(v):
        return None
    return v.similarity()[1:, 1:]


def _laplacian(w: np.ndarray) -> np.ndarray:
    d = w.sum(axis=1)
    inv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return np.eye(len(w)) - (inv[:, None] * w * inv[None, :])


def eigv(v: View) -> float | None:
    w = _samples_w(v)
    if w is None:
        return None
    lam = np.linalg.eigvalsh(_laplacian(w))
    return float(np.clip(1 - lam, 0, None).sum())


def degree(v: View) -> float | None:
    w = _samples_w(v)
    if w is None:
        return None
    m = len(w)
    return float(1 - w.sum(axis=1).mean() / m)


def eccentricity(v: View) -> float | None:
    w = _samples_w(v)
    if w is None:
        return None
    lam, vec = np.linalg.eigh(_laplacian(w))
    k = max(1, int((lam < 0.9).sum()))
    emb = vec[:, :k]
    return float(np.linalg.norm(emb - emb.mean(axis=0)))
