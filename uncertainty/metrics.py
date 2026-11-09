"""
How well an uncertainty score predicts mistakes (PLAN §8.2).

Convention: `unc` is an uncertainty score (higher = less sure); `wrong` is 1 for a wrong
answer. Selective-prediction curves keep the most certain answers first; ties are averaged
(the expected value under random tie-breaking), which matters for coarse scores such as the
number of clusters.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def _arrays(unc, wrong) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(unc, dtype=float)
    w = np.asarray(wrong, dtype=float)
    keep = ~np.isnan(u)
    return u[keep], w[keep]


def auroc(unc, wrong) -> float | None:
    from sklearn.metrics import roc_auc_score

    u, w = _arrays(unc, wrong)
    if len(set(w.tolist())) < 2:
        return None
    return float(roc_auc_score(w, u))


def auprc(unc, wrong) -> float | None:
    from sklearn.metrics import average_precision_score

    u, w = _arrays(unc, wrong)
    if len(set(w.tolist())) < 2:
        return None
    return float(average_precision_score(w, u))


def _expected_correct_by_rank(unc, correct) -> np.ndarray:
    """Correctness of answers sorted most-certain first, with tied scores averaged."""
    u = np.asarray(unc, dtype=float)
    c = np.asarray(correct, dtype=float)
    order = np.argsort(u, kind="mergesort")
    u, c = u[order], c[order]
    out = c.copy()
    start = 0
    for i in range(1, len(u) + 1):
        if i == len(u) or u[i] != u[start]:
            out[start:i] = c[start:i].mean()
            start = i
    return out


def accuracy_curve(unc, correct) -> np.ndarray:
    """Accuracy of the r most certain answers, r = 1..n."""
    e = _expected_correct_by_rank(unc, correct)
    return np.cumsum(e) / np.arange(1, len(e) + 1)


def aurac(unc, wrong) -> float | None:
    """Area under the rejection-accuracy curve (Farquhar et al., 2024)."""
    u, w = _arrays(unc, wrong)
    if len(u) == 0:
        return None
    return float(accuracy_curve(u, 1 - w).mean())


def accuracy_at_coverage(unc, wrong, coverage: float) -> float | None:
    u, w = _arrays(unc, wrong)
    if len(u) == 0:
        return None
    r = max(1, math.ceil(coverage * len(u)))
    return float(accuracy_curve(u, 1 - w)[r - 1])


def prr(unc, wrong) -> float | None:
    """Prediction rejection ratio: share of the ideal refusal gain achieved (1 = oracle, 0 = random)."""
    u, w = _arrays(unc, wrong)
    if len(set(w.tolist())) < 2:
        return None
    risk = lambda scores: (1 - accuracy_curve(scores, 1 - w)).mean()
    random_ = w.mean()
    oracle = risk(w)  # wrong answers ranked least certain
    method = risk(u)
    return float((random_ - method) / (random_ - oracle)) if random_ != oracle else None


def ece(conf, correct, bins: int = 15) -> float | None:
    """Expected calibration error of a confidence in [0, 1]."""
    c, y = _arrays(conf, correct)
    if len(c) == 0:
        return None
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(y[sel].mean() - c[sel].mean())
    return float(total)


def brier(conf, correct) -> float | None:
    c, y = _arrays(conf, correct)
    return float(((c - y) ** 2).mean()) if len(c) else None


def reliability(conf, correct, bins: int = 10) -> list[dict]:
    c, y = _arrays(conf, correct)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1], right=True), 0, bins - 1)
    out = []
    for b in range(bins):
        sel = idx == b
        if sel.any():
            out.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]), "n": int(sel.sum()),
                        "confidence": float(c[sel].mean()), "accuracy": float(y[sel].mean())})
    return out


def bootstrap_ci(stat: Callable[..., float | None], *arrays, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float | None, float | None]:
    """Percentile CI of stat(*arrays), resampling questions (paired across the arrays)."""
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    n = len(arrs[0])
    if n == 0:
        return None, None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat(*[a[idx] for a in arrs])
        if v is not None and not math.isnan(v):
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return None, None
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def bootstrap_auroc(unc, wrong, n_boot: int = 1000, seed: int = 0,
                    alpha: float = 0.05) -> tuple[float | None, float | None]:
    """Percentile CI of the AUROC, vectorised (rank formula), for the many tables of Objective 1."""
    from scipy.stats import rankdata

    u, w = _arrays(unc, wrong)
    n = len(u)
    if n < 4 or len(set(w.tolist())) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    ranks = rankdata(u[idx], axis=1)
    ws = w[idx]
    m = ws.sum(axis=1)
    k = n - m
    ok = (m > 0) & (k > 0)
    aucs = ((ranks * ws).sum(axis=1)[ok] - m[ok] * (m[ok] + 1) / 2) / (m[ok] * k[ok])
    if len(aucs) < n_boot * 0.5:
        return None, None
    return float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))


# ---------------------------------------------------------------------------- DeLong


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong(unc_a, unc_b, wrong) -> dict | None:
    """
    DeLong test for two correlated AUROCs on the same questions (fast version, Sun & Xu 2014).
    Returns both AUROCs, the difference, z and the two-sided p-value.
    """
    from scipy.stats import norm

    a, b, w = (np.asarray(x, dtype=float) for x in (unc_a, unc_b, wrong))
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b, w = a[keep], b[keep], w[keep]
    m = int(w.sum())
    k = len(w) - m
    if m < 2 or k < 2:
        return None
    order = np.argsort(-w, kind="mergesort")
    preds = np.vstack([a, b])[:, order]
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / k - (m + 1.0) / 2.0 / k
    v01 = (tz[:, :m] - tx) / k
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / k
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    diff = float(aucs[0] - aucs[1])
    if var <= 1e-15:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": 0.0, "p": 1.0, "n": len(w)}
    z = diff / math.sqrt(var)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": z,
            "p": float(2 * norm.sf(abs(z))), "n": len(w)}


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as given)."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted
