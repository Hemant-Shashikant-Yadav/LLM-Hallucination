"""
Uncertainty thresholds, tuned on each model's dev split only (PLAN §5).

On normalised semantic entropy (se_norm, 0 = all samples agree, 1 = all differ):
- low:  the largest threshold below which dev answers are right ≥ 90% of the time
        ("sure"; errors below it are the blind spot, confident hallucinations).
- high: the smallest threshold above which dev answers are wrong ≥ 50% of the time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uncertainty.store import model_dir, read_json, write_json

LOW_ACCURACY = 0.9
HIGH_ERROR = 0.5
MIN_ITEMS = 10
DEFAULTS = {"low": 0.0, "high": 0.5}


def tune(dev: pd.DataFrame) -> dict:
    d = dev.dropna(subset=["se_norm", "correct"])
    if len(d) < MIN_ITEMS:
        return {**DEFAULTS, "n_dev": len(d), "tuned": False}
    values = np.sort(d["se_norm"].unique())
    low = DEFAULTS["low"]
    for t in values:
        sel = d[d["se_norm"] <= t + 1e-9]
        if len(sel) >= MIN_ITEMS and sel["correct"].mean() >= LOW_ACCURACY:
            low = float(t)
    high = DEFAULTS["high"]
    for t in values[::-1]:
        sel = d[d["se_norm"] >= t - 1e-9]
        if len(sel) >= MIN_ITEMS and (1 - sel["correct"].mean()) >= HIGH_ERROR:
            high = float(t)
    high = max(high, low)
    return {"low": low, "high": high, "n_dev": len(d), "tuned": True}


def path(results_dir: Path, model: str) -> Path:
    return model_dir(results_dir, model) / "thresholds.json"


def tune_model(results_dir: Path, model: str, scores: pd.DataFrame) -> dict:
    t = tune(scores[scores["split"] == "dev"])
    write_json(path(results_dir, model), t)
    return t


def load(results_dir: Path, model: str) -> dict:
    return read_json(path(results_dir, model), None) or {**DEFAULTS, "tuned": False}
