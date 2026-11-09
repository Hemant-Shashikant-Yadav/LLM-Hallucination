"""
Identify: a calibrated probability that an answer is a hallucination (PLAN §9.5).

Logistic regression over the Objective 1 scores (missing scores imputed, with indicators),
Platt-calibrated, trained on the dev split of every model pooled and evaluated on the test
split. Compared with the best single method (chosen on dev) and with semantic entropy.
Two variants: 'full' (K = 10, every signal) and 'lite' (K = 4, no extra calls).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from uncertainty import metrics
from uncertainty.methods.registry import METHODS, PRIMARY
from uncertainty.score import LITE_METHODS

FEATURES = {
    "full": [m.name for m in METHODS],
    "lite": [f"lite__{m}" for m in LITE_METHODS],
}
LOW_ACCURACY = 0.9
HIGH_ERROR = 0.5


def make_model() -> CalibratedClassifierCV:
    base = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, C=1.0)),
    ])
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)


def _xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    x = df.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    return x, df["wrong"].to_numpy(dtype=int)


def level_thresholds(p: np.ndarray, wrong: np.ndarray) -> dict:
    """low: largest p below which ≥ 90% are right; high: smallest p above which ≥ 50% are wrong."""
    order = np.sort(np.unique(np.round(p, 3)))
    low, high = 0.2, 0.6
    for t in order:
        sel = p <= t
        if sel.sum() >= 10 and 1 - wrong[sel].mean() >= LOW_ACCURACY:
            low = float(t)
    for t in order[::-1]:
        sel = p >= t
        if sel.sum() >= 10 and wrong[sel].mean() >= HIGH_ERROR:
            high = float(t)
    return {"low": low, "high": max(high, low)}


def level(p: float, t: dict) -> str:
    return "low" if p <= t["low"] else "high" if p >= t["high"] else "medium"


def evaluate(df: pd.DataFrame, variant: str, n_boot: int) -> tuple[dict, dict]:
    """Returns (results for the report, the final fitted model bundle)."""
    feats = FEATURES[variant]
    d = df[df["wrong"].notna()]
    dev, test = d[d["split"] == "dev"], d[d["split"] == "test"]
    if len(dev) < 30 or len(test) < 10 or min(dev["wrong"].value_counts().reindex([0, 1], fill_value=0)) < 10:
        return {"variant": variant, "skipped": True, "n_train": len(dev), "n_test": len(test)}, {}
    x_dev, y_dev = _xy(dev, feats)
    x_test, y_test = _xy(test, feats)
    model = make_model().fit(x_dev, y_dev)
    p = model.predict_proba(x_test)[:, 1]

    prefix = "lite__" if variant == "lite" else ""
    candidates = [f for f in feats if f in dev and dev[f].notna().sum() >= 10]
    dev_auc = {f: metrics.auroc(pd.to_numeric(dev[f], errors="coerce"), y_dev) or 0 for f in candidates}
    best = max(dev_auc, key=dev_auc.get) if dev_auc else f"{prefix}{PRIMARY}"
    se_col = f"{prefix}{PRIMARY}"
    best_u = pd.to_numeric(test[best], errors="coerce").to_numpy(float)
    se_u = pd.to_numeric(test[se_col], errors="coerce").to_numpy(float)

    lo, hi = metrics.bootstrap_auroc(p, y_test, n_boot=n_boot)
    result = {
        "variant": variant, "n_train": len(dev), "n_test": len(test),
        "auroc": metrics.auroc(p, y_test), "auroc_lo": lo, "auroc_hi": hi,
        "aurac": metrics.aurac(p, y_test), "ece": metrics.ece(p, y_test), "brier": metrics.brier(p, y_test),
        "best_single": best, "best_single_auroc": metrics.auroc(best_u, y_test),
        "best_single_aurac": metrics.aurac(best_u, y_test),
        "se_auroc": metrics.auroc(se_u, y_test), "se_aurac": metrics.aurac(se_u, y_test),
        "delong_vs_best": metrics.delong(p, best_u, y_test),
        "delong_vs_se": metrics.delong(p, se_u, y_test),
        "per_model": {}, "leave_one_model_out": {},
    }
    for m, g in test.assign(_p=p).groupby("model", sort=False):
        result["per_model"][m] = {"n": len(g), "auroc": metrics.auroc(g["_p"], g["wrong"]),
                                  "se_auroc": metrics.auroc(pd.to_numeric(g[se_col], errors="coerce"), g["wrong"])}
    models = list(dict.fromkeys(d["model"]))
    if len(models) > 1:
        for m in models:
            tr, te = dev[dev["model"] != m], test[test["model"] == m]
            if len(tr) < 30 or len(te) < 10 or tr["wrong"].nunique() < 2:
                continue
            fit = make_model().fit(*_xy(tr, feats))
            result["leave_one_model_out"][m] = metrics.auroc(fit.predict_proba(_xy(te, feats)[0])[:, 1], te["wrong"])

    # Level thresholds from out-of-fold predictions on dev (never from test)
    oof = cross_val_predict(make_model(), x_dev, y_dev, cv=5, method="predict_proba")[:, 1]
    levels = level_thresholds(oof, y_dev)
    result["levels"] = levels
    result["level_rates"] = {
        name: {"share": float(np.mean(sel)), "error_rate": float(y_test[sel].mean()) if sel.any() else None}
        for name, sel in (("low", p <= levels["low"]), ("medium", (p > levels["low"]) & (p < levels["high"])),
                          ("high", p >= levels["high"]))
    }
    bundle = {"model": model, "features": feats, "levels": levels, "variant": variant, "trained_on": "dev split"}
    return result, bundle


def predict(bundle: dict, features: dict) -> tuple[float, str]:
    x = pd.DataFrame([{f: features.get(f) for f in bundle["features"]}]).apply(pd.to_numeric, errors="coerce")
    p = float(bundle["model"].predict_proba(x)[:, 1][0])
    return p, level(p, bundle["levels"])
