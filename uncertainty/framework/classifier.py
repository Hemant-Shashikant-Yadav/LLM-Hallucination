"""
Classify: what kind of uncertainty an answer carries (PLAN §9.5).

Three classifiers compared on the same labelled answers:
1. baseline: the pipeline's current routing rule (maths → reasoning, time cues → temporal,
   else knowledge); it never predicts aleatoric;
2. transparent rules: a depth-3 decision tree on a few readable features, printed as rules;
3. learned: multinomial logistic regression and gradient-boosted trees.

Evaluation uses out-of-fold predictions from 5-fold cross-validation grouped by question
(the same question answered by different models never sits in two folds), plus
leave-one-dataset-out, leave-one-model-out and leave-one-family-out runs, so the classifier
is shown to read uncertainty, not dataset or model identity. A second-stage classifier
names the aleatoric subtype.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from uncertainty.framework.features import CUES, KIND_COLS, PROBE_COLS, STRUCTURE, TASK_COLS
from uncertainty.framework.taxonomy import TYPES
from uncertainty.methods.registry import METHODS
from uncertainty.score import LITE_METHODS

FEATURES = {
    "full": [m.name for m in METHODS] + ["se_norm"] + STRUCTURE + CUES + KIND_COLS + TASK_COLS + PROBE_COLS,
    "lite": [f"lite__{m}" for m in LITE_METHODS] + ["lite__se_norm"] + [f"lite__{s}" for s in STRUCTURE]
            + CUES + KIND_COLS + TASK_COLS,
}
RULE_FEATURES = ["probe_ambiguous", "probe_unanswerable", "probe_false_premise", "probe_time_dependent",
                 "probe_calculation", "premise_minus_conclusion", "top_share", "confident_split",
                 "kind_short", "task_math", "task_time_sensitive", "q_time_words", "a_refusal", "a_correction"]
SUBTYPES = ["ambiguous", "unanswerable", "false_premise"]


def make(kind: str):
    impute = ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
    if kind == "rules":
        return Pipeline([impute, ("tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=10,
                                                                 class_weight="balanced", random_state=0))])
    if kind == "logreg":
        return Pipeline([impute, ("scale", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced"))])
    if kind == "boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                                              class_weight="balanced", random_state=0)
    raise ValueError(kind)


def _x(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df.reindex(columns=features).apply(pd.to_numeric, errors="coerce")


def _folds(y: np.ndarray, groups: np.ndarray, n: int = 5):
    smallest = min(np.unique(y, return_counts=True)[1])
    n = max(2, min(n, smallest))
    return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups)


def oof_predict(kind: str, x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=object)
    for tr, te in _folds(y, groups):
        pred[te] = make(kind).fit(x.iloc[tr], y[tr]).predict(x.iloc[te])
    return pred


def macro_f1(y, pred, labels=None) -> float:
    labels = labels or sorted(set(y))
    return float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))


def _boot_macro_f1(y: np.ndarray, pred: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    """Macro-F1 over the true labels, for every bootstrap row of `idx` (labels encoded 0..k-1)."""
    n_boot = idx.shape[0]
    codes = y[idx] * (k + 1) + pred[idx]  # pred may be k ('a label the truth never has')
    cm = np.zeros((n_boot, k * (k + 1)), dtype=np.int64)
    rows = np.repeat(np.arange(n_boot), idx.shape[1])
    np.add.at(cm, (rows, codes.ravel()), 1)
    cm = cm.reshape(n_boot, k, k + 1)
    tp = cm[:, np.arange(k), np.arange(k)]
    support, predicted = cm.sum(axis=2), cm.sum(axis=1)[:, :k]
    f1 = np.where(support + predicted > 0, 2 * tp / np.maximum(support + predicted, 1), 0.0)
    return f1.mean(axis=1)


def bootstrap_f1(y, pred, n_boot: int, seed: int = 0, other=None) -> tuple[float, float]:
    """CI of macro-F1, or of macro-F1(pred) − macro-F1(other) when `other` is given (paired)."""
    labels = sorted(set(y))
    code = {lab: i for i, lab in enumerate(labels)}
    enc = lambda a: np.array([code.get(v, len(labels)) for v in a])  # noqa: E731
    yy, pp = enc(y), enc(pred)
    idx = np.random.default_rng(seed).integers(0, len(yy), (n_boot, len(yy)))
    vals = _boot_macro_f1(yy, pp, idx, len(labels))
    if other is not None:
        vals = vals - _boot_macro_f1(yy, enc(other), idx, len(labels))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def per_class(y, pred) -> list[dict]:
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=TYPES, zero_division=0)
    return [{"type": t, "precision": float(a), "recall": float(b), "f1": float(c), "support": int(d)}
            for t, a, b, c, d in zip(TYPES, p, r, f, s)]


def _held_out(kind: str, data: pd.DataFrame, x: pd.DataFrame, y: np.ndarray, column: str) -> dict:
    out = {}
    for value in data[column].unique():
        te = (data[column] == value).to_numpy()
        tr = ~te
        if te.sum() < 10 or len(set(y[tr])) < 2:
            continue
        pred = make(kind).fit(x[tr], y[tr]).predict(x[te])
        present = sorted(set(y[te]))
        out[str(value)] = {"n": int(te.sum()), "macro_f1": macro_f1(y[te], pred, present),
                           "accuracy": float(np.mean(pred == y[te])), "types_in_test": present,
                           "types_in_train": sorted(set(y[tr]))}
    return out


def evaluate(data: pd.DataFrame, variant: str, n_boot: int) -> tuple[dict, dict]:
    """`data`: labelled answers (scores + 'type', 'subtype', 'baseline', 'family')."""
    feats = FEATURES[variant]
    data = data.reset_index(drop=True)
    y = data["type"].to_numpy(dtype=object)
    groups = data["id"].to_numpy()
    x = _x(data, feats)
    result: dict = {"variant": variant, "n": len(data),
                    "type_counts": data["type"].value_counts().to_dict(),
                    "label_sources": data["label_source"].value_counts().to_dict()}
    if len(set(y)) < 2 or len(data) < 30:
        logger.warning("Objective 2 ({}): too few labelled answers ({}) to train", variant, len(data))
        result["skipped"] = True
        return result, {}

    preds = {"baseline": data["baseline"].to_numpy(dtype=object)}
    rule_feats = [f for f in RULE_FEATURES if f in feats]
    preds["rules"] = oof_predict("rules", _x(data, rule_feats), y, groups)
    for kind in ("logreg", "boosting"):
        preds[kind] = oof_predict(kind, x, y, groups)
    result["classifiers"] = {}
    for name, pred in preds.items():
        lo, hi = bootstrap_f1(y, pred, n_boot)
        result["classifiers"][name] = {
            "macro_f1": macro_f1(y, pred), "macro_f1_lo": lo, "macro_f1_hi": hi,
            "accuracy": float(np.mean(pred == y)), "per_class": per_class(y, pred),
            "confusion": confusion_matrix(y, pred, labels=TYPES).tolist(),
        }
    best = max(("logreg", "boosting"), key=lambda k: result["classifiers"][k]["macro_f1"])
    lo, hi = bootstrap_f1(y, preds[best], n_boot, other=preds["baseline"])
    result["best"] = best
    result["best_vs_baseline"] = {
        "diff": result["classifiers"][best]["macro_f1"] - result["classifiers"]["baseline"]["macro_f1"],
        "diff_lo": lo, "diff_hi": hi, "beats_baseline": lo > 0,
    }
    result["generalisation"] = {
        "leave_one_dataset_out": _held_out(best, data, x, y, "set"),
        "leave_one_model_out": _held_out(best, data, x, y, "model"),
        "leave_one_family_out": _held_out(best, data, x, y, "family"),
    }
    result["oof_best"] = preds[best].tolist()

    final = make(best).fit(x, y)
    rules = make("rules").fit(_x(data, rule_feats), y)
    result["rules_text"] = export_text(rules.named_steps["tree"], feature_names=_rule_names(rules, rule_feats))
    result["importance"] = _importance(best, data, x, y)

    bundle = {"model": final, "features": feats, "kind": best, "variant": variant,
              "rules": rules, "rule_features": rule_feats}
    sub = data[data["type"] == "aleatoric"].dropna(subset=["subtype"])
    if variant == "full" and sub["subtype"].nunique() >= 2 and len(sub) >= 20:
        ys, xs, gs = sub["subtype"].to_numpy(dtype=object), _x(sub, feats), sub["id"].to_numpy()
        pred = oof_predict("logreg", xs, ys, gs)
        result["subtype"] = {"n": len(sub), "counts": sub["subtype"].value_counts().to_dict(),
                             "macro_f1": macro_f1(ys, pred), "accuracy": float(np.mean(pred == ys)),
                             "confusion": confusion_matrix(ys, pred, labels=SUBTYPES).tolist()}
        bundle["subtype_model"] = make("logreg").fit(xs, ys)
    return result, bundle


def _rule_names(pipe: Pipeline, feats: list[str]) -> list[str]:
    imp = pipe.named_steps["impute"]
    names = list(feats)
    if getattr(imp, "indicator_", None) is not None:
        names += [f"missing_{feats[i]}" for i in imp.indicator_.features_]
    return names


def _importance(kind: str, data: pd.DataFrame, x: pd.DataFrame, y: np.ndarray) -> list[dict]:
    """Permutation importance (macro-F1) of a model fitted on dev answers, measured on test answers."""
    dev = (data["split"] == "dev").to_numpy()
    test = ~dev
    if dev.sum() < 30 or test.sum() < 30 or len(set(y[dev])) < 2:
        return []
    model = make(kind).fit(x[dev], y[dev])
    imp = permutation_importance(model, x[test], y[test], scoring="f1_macro", n_repeats=5, random_state=0)
    order = np.argsort(-imp.importances_mean)
    return [{"feature": x.columns[i], "importance": float(imp.importances_mean[i]),
             "std": float(imp.importances_std[i])} for i in order[:20]]


def predict(bundle: dict, features: dict) -> tuple[str, dict, str | None]:
    x = pd.DataFrame([{f: features.get(f) for f in bundle["features"]}]).apply(pd.to_numeric, errors="coerce")
    model = bundle["model"]
    probs = dict(zip(model.classes_, model.predict_proba(x)[0]))
    type_ = max(probs, key=probs.get)
    subtype = None
    if type_ == "aleatoric" and "subtype_model" in bundle:
        subtype = str(bundle["subtype_model"].predict(x)[0])
    return type_, {t: round(float(probs.get(t, 0.0)), 4) for t in TYPES}, subtype
