"""
Judge consistency for TruthfulQA.

Re-grades saved answers with a second judge model (answers are not regenerated), then
reports (a) raw vs framework under each judge and (b) how often the two judges agree,
as raw agreement and Cohen's kappa. Use it to put every TruthfulQA number on one fixed
judge, and to show how much a result depends on the choice of judge.

Usage:
  python -m evaluation.judge_agreement evaluation/results_v4_llama2_only/results_truthfulqa.csv \
      --judge llama3.1:8b --out evaluation/judge_check/phase1_llama2_rejudged.csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    observed = float(np.mean(a == b))
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


async def rejudge(df: pd.DataFrame, judge: str) -> pd.DataFrame:
    from ollama import AsyncClient

    from evaluation.truthfulqa_judge import judge_answer

    client = AsyncClient(timeout=600)
    out = df.copy()
    verdicts = []
    for n, row in enumerate(df.itertuples(index=False), start=1):
        v = await judge_answer(client, row.prompt, str(row.answer), json.loads(row.correct_answers),
                               json.loads(row.incorrect_answers), model=judge)
        verdicts.append(v)
        if n % 25 == 0:
            print(f"  judged {n}/{len(df)}", flush=True)
    out["judge2_truthful"] = [v["truthful"] for v in verdicts]
    out["judge2_informative"] = [v["informative"] for v in verdicts]
    out["judge2_reason"] = [v["reason"] for v in verdicts]
    return out


def report(df: pd.DataFrame, judge1: str, judge2: str) -> dict:
    res = {}
    for col, name in [("judge_truthful", judge1), ("judge2_truthful", judge2)]:
        by = df.groupby("config")[col].apply(lambda s: round(100 * s.astype(bool).mean(), 1)).to_dict()
        res[name] = by
    a, b = df["judge_truthful"].astype(bool).to_numpy(), df["judge2_truthful"].astype(bool).to_numpy()
    res["agreement_pct"] = round(100 * float(np.mean(a == b)), 1)
    res["cohen_kappa"] = round(cohen_kappa(a, b), 3)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--judge", default="llama3.1:8b")
    ap.add_argument("--first-judge-name", default="previous judge")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        judged = pd.read_csv(out)
    else:
        judged = asyncio.run(rejudge(df, args.judge))
        judged.to_csv(out, index=False)
    summary = report(judged, args.first_judge_name, args.judge)
    print(json.dumps(summary, indent=1))
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
