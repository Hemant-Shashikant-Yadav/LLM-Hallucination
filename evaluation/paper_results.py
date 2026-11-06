"""
Every headline number for the paper, recomputed from the stored result files.

No model is called: GSM8K is re-scored with the current answer reader, the hard and
ladder sets with `hard_eval.is_correct` (yes/no verdicts from the cached judge), and
TruthfulQA comes from the stored LLM-judge summary. Writes evaluation/paper_results.json
and prints one table.

Usage:
  python -m evaluation.paper_results
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from app.answer_utils import extract_final_number, numbers_match

EVAL = Path(__file__).resolve().parent
OUT = EVAL / "paper_results.json"


def _paired(raw: list[bool], fw: list[bool]) -> dict:
    fixed = sum(f and not r for r, f in zip(raw, fw))
    broken = sum(r and not f for r, f in zip(raw, fw))
    p = binomtest(min(fixed, broken), fixed + broken, 0.5).pvalue if fixed + broken else 1.0
    return {"n": len(raw), "raw": sum(raw), "framework": sum(fw),
            "fixed": fixed, "broken": broken, "mcnemar_p": round(p, 3)}


def gsm8k(run: str) -> dict:
    df = pd.read_csv(EVAL / run / "results_gsm8k.csv")
    ok = {(r.sample_id, r.config): bool(numbers_match(extract_final_number(str(r.answer)),
                                                      str(r.numerical_answer)))
          for r in df.itertuples()}
    ids = sorted({s for s, _ in ok}, key=lambda s: int(s.split("_")[1]))
    out = _paired([ok[(s, "baseline")] for s in ids], [ok[(s, "full_framework")] for s in ids])
    fw = df[df.config == "full_framework"]
    out.update(source=f"evaluation/{run}/results_gsm8k.csv",
               framework_latency_s=round(fw.wall_latency_ms.mean() / 1000, 1),
               framework_tokens=round(fw.tokens_generated.mean()))
    if "is_correct" in df:  # the verdicts stored when the run was first scored
        by = df.groupby("config").is_correct.sum()
        out["as_first_scored"] = {"raw": int(by["baseline"]), "framework": int(by["full_framework"])}
    return out


def truthfulqa(run: str) -> dict:
    s = pd.read_csv(EVAL / run / "summary_truthfulqa.csv").set_index("config")
    return {"n": int(s.loc["baseline", "n"]),
            "raw_pct": float(s.loc["baseline", "judge_truthful_pct"]),
            "framework_pct": float(s.loc["full_framework", "judge_truthful_pct"]),
            "mcnemar_p": float(s.loc["full_framework", "judge_truthful_mcnemar_p"]),
            "rouge_raw_pct": float(s.loc["baseline", "rouge_acc_pct"]),
            "rouge_framework_pct": float(s.loc["full_framework", "rouge_acc_pct"]),
            "source": f"evaluation/{run}/summary_truthfulqa.csv (LLM judge)"}


def question_set(set_name: str, tag: str) -> dict:
    from evaluation.hard_eval import is_correct

    judged = json.loads((EVAL / "hard_results" / "yesno_judge_cache.json").read_text(encoding="utf-8"))
    path = EVAL / "hard_results" / f"{set_name}_{tag}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_cat: dict[str, tuple[list, list]] = {}
    for row in rows:
        raw, fw = by_cat.setdefault(row["category"], ([], []))
        raw.append(is_correct(row, row["draft"], judged))
        fw.append(is_correct(row, row["answer"], judged))
    out = {cat: _paired(r, f) for cat, (r, f) in by_cat.items()}
    out["ALL"] = _paired([x for r, _ in by_cat.values() for x in r], [x for _, f in by_cat.values() for x in f])
    out["source"] = f"evaluation/hard_results/{path.name}"
    return out


def mcq_10th() -> dict:
    rows = json.loads((EVAL / "mcq_10th_results.json").read_text(encoding="utf-8"))
    out = _paired([r["baseline_ok"] for r in rows], [r["framework_ok"] for r in rows])
    out["source"] = "evaluation/mcq_10th_results.json"
    return out


def smoke() -> dict:
    log = EVAL / "run_logs" / "2026-11-06_smoke_final.log"
    passed, total = log.read_text(encoding="utf-8").strip().splitlines()[-1].split()[0].split("/")
    return {"passed": int(passed), "n": int(total), "source": f"evaluation/run_logs/{log.name}"}


def main() -> None:
    results = {
        "gsm8k_test_v6": gsm8k("results_v6_qwen3b"),
        "gsm8k_test_v5": gsm8k("results_v5_qwen3b"),
        "gsm8k_dev_v3_design": gsm8k("dev_qwen3b_v3"),          # items 500-539: chose Layer 3
        "gsm8k_dev_v4_recheck": gsm8k("dev_qwen3b_v4"),         # same items, after hard-set changes
        "truthfulqa_test_v5": truthfulqa("results_v5_qwen3b"),
        "truthfulqa_dev_v2_design": truthfulqa("dev_qwen3b_v2"),  # chose evidence-gated Layer 2
        "hard_sets_final": question_set("hard", "final"),
        "hard_sets_old_framework": question_set("hard", "before"),
        "ladder_first_run": question_set("ladder", "final"),
        "ladder_v2": question_set("ladder", "v2"),
        "mcq_10th": mcq_10th(),
        "smoke_general": smoke(),
        "judge_check_phase1": json.loads(
            (EVAL / "judge_check" / "phase1_llama2_rejudged.summary.json").read_text(encoding="utf-8")),
        "phase1_llama2_gsm8k": gsm8k("results_v4_llama2_only"),
        "phase1_llama2_truthfulqa": truthfulqa("results_v4_llama2_only"),
    }
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"{'Test':34} {'N':>4} {'Raw':>7} {'Framework':>10} {'Fixed/broken':>13} {'p':>6}")
    for name, r in results.items():
        if "framework" in r and "raw" in r:
            row, n = r, r["n"]
        elif "ALL" in r:
            row, n = r["ALL"], r["ALL"]["n"]
        else:
            continue
        print(f"{name:34} {n:4} {row['raw']:7} {row['framework']:10} "
              f"{row['fixed']:>6}/{row['broken']:<6} {row['mcnemar_p']:6}")
    for name in ("truthfulqa_test_v5", "truthfulqa_dev_v2_design", "phase1_llama2_truthfulqa"):
        t = results[name]
        print(f"{name:34} {t['n']:4} {t['raw_pct']:6}% {t['framework_pct']:9}% {'':13} {t['mcnemar_p']:6}")
    print(f"smoke_general: {results['smoke_general']['passed']}/{results['smoke_general']['n']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
