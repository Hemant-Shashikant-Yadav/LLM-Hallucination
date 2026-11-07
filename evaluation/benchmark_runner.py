"""
M.Tech Thesis Benchmark Runner — Paired Multi-Configuration Evaluation.

Configurations:
  baseline           Raw generator (DID_OLLAMA_MODEL). Taken from the SAME greedy draft the
                     framework produces, so baseline and framework are exactly paired.
  full_framework     POST /api/v1/query          (Layer 1 triage → Layer 2 / Layer 3)
  naive_rag          POST /api/v1/query/layer2   (always run Layer 2)          [optional]
  reference_verifier POST /api/v1/query/layer1 with the verifier model (llama3.1:8b)
                     — the raw verifier model on its own, as an upper reference [optional]

Datasets:
  TruthfulQA (generation, validation) — scored by an independent LLM judge + ROUGE-L acc
  GSM8K      (main, test)             — strict final-answer match

Usage:
  python -m evaluation.benchmark_runner --n 100 --out-dir evaluation/results/phase1_llama2_all_configs
  python -m evaluation.benchmark_runner --judge-only --out-dir evaluation/results/phase1_llama2_all_configs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from loguru import logger

from app.answer_utils import extract_final_number, normalize_number
from app.config import settings

logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)

# ── Constants ────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"
REQUEST_TIMEOUT = 1800.0
# The pipeline model from .env (DID_OLLAMA_MODEL): one model in every layer. The Phase 1
# llama2:7b default silently ran a qwen benchmark on llama2, partly on CPU.
GENERATOR_MODEL = settings.ollama_model
VERIFIER_MODEL = "llama3.1:8b"
# Each run should get its own --out-dir under evaluation/results/; never reuse a finished one
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results" / "latest_run"

CONFIG_ENDPOINTS = {
    "full_framework": ("/api/v1/query", GENERATOR_MODEL),
    "naive_rag": ("/api/v1/query/layer2", GENERATOR_MODEL),
    "reference_verifier": ("/api/v1/query/layer1", VERIFIER_MODEL),
    "baseline": ("/api/v1/query/layer1", GENERATOR_MODEL),  # only used if not paired
}
CONFIG_ORDER = ["baseline", "naive_rag", "full_framework", "reference_verifier"]
DISPLAY_NAMES = {
    "baseline": f"Raw LLM ({GENERATOR_MODEL})",
    "naive_rag": "Naive RAG (always Layer 2)",
    "full_framework": "Defense-in-Depth (Proposed)",
    "reference_verifier": "Raw verifier (llama3.1:8b)",
}


# ── Dataset Loaders ──────────────────────────────────────────────────────────


# Pinned Hugging Face dataset versions: a rerun reads exactly the questions in the paper
# (frozen copies with checksums: data/eval_sets/full/, python -m evaluation.export_eval_sets)
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
TRUTHFULQA_REVISION = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"


def load_truthfulqa(n: int, offset: int = 0) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation", revision=TRUTHFULQA_REVISION)
    samples = []
    for i in range(offset, min(offset + n, len(ds))):
        row = ds[i]
        correct = [row["best_answer"], *row["correct_answers"]]
        samples.append({
            "id": f"tqa_{i}",
            "prompt": row["question"],
            "correct_answers": list(dict.fromkeys(c for c in correct if c)),
            "incorrect_answers": list(row["incorrect_answers"]),
            "category": row.get("category", ""),
        })
    logger.info("Loaded {} TruthfulQA samples", len(samples))
    return samples


def load_gsm8k(n: int, offset: int = 0) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test", revision=GSM8K_REVISION)
    samples = []
    for i in range(offset, min(offset + n, len(ds))):
        row = ds[i]
        gold = normalize_number(row["answer"].split("####")[-1].strip())
        samples.append({
            "id": f"gsm_{i}",
            "prompt": row["question"],
            "correct_answers": [gold] if gold else [],
            "numerical_answer": gold or "",
        })
    logger.info("Loaded {} GSM8K samples", len(samples))
    return samples


# ── Scoring ──────────────────────────────────────────────────────────────────


def eval_gsm8k(answer: str, numerical_answer: str) -> bool:
    """Strict: the extracted final number must equal the gold answer."""
    if not numerical_answer:
        return False
    return extract_final_number(answer) == numerical_answer


def eval_gsm8k_legacy(answer: str, numerical_answer: str) -> bool:
    """Previous (lenient) metric: gold number appears anywhere as a substring."""
    return bool(numerical_answer) and numerical_answer in answer.replace(",", "").replace("$", "")


# ── API Client ───────────────────────────────────────────────────────────────


async def query_api(
    client: httpx.AsyncClient, prompt: str, endpoint: str, model: str
) -> dict[str, Any]:
    """Returns the response JSON, or {"error": ...} after retries — never a fake answer."""
    body = {"prompt": prompt, "model": model, "include_audit": True}
    error = ""
    for attempt in range(2):
        try:
            resp = await client.post(f"{API_BASE}{endpoint}", json=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            error = f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        logger.warning("API {} failed (attempt {}): {}", endpoint, attempt + 1, error)
        await asyncio.sleep(5)
    return {"error": error}


def _row(
    dataset: str, sample: dict[str, Any], config: str, answer: str, pr: dict[str, Any],
    wall_ms: float, tokens: int,
) -> dict[str, Any]:
    triage = pr.get("triage_result") or {}
    ec = triage.get("error_classification") or {}
    return {
        "dataset": dataset,
        "sample_id": sample["id"],
        "prompt": sample["prompt"],
        "config": config,
        "answer": answer,
        "total_latency_ms": round(pr.get("total_latency_ms", 0.0), 1),
        "wall_latency_ms": round(wall_ms, 1),
        "tokens_generated": tokens,
        "triage_decision": triage.get("decision", ""),
        "classified_error_type": ec.get("error_type", ""),
        "semantic_entropy": triage.get("semantic_entropy", np.nan),
        "layers_invoked": pr.get("layers_invoked", ""),
        "answer_changed": pr.get("layer_answer_changed", False),
        "trace_id": pr.get("trace_id", ""),
        "degradations": json.dumps(pr.get("degradations", [])),
        "reference_answer": (sample["correct_answers"] or [""])[0],
        "correct_answers": json.dumps(sample["correct_answers"]),
        "incorrect_answers": json.dumps(sample.get("incorrect_answers", [])),
        "numerical_answer": sample.get("numerical_answer", ""),
    }


# ── Benchmark Core ───────────────────────────────────────────────────────────


async def benchmark_dataset(
    dataset: str,
    samples: list[dict[str, Any]],
    configs: list[str],
    csv_path: Path,
) -> pd.DataFrame:
    done: set[tuple[str, str]] = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        done = set(zip(prev["sample_id"], prev["config"]))
        logger.info("Resuming {} — {} rows already present", dataset, len(prev))

    paired_baseline = "baseline" in configs and "full_framework" in configs
    failures: list[tuple[str, str, str]] = []
    audit_path = csv_path.with_name(csv_path.stem + "_audit.jsonl")

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for idx, sample in enumerate(samples):
            rows: list[dict[str, Any]] = []
            logger.info("[{}/{}] {}: {:.70}", idx + 1, len(samples), dataset, sample["prompt"])

            for config in configs:
                if (sample["id"], config) in done:
                    continue
                if config == "baseline" and paired_baseline:
                    continue  # filled from the framework draft below

                endpoint, model = CONFIG_ENDPOINTS[config]
                t0 = time.perf_counter()
                resp = await query_api(client, sample["prompt"], endpoint, model)
                wall_ms = (time.perf_counter() - t0) * 1000
                if "error" in resp:
                    # Not written: a failed call is not a wrong answer. Re-running
                    # resumes and retries exactly the missing (sample, config) rows.
                    failures.append((sample["id"], config, resp["error"]))
                    logger.error("{} / {} not recorded: {}", sample["id"], config, resp["error"])
                    continue
                pr = resp.get("pipeline_result") or {}
                # Full audit trail (programs, votes, NLI verdicts) for error analysis
                with audit_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"sample_id": sample["id"], "config": config,
                                         "pipeline_result": pr}) + "\n")
                if pr.get("degradations"):
                    logger.warning(
                        "{} / {} degraded (trace {}): {}",
                        sample["id"], config, pr.get("trace_id"), pr["degradations"],
                    )

                rows.append(_row(
                    dataset, sample, config, resp.get("answer", ""), pr,
                    wall_ms, pr.get("tokens_generated", 0),
                ))

                if config == "full_framework" and paired_baseline and (
                    sample["id"], "baseline") not in done:
                    triage = pr.get("triage_result") or {}
                    draft = triage.get("draft_response", "[ERROR]")
                    rows.append(_row(
                        dataset, sample, "baseline", draft,
                        {"total_latency_ms": pr.get("draft_latency_ms", 0.0),
                         "layers_invoked": "draft_only"},
                        pr.get("draft_latency_ms", 0.0), pr.get("draft_tokens", 0),
                    ))

            if rows:
                new = pd.DataFrame(rows)
                if csv_path.exists():
                    # Resuming an older CSV: match its header so columns stay aligned
                    new = new.reindex(columns=pd.read_csv(csv_path, nrows=0).columns)
                new.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    if failures:
        logger.error(
            "{}: {} calls failed and were NOT recorded — re-run the same command to "
            "retry them. First: {}", dataset, len(failures), failures[0],
        )
    return pd.read_csv(csv_path)


# ── Scoring passes ───────────────────────────────────────────────────────────


def score_gsm8k(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gold = df["numerical_answer"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["is_correct"] = [eval_gsm8k(str(a), g) for a, g in zip(df["answer"], gold)]
    df["is_correct_legacy"] = [eval_gsm8k_legacy(str(a), g) for a, g in zip(df["answer"], gold)]
    return df


async def score_truthfulqa(df: pd.DataFrame, judge_model: str) -> pd.DataFrame:
    from ollama import AsyncClient

    from evaluation.truthfulqa_judge import judge_answer, rouge_acc

    df = df.copy()
    for col in ("judge_truthful", "judge_informative", "judge_reason"):
        if col not in df.columns:
            df[col] = None
    client = AsyncClient(timeout=600)

    todo = df.index[df["judge_truthful"].isna()].tolist()
    logger.info("Judging {} TruthfulQA answers with {}", len(todo), judge_model)
    for n, i in enumerate(todo, start=1):
        row = df.loc[i]
        verdict = await judge_answer(
            client, row["prompt"], str(row["answer"]),
            json.loads(row["correct_answers"]), json.loads(row["incorrect_answers"]),
            model=judge_model,
        )
        df.at[i, "judge_truthful"] = verdict["truthful"]
        df.at[i, "judge_informative"] = verdict["informative"]
        df.at[i, "judge_reason"] = verdict["reason"]
        if n % 20 == 0:
            logger.info("  judged {}/{}", n, len(todo))

    df["judge_truthful"] = df["judge_truthful"].astype(bool)
    df["judge_informative"] = df["judge_informative"].astype(bool)
    df["truthful_informative"] = df["judge_truthful"] & df["judge_informative"]
    df["rouge_acc"] = [
        rouge_acc(str(a), json.loads(c), json.loads(ic))
        for a, c, ic in zip(df["answer"], df["correct_answers"], df["incorrect_answers"])
    ]
    df["is_correct"] = df["judge_truthful"]
    return df


# ── Statistics & Reporting ───────────────────────────────────────────────────


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return (np.nan, np.nan)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact McNemar test on paired binary outcomes."""
    from scipy.stats import binomtest

    b_only = int(np.sum(~a & b))
    a_only = int(np.sum(a & ~b))
    if a_only + b_only == 0:
        return 1.0
    return float(binomtest(b_only, a_only + b_only, 0.5).pvalue)


def build_summary(df: pd.DataFrame, dataset: str, metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    base = df[df["config"] == "baseline"].set_index("sample_id")
    for config in [c for c in CONFIG_ORDER if c in df["config"].unique()]:
        sub = df[df["config"] == config].set_index("sample_id")
        row: dict[str, Any] = {"dataset": dataset, "config": config, "n": len(sub)}
        for m in metric_cols:
            vals = sub[m].astype(bool).to_numpy()
            lo, hi = bootstrap_ci(vals.astype(float))
            row[f"{m}_pct"] = round(100 * vals.mean(), 2)
            row[f"{m}_ci95"] = f"[{100 * lo:.1f}, {100 * hi:.1f}]"
            if config != "baseline" and len(base):
                common = sub.index.intersection(base.index)
                row[f"{m}_delta_vs_baseline"] = round(
                    100 * (sub.loc[common, m].astype(bool).mean()
                           - base.loc[common, m].astype(bool).mean()), 2)
                row[f"{m}_mcnemar_p"] = round(mcnemar_p(
                    base.loc[common, m].astype(bool).to_numpy(),
                    sub.loc[common, m].astype(bool).to_numpy()), 4)
        row["avg_latency_s"] = round(sub["wall_latency_ms"].mean() / 1000, 2)
        row["avg_tokens"] = round(sub["tokens_generated"].mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def routing_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    fw = df[df["config"] == "full_framework"].set_index("sample_id")
    base = df[df["config"] == "baseline"].set_index("sample_id")
    fw = fw.assign(baseline_correct=base["is_correct"].reindex(fw.index).astype(bool))
    return (
        fw.groupby("triage_decision")
        .agg(
            n=("is_correct", "size"),
            framework_acc=("is_correct", "mean"),
            baseline_acc=("baseline_correct", "mean"),
            changed=("answer_changed", "mean"),
            mean_entropy=("semantic_entropy", "mean"),
        )
        .round(3)
        .reset_index()
    )


def print_latex(summary: pd.DataFrame, dataset: str, metric: str) -> str:
    tbl = summary[["config", f"{metric}_pct", f"{metric}_ci95", "avg_latency_s", "avg_tokens"]].copy()
    tbl["config"] = tbl["config"].map(DISPLAY_NAMES)
    tbl.columns = ["Configuration", "Accuracy (%)", "95% CI", "Avg Latency (s)", "Avg Tokens"]
    latex = tbl.to_latex(
        index=False,
        caption=f"Benchmark Results on {dataset} (N={int(summary['n'].iloc[0])})",
        label=f"tab:{dataset.lower()}_benchmark",
        column_format="lcccc",
        escape=False,
    )
    print(latex)
    return latex


def report(out_dir: Path) -> None:
    sections = []
    for name, metrics in [("truthfulqa", ["judge_truthful", "truthful_informative", "rouge_acc"]),
                          ("gsm8k", ["is_correct", "is_correct_legacy"])]:
        path = out_dir / f"results_{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not set(metrics).issubset(df.columns):
            logger.warning("{} is not scored yet — skipping report", path)
            continue
        summary = build_summary(df, name, metrics)
        summary.to_csv(out_dir / f"summary_{name}.csv", index=False)
        print("\n" + "=" * 80 + f"\n  {name.upper()} summary\n" + "=" * 80)
        print(summary.T.to_string(header=False))
        if "full_framework" in df["config"].values and "baseline" in df["config"].values:
            rb = routing_breakdown(df)
            print("\nRouting breakdown (full_framework):")
            print(rb.to_string(index=False))
            rb.to_csv(out_dir / f"routing_{name}.csv", index=False)
        sections.append(print_latex(summary, name, metrics[0]))
    (out_dir / "latex_tables.tex").write_text("\n".join(sections))


# ── Main Entry ───────────────────────────────────────────────────────────────


async def run(args: argparse.Namespace) -> None:
    global API_BASE
    API_BASE = args.api_base
    if args.generator:
        for k, (ep, model) in CONFIG_ENDPOINTS.items():
            if model == GENERATOR_MODEL:
                CONFIG_ENDPOINTS[k] = (ep, args.generator)
        DISPLAY_NAMES["baseline"] = f"Raw LLM ({args.generator})"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.judge_only:
        async with httpx.AsyncClient(timeout=10) as hc:
            try:
                r = await hc.get(f"{API_BASE}/api/v1/health")
                r.raise_for_status()
                health = r.json()
            except Exception as exc:
                logger.error("API not reachable at {}: {}", API_BASE, exc)
                sys.exit(1)
            if not health.get("initialized"):
                logger.error("API is up but not initialized: {}", health.get("init_error"))
                sys.exit(1)
            logger.info("API healthy: {}", health.get("config"))

        for name in args.datasets:
            loader = load_truthfulqa if name == "truthfulqa" else load_gsm8k
            samples = loader(args.n, args.offset)
            await benchmark_dataset(name, samples, args.configs, out_dir / f"results_{name}.csv")

    if args.skip_scoring:
        return

    gsm_path = out_dir / "results_gsm8k.csv"
    if gsm_path.exists():
        score_gsm8k(pd.read_csv(gsm_path)).to_csv(gsm_path, index=False)

    tqa_path = out_dir / "results_truthfulqa.csv"
    if tqa_path.exists():
        df = await score_truthfulqa(pd.read_csv(tqa_path), args.judge_model)
        df.to_csv(tqa_path, index=False)

    report(out_dir)
    logger.info("✅ Benchmark complete → {}", out_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="Samples per dataset")
    ap.add_argument("--offset", type=int, default=0, help="Start index in each dataset")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "truthfulqa"],
                    choices=["gsm8k", "truthfulqa"])
    ap.add_argument("--configs", nargs="+", default=["baseline", "full_framework"],
                    choices=list(CONFIG_ENDPOINTS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--judge-model", default="qwen2.5:7b")
    ap.add_argument("--api-base", default=API_BASE, help="Framework server URL")
    ap.add_argument("--generator", default=None, help="Override generator model for all configs")
    ap.add_argument("--judge-only", action="store_true", help="Only score + report existing CSVs")
    ap.add_argument("--skip-scoring", action="store_true", help="Only generate answers")
    return ap.parse_args(argv)


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
