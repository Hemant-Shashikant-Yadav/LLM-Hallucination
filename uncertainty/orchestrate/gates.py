"""
Checks that stop a bad run early (PLAN §1.1), instead of after days of useless output.

- Readability: worked answers (numbers, expressions, options, yes/no) must give a readable
  final answer in at least 90% of samples, or the prompts or the answer reader are wrong.
- Cut-off: at most 10% of samples may hit the token limit.
- Empty answers: at most 10% of greedy answers may be empty.
- Judge: more than 2% of judge calls without a verdict is recorded as a degradation (those
  answers are left unlabelled); more than 20% means the judge is broken (not pulled, no
  JSON) and fails the gate. A single failure in a small batch must not throw away a model's
  finished generation, so the hard limit is deliberately loose.
"""

from __future__ import annotations

from uncertainty.answers import read_answer
from uncertainty.sets.load import STRUCTURED

MIN_READABLE = 0.9
MAX_CUT_OFF = 0.1
MAX_EMPTY = 0.1
WARN_JUDGE_FAILURES = 0.02
MAX_JUDGE_FAILURES = 0.2


class GateFailure(RuntimeError):
    pass


def generation_stats(records: list[dict]) -> dict:
    structured = [r for r in records if r["kind"] in STRUCTURED]
    readable = total = 0
    for r in structured:
        for s in r["samples"]:
            if s["text"].strip() and s.get("done_reason") != "length":
                total += 1
                readable += read_answer(r["kind"], s["text"], r["question"]) is not None
    samples = [s for r in records for s in r["samples"]]
    return {
        "questions": len(records),
        "readable_share": readable / total if total else None,
        "cut_off_share": sum(s.get("done_reason") == "length" for s in samples) / len(samples) if samples else 0.0,
        "empty_greedy_share": sum(not r["greedy"]["text"].strip() for r in records) / len(records) if records else 0.0,
    }


def check_generation(records: list[dict], where: str) -> dict:
    st = generation_stats(records)
    problems = []
    if st["readable_share"] is not None and st["readable_share"] < MIN_READABLE:
        problems.append(f"only {st['readable_share']:.0%} of worked samples have a readable final answer "
                        f"(need {MIN_READABLE:.0%}): check the prompts and answer reading")
    if st["cut_off_share"] > MAX_CUT_OFF:
        problems.append(f"{st['cut_off_share']:.0%} of samples hit the token limit (max {MAX_CUT_OFF:.0%}): "
                        "raise UQ_MAX_TOKENS_* or check for a thinking model without think=false")
    if st["empty_greedy_share"] > MAX_EMPTY:
        problems.append(f"{st['empty_greedy_share']:.0%} of greedy answers are empty")
    if problems:
        raise GateFailure(f"{where}: " + "; ".join(problems))
    return st


def check_judge(stats: dict, where: str) -> str | None:
    """Raises when the judge is broken; returns a warning to record when it failed more than 2%."""
    rate = stats.get("judge_failure_rate", 0.0)
    if stats.get("judge_calls", 0) >= 10 and rate > MAX_JUDGE_FAILURES:
        raise GateFailure(f"{where}: the judge failed on {rate:.1%} of calls (max {MAX_JUDGE_FAILURES:.0%}); "
                          "check the judge model is pulled and answers in JSON")
    if rate > WARN_JUDGE_FAILURES:
        return (f"{where}: the judge gave no verdict on {stats.get('judge_failures')} of {stats.get('judge_calls')} "
                f"calls ({rate:.1%}); those answers are unlabelled")
    return None
