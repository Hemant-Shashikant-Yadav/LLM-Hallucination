"""
Live smoke test: basic web/factual and math questions through the running API.

Acceptance test for the general-question requirement (smoke list in doc/06_roadmap.md):
every question must return HTTP 200, take the expected route, contain the expected
answer, and finish without degradations.

Usage:
  python -m evaluation.smoke_general --api-base http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import httpx

from app.answer_utils import extract_final_number, numbers_match

# (question, expected task_type, check) — check is a number (compared with the final
# answer), a list of keywords (any must appear), or None (just a sensible answer).
SMOKE = [
    ("What is the capital of Australia?", "factual", ["canberra"]),
    ('Who wrote "Pride and Prejudice"?', "factual", ["austen"]),
    ("What happened on July 20, 1969?", "factual", ["moon", "apollo"]),
    ("What was India's population in 2011?", "factual",
     ["1.21", "1.2 billion", "121", "1,210", "1210"]),
    ("Which is taller, Mount Everest (8849 m) or K2 (8611 m)?", "factual", ["everest"]),
    ("Who is the current Prime Minister of the UK?", "time_sensitive", None),
    ("Explain how vaccines work.", "factual", ["immune"]),
    ("What is 15% of 240?", "math", "36"),
    ("What is three times seven?", "math", "21"),
    ("What is the square root of 144?", "math", "12"),
    ("Solve 2x + 3 = 11.", "math", "4"),
    ("Simplify 3/4 + 1/8.", "math", "0.875"),
    ("Convert 100 °F to Celsius.", "math", "37.78"),
    ("What is the derivative of x^2?", "math", ["2x", "2 x", "2*x"]),
    ("A train travels 60 km/h for 2.5 hours. How far does it go?", "math", "150"),
    ("Tom has 3 boxes of 12 eggs and breaks 5. How many eggs are left?", "math", "31"),
]


def _check(answer: str, check) -> bool:
    if check is None:
        return len(answer.strip()) > 10
    if isinstance(check, str):
        return numbers_match(extract_final_number(answer), check)
    low = answer.lower()
    return any(k in low for k in check)


async def main(api_base: str) -> int:
    failures = 0
    async with httpx.AsyncClient(timeout=900) as client:
        for question, route, check in SMOKE:
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    f"{api_base}/api/v1/query", json={"prompt": question, "include_audit": True}
                )
                resp.raise_for_status()
                body = resp.json()
            except Exception as exc:
                failures += 1
                print(f"FAIL  {question}\n      request error: {exc}")
                continue
            pr = body.get("pipeline_result") or {}
            answer = body.get("answer", "")
            problems = []
            if pr.get("task_type") != route:
                problems.append(f"route {pr.get('task_type')!r} != {route!r}")
            if not _check(answer, check):
                problems.append(f"answer check failed ({check})")
            if pr.get("degradations"):
                problems.append(f"degraded: {pr['degradations']}")
            status = "OK  " if not problems else "FAIL"
            failures += bool(problems)
            print(f"{status}  {question}  [{pr.get('task_type')} → {pr.get('layers_invoked')}, "
                  f"{time.perf_counter() - t0:.0f}s]")
            print(f"      {answer.strip()[:160]!r}")
            for p in problems:
                print(f"      ! {p}")
    print(f"\n{len(SMOKE) - failures}/{len(SMOKE)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-base", default="http://127.0.0.1:8000")
    sys.exit(asyncio.run(main(ap.parse_args().api_base)))
