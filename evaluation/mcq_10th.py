"""
10th-standard competitive-exam MCQs: permutations & combinations and circles.

Runs every question through the live API and compares the raw draft (baseline)
with the framework's answer, scoring the chosen option letter.

Usage:
  python -m evaluation.mcq_10th --api-base http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

from app.answer_utils import extract_choice, parse_options

# (topic, question with options, correct option, worked answer)
MCQS = [
    ("P&C", "In how many different ways can the letters of the word APPLE be arranged?\n"
            "(a) 120\n(b) 60\n(c) 24\n(d) 30", "b", "5!/2! = 60"),
    ("P&C", "In how many ways can 5 people be seated around a circular table?\n"
            "(a) 120\n(b) 24\n(c) 60\n(d) 25", "b", "(5-1)! = 24"),
    ("P&C", "In how many ways can a committee of 3 members be selected from 7 people?\n"
            "(a) 210\n(b) 21\n(c) 35\n(d) 42", "c", "7C3 = 35"),
    ("P&C", "How many 3-digit numbers can be formed using the digits 1, 2, 3, 4, 5 if no digit is repeated?\n"
            "(a) 125\n(b) 60\n(c) 10\n(d) 120", "b", "5 x 4 x 3 = 60"),
    ("P&C", "What is the value of 8P3?\n(a) 336\n(b) 56\n(c) 512\n(d) 24", "a", "8 x 7 x 6 = 336"),
    ("P&C", "How many diagonals does a hexagon have?\n(a) 15\n(b) 6\n(c) 9\n(d) 12", "c", "6(6-3)/2 = 9"),
    ("P&C", "In how many ways can 4 boys and 3 girls be arranged in a row so that all the girls sit together?\n"
            "(a) 5040\n(b) 720\n(c) 144\n(d) 1440", "b", "5! x 3! = 720"),
    ("P&C", "In a party, 10 people shake hands with each other exactly once. How many handshakes take place?\n"
            "(a) 90\n(b) 45\n(c) 100\n(d) 20", "b", "10C2 = 45"),
    ("Circles", "What is the area of a circle of radius 7 cm? (Take π = 22/7)\n"
                "(a) 44 cm²\n(b) 154 cm²\n(c) 308 cm²\n(d) 49 cm²", "b", "22/7 x 49 = 154"),
    ("Circles", "What is the circumference of a circle of diameter 14 cm? (Take π = 22/7)\n"
                "(a) 22 cm\n(b) 88 cm\n(c) 44 cm\n(d) 154 cm", "c", "22/7 x 14 = 44"),
    ("Circles", "A tangent PQ at a point P of a circle of radius 5 cm meets a line through the centre O "
                "at a point Q such that OQ = 13 cm. What is the length of PQ?\n"
                "(a) 12 cm\n(b) 13 cm\n(c) 8.5 cm\n(d) 18 cm", "a", "sqrt(13² - 5²) = 12"),
    ("Circles", "What is the area of a sector of angle 90° of a circle of radius 14 cm? (Take π = 22/7)\n"
                "(a) 154 cm²\n(b) 616 cm²\n(c) 308 cm²\n(d) 77 cm²", "a", "(90/360) x 22/7 x 196 = 154"),
    ("Circles", "What is the length of an arc that subtends an angle of 60° at the centre of a circle "
                "of radius 21 cm? (Take π = 22/7)\n(a) 44 cm\n(b) 22 cm\n(c) 66 cm\n(d) 11 cm", "b",
                "(60/360) x 2 x 22/7 x 21 = 22"),
    ("Circles", "A chord of length 16 cm is drawn in a circle of radius 10 cm. "
                "What is the distance of the chord from the centre?\n(a) 8 cm\n(b) 6 cm\n(c) 4 cm\n(d) 12 cm",
                "b", "sqrt(10² - 8²) = 6"),
    ("Circles", "The radii of two circles are 19 cm and 9 cm. What is the radius of the circle whose "
                "circumference is equal to the sum of the circumferences of the two circles?\n"
                "(a) 28 cm\n(b) 10 cm\n(c) 14 cm\n(d) 21 cm", "a", "2πR = 2π(19 + 9), R = 28"),
]


async def main(api_base: str, out: Path) -> int:
    rows = []
    async with httpx.AsyncClient(timeout=900) as client:
        for topic, question, correct, worked in MCQS:
            options = parse_options(question)
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    f"{api_base}/api/v1/query", json={"prompt": question, "include_audit": True}
                )
                resp.raise_for_status()
                body = resp.json()
            except Exception as exc:
                print(f"ERROR {question.splitlines()[0][:70]}: {exc}")
                rows.append({"topic": topic, "question": question, "error": str(exc)})
                continue
            pr = body.get("pipeline_result") or {}
            draft = (pr.get("triage_result") or {}).get("draft_response", "")
            base_choice = extract_choice(draft, options)
            fw_choice = extract_choice(body.get("answer", ""), options)
            row = {
                "topic": topic, "question": question.splitlines()[0], "correct": correct,
                "worked": worked, "baseline_choice": base_choice, "framework_choice": fw_choice,
                "baseline_ok": base_choice == correct, "framework_ok": fw_choice == correct,
                "route": pr.get("task_type"), "layers": pr.get("layers_invoked"),
                "H_sem": (pr.get("triage_result") or {}).get("semantic_entropy"),
                "degradations": pr.get("degradations"), "seconds": round(time.perf_counter() - t0, 1),
                "answer": body.get("answer", ""), "draft": draft,
            }
            rows.append(row)
            mark = lambda ok: "✓" if ok else "✗"
            print(f"{topic:8} raw {base_choice or '-'} {mark(row['baseline_ok'])} | framework "
                  f"{fw_choice or '-'} {mark(row['framework_ok'])} | correct {correct} | "
                  f"{row['route']} → {row['layers']} {row['seconds']:.0f}s | {row['question'][:60]}")

    scored = [r for r in rows if "error" not in r]
    for topic in ("P&C", "Circles", None):
        sub = [r for r in scored if topic is None or r["topic"] == topic]
        if not sub:
            continue
        b = sum(r["baseline_ok"] for r in sub)
        f = sum(r["framework_ok"] for r in sub)
        print(f"{topic or 'ALL':8} raw {b}/{len(sub)} | framework {f}/{len(sub)}")
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved → {out}")
    return 0 if len(scored) == len(rows) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-base", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="evaluation/results/mcq_10th.json")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.api_base, Path(args.out))))
