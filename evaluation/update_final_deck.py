"""
Update final.pptx in place with the numbers and figures of one results run.

The deck layout stays as it is: no rows, columns or slides are added. Only the
numbers, the model names and the two figures (architecture, routing) change.

Usage:
    python -m evaluation.update_final_deck \
        --results evaluation/results_v4_llama31_only --model llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pptx import Presentation

ROUTES = {
    "knowledge_uncertain": "Knowledge",
    "reasoning_uncertain": "Reasoning",
    "safe": "Safe",
}


def pct(v: float) -> str:
    return f"{v:.1f}%"


def delta(v: float) -> str:
    return f"{v:+.1f} pts"


def pval(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def pstat(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def set_cell(cell, text: str) -> None:
    runs = cell.text_frame.paragraphs[0].runs
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""


def set_body(shape, text: str) -> None:
    """Replace the body run; keep a bold 'Label: ' run when there is one."""
    runs = shape.text_frame.paragraphs[0].runs
    target = runs[1] if len(runs) > 1 else runs[0]
    target.text = text
    for r in runs[2:]:
        r.text = ""


def by_name(slide, name: str):
    return next(sh for sh in slide.shapes if sh.name == name)


def tables(slide):
    return [sh.table for sh in slide.shapes if sh.has_table]


def replace_picture(slide, name: str, png: Path) -> None:
    pic = by_name(slide, name)
    rid = pic._element.blipFill.blip.rEmbed
    slide.part.related_part(rid)._blob = png.read_bytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="final.pptx")
    ap.add_argument("--results", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    res = Path(args.results)
    m = args.model
    gs = pd.read_csv(res / "summary_gsm8k.csv").set_index("config")
    ts = pd.read_csv(res / "summary_truthfulqa.csv").set_index("config")
    gr = pd.read_csv(res / "routing_gsm8k.csv").set_index("triage_decision")
    tr = pd.read_csv(res / "routing_truthfulqa.csv").set_index("triage_decision")
    auroc = json.loads((res / "figures" / "aurocs.json").read_text())

    gb, gf = gs.loc["baseline"], gs.loc["full_framework"]
    tb, tf = ts.loc["baseline"], ts.loc["full_framework"]
    g_p, t_p = gf["is_correct_mcnemar_p"], tf["judge_truthful_mcnemar_p"]
    g_d, t_d = gf["is_correct_delta_vs_baseline"], tf["judge_truthful_delta_vs_baseline"]
    g_line = f"GSM8K {pct(gb['is_correct_pct'])} → {pct(gf['is_correct_pct'])} ({pstat(g_p)})"
    t_line = f"TruthfulQA {pct(tb['judge_truthful_pct'])} → {pct(tf['judge_truthful_pct'])} ({pstat(t_p)})"

    prs = Presentation(args.deck)
    s = prs.slides

    # Slide 8: plan / execution / result
    t = tables(s[7])[0]
    set_cell(t.cell(1, 3), f"Routes by measured uncertainty; AUROC {auroc['GSM8K']:.2f} (GSM8K), "
                           f"{auroc['TruthfulQA']:.2f} (TruthfulQA)")
    set_cell(t.cell(2, 2), "DeBERTa-v3 consensus NLI + web evidence + evidence-grounded rewrite "
                           f"and Chain-of-Verification ({m})")
    set_cell(t.cell(2, 3), f"TruthfulQA truthful: {pct(tb['judge_truthful_pct'])} → {pct(tf['judge_truthful_pct'])}")
    set_cell(t.cell(3, 3), f"GSM8K accuracy: {pct(gb['is_correct_pct'])} → {pct(gf['is_correct_pct'])}")
    set_body(by_name(s[7], "Google Shape;242;p22"),
             f"the framework improves on the raw baseline on both benchmarks — {g_line}, {t_line}. "
             "The cost is extra latency and tokens.")

    # Slide 9: implemented system
    replace_picture(s[8], "Google Shape;256;p23", res / "figures" / "fig_architecture.png")
    t = tables(s[8])[0]
    set_cell(t.cell(1, 1), f"{m} (Ollama)")
    set_cell(t.cell(2, 1), f"{m} (Ollama)")
    set_body(by_name(s[8], "Google Shape;259;p23"),
             f"one model ({m}) drafts, verifies and repairs. Every gain comes from the pipeline itself, "
             "not from a stronger model. The evaluation judge is a separate model.")

    # Slide 10: evaluation setup
    t = tables(s[9])[0]
    set_cell(t.cell(2, 1), f"Baseline (raw {m}) · Defense-in-Depth framework ({m} in every role)")

    # Slide 11: GSM8K
    t1, t2 = tables(s[10])
    set_cell(t1.cell(1, 0), f"Baseline: raw {m}")
    for row, r in ((1, gb), (2, gf)):
        set_cell(t1.cell(row, 1), pct(r["is_correct_pct"]))
        set_cell(t1.cell(row, 4), f"{r['avg_latency_s']:.1f}")
        set_cell(t1.cell(row, 5), f"{r['avg_tokens']:,.0f}")
    set_cell(t1.cell(2, 2), delta(g_d))
    set_cell(t1.cell(2, 3), pval(g_p))
    fill_routes(t2, gr)
    set_body(by_name(s[10], "Google Shape;291;p25"), delta(g_d))
    set_body(by_name(s[10], "Google Shape;294;p25"), pstat(g_p))
    set_body(by_name(s[10], "Google Shape;297;p25"), f"{gf['avg_tokens'] / gb['avg_tokens']:.1f}×")

    # Slide 12: TruthfulQA
    t1, t2 = tables(s[11])
    set_cell(t1.cell(1, 0), f"Baseline: raw {m}")
    for row, r in ((1, tb), (2, tf)):
        set_cell(t1.cell(row, 1), pct(r["judge_truthful_pct"]))
        set_cell(t1.cell(row, 4), pct(r["truthful_informative_pct"]))
        set_cell(t1.cell(row, 5), pct(r["rouge_acc_pct"]))
        set_cell(t1.cell(row, 6), f"{r['avg_latency_s']:.1f}")
        set_cell(t1.cell(row, 7), f"{r['avg_tokens']:,.0f}")
    set_cell(t1.cell(2, 2), delta(t_d))
    set_cell(t1.cell(2, 3), pval(t_p))
    fill_routes(t2, tr)
    set_body(by_name(s[11], "Google Shape;316;p26"), delta(t_d))
    set_body(by_name(s[11], "Google Shape;319;p26"), pstat(t_p))
    set_body(by_name(s[11], "Google Shape;322;p26"), f"{tf['avg_tokens'] / tb['avg_tokens']:.1f}×")

    # Slide 13: routing figure
    replace_picture(s[12], "Google Shape;337;p27", res / "figures" / "fig_routing.png")

    # Slide 14: discussion
    set_body(by_name(s[13], "Google Shape;354;p28"),
             f"the framework beats the raw {m} baseline on both benchmarks "
             f"(GSM8K {delta(g_d)}, {pstat(g_p)}; TruthfulQA {delta(t_d)}, {pstat(t_p)}).")
    set_body(by_name(s[13], "Google Shape;362;p28"),
             "the dev slice showed that drafts were often wrong even at zero entropy, "
             "so every quantitative query is verified (T_safe = 0).")

    prs.save(args.deck)
    print(f"Updated {args.deck}")


def fill_routes(table, routing: pd.DataFrame) -> None:
    """Fill the existing route rows; a route with no queries gets zeros."""
    seen = set()
    for row in range(1, len(table.rows)):
        label = table.cell(row, 0).text
        key = next((k for k, v in ROUTES.items() if label.startswith(v)), None)
        seen.add(key)
        if key in routing.index:
            r = routing.loc[key]
            vals = [f"{int(r['n'])}", pct(100 * r["baseline_acc"]), pct(100 * r["framework_acc"]),
                    pct(100 * r["changed"]), f"{r['mean_entropy']:.2f}"]
        else:
            vals = ["0", "–", "–", "–", "–"]
        for col, v in enumerate(vals, start=1):
            set_cell(table.cell(row, col), v)
    missing = set(routing.index) - seen
    if missing:
        print(f"WARNING: routes without a table row: {sorted(missing)}")


if __name__ == "__main__":
    main()
