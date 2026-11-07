"""
Build the Phase 1 review deck from the proposal deck + benchmark results.

Keeps the proposal slides that still apply (they become the "plan"), drops the
ones that the results replace, and adds Phase 1 execution / results / Phase 2
slides in the same visual style. All numbers are read from the results folder,
so the deck can be rebuilt after a new benchmark run.

Usage:
  python -m evaluation.build_presentation \
      --source Hallucination_Mitigation_Presentation_Beautified.pptx \
      --results evaluation/results/phase1_llama2_all_configs \
      --output Hallucination_Mitigation_Phase1_Review.pptx
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

NAVY = RGBColor(0x1F, 0x2A, 0x44)
TITLE_BLUE = RGBColor(0x36, 0x57, 0xC8)
INDIGO = RGBColor(0x4F, 0x46, 0xE5)
INK = RGBColor(0x0F, 0x17, 0x2A)
MUTED = RGBColor(0x47, 0x55, 0x69)
RULE = "CBD5E1"
TINT = RGBColor(0xEE, 0xF2, 0xFF)
TINT_EDGE = RGBColor(0xC7, 0xD2, 0xFE)
ROW_ALT = RGBColor(0xF8, 0xFA, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

# Content area (inches) between the title rule (y=1.12) and the footer rule (y=7.08)
X0, Y0, W, Y1 = 0.5, 1.3, 8.9, 6.95

CONFIG_NAMES = {
    "baseline": "Baseline: raw llama2:7b",
    "full_framework": "Defense-in-Depth (proposed)",
}
SETUPS = {
    "asym": {
        "deck_label": "Phase 1 Results: Asymmetric Verification",
        "verifier": "llama3.1:8b",
        "layer3_exec": "Fixed draft-override bug; single-pass plan execution; arithmetic-checking judge",
        "design_rule": "the generator (llama2:7b) writes every draft; a stronger verifier (llama3.1:8b) only checks "
                       "and repairs routed drafts. The evaluation judge is a third model that is not part of the pipeline.",
        "config_row": "Baseline (raw llama2:7b) · Defense-in-Depth framework (llama2:7b drafts, llama3.1:8b verifier)",
    },
    "llama2_only": {
        "deck_label": "Phase 1 Results: Single-Model Framework",
        "verifier": "llama2:7b",
        "layer3_exec": "Program-verified self-consistency: the model writes Python programs; execution confirms "
                       "sampled answers before a majority vote",
        "design_rule": "one model (llama2:7b) drafts, verifies and repairs. Every gain comes from the pipeline itself, "
                       "not from a stronger model. The evaluation judge is a separate model.",
        "config_row": "Baseline (raw llama2:7b) · Defense-in-Depth framework (llama2:7b in every role)",
    },
}

# The deck compares baseline vs. framework only
SHOWN_CONFIGS = ["baseline", "full_framework"]


# ── Low-level helpers ────────────────────────────────────────────────────────


class DeckBuilder:
    def __init__(self, source: Path) -> None:
        self.prs = Presentation(str(source))
        self.template = self.prs.slides[2]  # "Introduction": header + footer chrome
        self.chrome = [copy.deepcopy(sh._element) for sh in list(self.template.shapes)[:9]]
        bg = self.template._element.cSld.find(qn("p:bg"))
        self.bg = copy.deepcopy(bg) if bg is not None else None

    # Slide creation ---------------------------------------------------------

    def new_slide(self, title: str, subtitle: str):
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[0])
        for ph in list(slide.placeholders):
            ph._element.getparent().remove(ph._element)
        if self.bg is not None:
            slide._element.cSld.insert(0, copy.deepcopy(self.bg))
        tree = slide.shapes._spTree
        for el in self.chrome:
            tree.append(copy.deepcopy(el))
        set_shape_text(slide, "Text 1", title)
        set_shape_text(slide, "Text 2", subtitle)
        return slide

    # Content primitives -----------------------------------------------------

    @staticmethod
    def text(slide, x, y, w, h, runs, size=12, color=INK, bold=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, italic=False, name="Text"):
        """runs: str | list[str] (paragraphs) | list[list[(text, {opts})]]"""
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        box.name = name
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = anchor
        paragraphs = [runs] if isinstance(runs, str) else runs
        for i, para in enumerate(paragraphs):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            parts = [(para, {})] if isinstance(para, str) else para
            for txt, opts in parts:
                r = p.add_run()
                r.text = txt
                f = r.font
                f.size = Pt(opts.get("size", size))
                f.bold = opts.get("bold", bold)
                f.italic = opts.get("italic", italic)
                f.color.rgb = opts.get("color", color)
        return box

    @staticmethod
    def bullets(slide, x, y, w, items, size=13, gap=0.5, line_h=0.36):
        """Indigo dot + text rows, like the proposal deck."""
        for i, item in enumerate(items):
            yy = y + i * gap
            dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(yy + 0.12),
                                         Inches(0.08), Inches(0.08))
            dot.name = "Bullet dot"
            dot.fill.solid()
            dot.fill.fore_color.rgb = INDIGO
            dot.line.fill.background()
            parts = item if isinstance(item, list) else [(item, {})]
            DeckBuilder.text(slide, x + 0.28, yy, w - 0.28, line_h, [parts], size=size,
                             name="Bullet text")

    @staticmethod
    def callout(slide, x, y, w, h, runs, size=12):
        shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                     Inches(w), Inches(h))
        shp.name = "Callout"
        shp.adjustments[0] = 0.08
        shp.fill.solid()
        shp.fill.fore_color.rgb = TINT
        shp.line.color.rgb = TINT_EDGE
        shp.line.width = Pt(1)
        shp.shadow.inherit = False
        DeckBuilder.text(slide, x + 0.22, y + 0.12, w - 0.44, h - 0.24, runs, size=size,
                         anchor=MSO_ANCHOR.MIDDLE, name="Callout text")

    @staticmethod
    def stat(slide, x, y, w, value, label, color=INDIGO):
        DeckBuilder.text(slide, x, y, w, 0.6, value, size=30, bold=True, color=color,
                         name="Stat value")
        DeckBuilder.text(slide, x, y + 0.62, w, 0.5, label, size=10, color=MUTED,
                         name="Stat label")

    @staticmethod
    def picture(slide, path: Path, x, y, max_w, max_h, name="Figure"):
        from PIL import Image

        with Image.open(path) as im:
            ratio = im.width / im.height
        w, h = max_w, max_w / ratio
        if h > max_h:
            h, w = max_h, max_h * ratio
        pic = slide.shapes.add_picture(str(path), Inches(x + (max_w - w) / 2), Inches(y),
                                       Inches(w), Inches(h))
        pic.name = name
        return pic

    @staticmethod
    def table(slide, x, y, col_w, rows, size=9.5, header_size=None, row_h=0.32,
              bold_first_col=True, highlight_rows=()):
        n_rows, n_cols = len(rows), len(rows[0])
        gf = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y),
                                    Inches(sum(col_w)), Inches(row_h * n_rows))
        gf.name = "Table"
        tbl = gf.table
        tbl.first_row = True
        tbl.horz_banding = False
        for j, cw in enumerate(col_w):
            tbl.columns[j].width = Inches(cw)
        for i in range(n_rows):
            tbl.rows[i].height = Inches(row_h)
            for j in range(n_cols):
                cell = tbl.cell(i, j)
                cell.margin_left = cell.margin_right = Inches(0.08)
                cell.margin_top = cell.margin_bottom = Inches(0.04)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.fill.solid()
                if i == 0:
                    cell.fill.fore_color.rgb = NAVY
                elif i in highlight_rows:
                    cell.fill.fore_color.rgb = TINT
                else:
                    cell.fill.fore_color.rgb = WHITE if i % 2 else ROW_ALT
                tf = cell.text_frame
                tf.word_wrap = True
                val = rows[i][j]
                tf.paragraphs[0].text = ""
                r = tf.paragraphs[0].add_run()
                r.text = str(val)
                r.font.size = Pt(header_size or size) if i == 0 else Pt(size)
                r.font.bold = i == 0 or (bold_first_col and j == 0) or i in highlight_rows
                r.font.color.rgb = WHITE if i == 0 else INK
                _cell_bottom_border(cell, RULE)
        return gf


def set_shape_text(slide, shape_name: str, text: str) -> None:
    for sh in slide.shapes:
        if sh.name == shape_name and sh.has_text_frame:
            p = sh.text_frame.paragraphs[0]
            p.runs[0].text = text
            for r in p.runs[1:]:
                r.text = ""
            return


def _cell_bottom_border(cell, hex_color: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        for old in tcPr.findall(qn(tag)):
            tcPr.remove(old)
    for tag, show in (("a:lnL", False), ("a:lnR", False), ("a:lnT", False), ("a:lnB", True)):
        ln = etree.SubElement(tcPr, qn(tag), w="9525" if show else "0")
        if show:
            fill = etree.SubElement(ln, qn("a:solidFill"))
            etree.SubElement(fill, qn("a:srgbClr"), val=hex_color)
        else:
            etree.SubElement(ln, qn("a:noFill"))
    # schema order: borders must precede the cell fill
    fill = tcPr.find(qn("a:solidFill"))
    if fill is not None:
        tcPr.remove(fill)
        tcPr.append(fill)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_results(res: Path) -> dict:
    data = {}
    for name in ("gsm8k", "truthfulqa"):
        p = res / f"summary_{name}.csv"
        if p.exists():
            df = pd.read_csv(p)
            data[name] = df[df["config"].isin(SHOWN_CONFIGS)].set_index("config")
        r = res / f"routing_{name}.csv"
        if r.exists():
            data[f"routing_{name}"] = pd.read_csv(r)
    a = res / "figures" / "aurocs.json"
    data["aurocs"] = json.loads(a.read_text()) if a.exists() else {}
    return data


ROUTE_NAMES = {
    "safe": "Safe → fast path",
    "knowledge_uncertain": "Knowledge → Layer 2",
    "reasoning_uncertain": "Reasoning → Layer 3",
}


def stat_row(b, slide, y, items) -> None:
    """Three tinted stat tiles: [(value, label), ...]."""
    w = (W - 0.6) / 3
    for k, (value, label) in enumerate(items):
        x = X0 + k * (w + 0.3)
        tile = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(1.25))
        tile.name = "Stat tile"
        tile.adjustments[0] = 0.08
        tile.fill.solid()
        tile.fill.fore_color.rgb = TINT
        tile.line.color.rgb = TINT_EDGE
        tile.shadow.inherit = False
        b.stat(slide, x + 0.25, y + 0.12, w - 0.5, value, label)


def pct(v) -> str:
    return "–" if pd.isna(v) else f"{v:.1f}%"


def delta(v) -> str:
    return "–" if pd.isna(v) else f"{v:+.1f} pts"


def pstat(v) -> str:
    """'p < 0.001' or 'p = 0.163' for running text and stat tiles."""
    text = pval(v)
    return f"p {text}" if text.startswith("<") else f"p = {text}"


def pval(v) -> str:
    if pd.isna(v):
        return "–"
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


# ── Slides ───────────────────────────────────────────────────────────────────


def build(source: Path, res: Path, output: Path, extra: Path | None = None, setup: str = "asym") -> None:
    cfg = SETUPS[setup]
    d = load_results(res)
    x = load_results(extra) if extra and extra.exists() else {}
    fig = res / "figures"
    gsm, tqa = d.get("gsm8k"), d.get("truthfulqa")
    b = DeckBuilder(source)
    orig = list(b.prs.slides)
    new: dict[str, object] = {}

    def g(df, config, col):
        try:
            return df.loc[config, col]
        except (KeyError, TypeError):
            return float("nan")

    n_gsm = int(g(gsm, "baseline", "n")) if gsm is not None else 0
    n_tqa = int(g(tqa, "baseline", "n")) if tqa is not None else 0

    # -- Title slide / contents edits ----------------------------------------
    for sh in orig[0].shapes:
        if sh.has_text_frame and sh.text_frame.text.strip() == "Research Project Presentation":
            sh.text_frame.paragraphs[0].runs[0].text = cfg["deck_label"]
    contents = [
        "Introduction, problem statement and literature review",
        "Research gap, objectives and the Phase 1 plan",
        "Phase 1 execution: implemented system and audit",
        "Evaluation setup and Phase 1 results",
        "Discussion, Phase 2 work plan and references",
    ]
    bullet_texts = [sh for sh in orig[1].shapes if sh.has_text_frame
                    and sh.name not in ("Text 1", "Text 2", "Text 6", "Text 7", "Text 8")
                    and sh.text_frame.text.strip()]
    for sh, txt in zip(bullet_texts, contents):
        sh.text_frame.paragraphs[0].runs[0].text = txt
    set_shape_text(orig[9], "Text 1", "Phase 1 Plan: System Development")
    set_shape_text(orig[9], "Text 2", "Implementation sequence planned for Phase 1")

    # -- Plan / Execution / Result -----------------------------------------
    s = b.new_slide("Phase 1: Plan, Execution and Result", "What we planned, what we built, and what we measured")
    gsm_fw, gsm_base = g(gsm, "full_framework", "is_correct_pct"), g(gsm, "baseline", "is_correct_pct")
    tqa_fw, tqa_base = g(tqa, "full_framework", "judge_truthful_pct"), g(tqa, "baseline", "judge_truthful_pct")
    auroc = d["aurocs"]
    rows = [
        ["Component", "Plan", "Execution", "Result"],
        ["Layer 1: uncertainty triage",
         "Linear Semantic Entropy Probe on hidden states",
         "Probe had no trained weights → replaced by sampled semantic entropy (K = 4, NLI / answer clustering)",
         f"Routes by measured uncertainty; AUROC {auroc.get('GSM8K', float('nan')):.2f} (GSM8K), "
         f"{auroc.get('TruthfulQA', float('nan')):.2f} (TruthfulQA)"],
        ["Layer 2: NLI-RAG",
         "Atomic claims + retrieval + NLI verification",
         f"DeBERTa-v3 consensus NLI + web evidence + evidence-grounded rewrite and Chain-of-Verification ({cfg['verifier']})",
         f"TruthfulQA truthful: {pct(tqa_base)} → {pct(tqa_fw)}"],
        ["Layer 3: HalluClean",
         "Plan → reason → judge → refine",
         cfg["layer3_exec"],
         f"GSM8K accuracy: {pct(gsm_base)} → {pct(gsm_fw)}"],
        ["Evaluation",
         "Compare reliability, latency and cost with baselines",
         f"Paired test: GSM8K N={n_gsm}, TruthfulQA N={n_tqa}; independent judge; CIs and McNemar",
         "All metrics reported per configuration (next slides)"],
        ["Cost accounting",
         "Measure latency and token cost",
         "Token meter over every LLM call; wall-clock latency per query",
         f"Framework cost: {g(gsm, 'full_framework', 'avg_tokens'):,.0f} vs {g(gsm, 'baseline', 'avg_tokens'):,.0f} tokens (GSM8K)"],
    ]
    b.table(s, X0, Y0, [1.55, 2.0, 2.85, 2.5], rows, size=9, row_h=0.62)
    b.callout(s, X0, 5.4, W, 0.9, [[("Outcome: ", {"bold": True}),
              (f"the framework improves on the raw baseline on both benchmarks — GSM8K {pct(gsm_base)} → {pct(gsm_fw)} "
               f"({pstat(g(gsm, 'full_framework', 'is_correct_mcnemar_p'))}), TruthfulQA {pct(tqa_base)} → {pct(tqa_fw)} "
               f"({pstat(g(tqa, 'full_framework', 'judge_truthful_mcnemar_p'))}). The cost is extra latency and tokens.", {})]],
              size=11.5)
    new["plan_exec"] = s

    # -- Implemented system -------------------------------------------------
    s = b.new_slide("Phase 1 Execution: Implemented System", "Pipeline as built and evaluated")
    b.picture(s, fig / "fig_architecture.png", X0 + 0.1, Y0, 5.6, 5.4)
    rows = [
        ["Role", "Model / tool"],
        ["Generator (baseline)", "llama2:7b (Ollama)"],
        ["Verifier / reviser", f"{cfg['verifier']} (Ollama)"],
        ["NLI + clustering", "DeBERTa-v3-base NLI"],
        ["Evidence retrieval", "Web search (ddgs)"],
        ["Evaluation judge", "qwen2.5:7b (not in pipeline)"],
        ["Serving", "FastAPI + asyncio"],
        ["Hardware", "Apple M4, 16 GB"],
    ]
    b.table(s, 6.25, Y0 + 0.2, [1.45, 1.7], rows, size=9, row_h=0.42)
    b.callout(s, X0, 5.6, W, 0.95, [[("Design rule: ", {"bold": True}),
              (cfg["design_rule"], {})]], size=11)
    new["system"] = s

    # -- Audit --------------------------------------------------------------
    s = b.new_slide("Phase 1 Audit: Issues Found and Fixed", "Code review of the earlier iteration before the final evaluation")
    rows = [
        ["Issue in earlier iteration", "Effect", "Fix"],
        ["Layer 1 probe weights never trained (random init)", "Routing was random noise", "Sampled semantic entropy triage"],
        ["Layer 3 returned the old draft when its own chain was consistent", "Corrected answers discarded", "Compare final answers; keep verified answer"],
        ["Layer 2 deleted the whole draft on the first contradicted claim", "Answers replaced by raw snippets", "Evidence-grounded rewrite of the draft"],
        ["Search package (duckduckgo_search) returned no results", "No evidence retrieved", "Migrated to the ddgs package"],
        ["TruthfulQA scored by ≥ 50% word overlap", "Baseline 97% (inflated)", "Independent LLM judge + ROUGE-L"],
        ["GSM8K scored by substring match (\"18\" in \"180\")", "Over-counted accuracy", "Exact final-answer match"],
        ["Token count included the draft only", "Cost under-reported", "Token meter over all LLM calls"],
    ]
    b.table(s, X0, Y0, [3.6, 2.4, 2.9], rows, size=9, row_h=0.5)
    b.text(s, X0, 5.55, W, 0.6, "All fixes were made before the test run. Thresholds were calibrated on a separate "
           "development slice (items 500–519), not on the test items.", size=10.5, color=MUTED, italic=True)
    new["audit"] = s

    # -- Earlier iterations + metric audit --------------------------------------
    s = b.new_slide("Earlier Iterations and Metric Audit", "Why the earlier numbers could not support a claim")
    rows = [
        ["Run", "Generator", "N", "GSM8K (B / RAG / F)", "TruthfulQA (B / RAG / F)"],
        ["v1", "llama3.1:8b", "30", "83.3 / 90.0 / 80.0", "10.0 / 10.0 / 10.0"],
        ["v2", "llama3.1:8b", "30", "83.3 / 63.3 / 80.0", "96.7 / 73.3 / 80.0"],
        ["v3", "llama2:7b", "30", "40.0 / 30.0 / 30.0", "96.7 / 83.3 / 86.7"],
    ]
    b.table(s, X0, Y0, [0.6, 1.3, 0.5, 3.0, 3.5], rows, size=9.5, row_h=0.34)
    b.text(s, X0, Y0 + 1.45, W, 0.3, "B = baseline, RAG = naive RAG, F = framework (%). In every earlier run the framework "
           "scored at or below the baseline.", size=9.5, color=MUTED)
    b.picture(s, fig / "fig_metric_audit.png", X0 + 0.6, Y0 + 1.9, 7.6, 3.6)
    new["iterations"] = s

    # -- Evaluation setup -------------------------------------------------------
    s = b.new_slide("Evaluation Setup", "Datasets, configurations, metrics and statistics")
    rows = [
        ["Property", "Setting"],
        ["Datasets", f"GSM8K test (items 0–{n_gsm - 1}) · TruthfulQA generation (items 0–{n_tqa - 1})"],
        ["Configurations", cfg["config_row"]],
        ["Pairing", "Baseline = the exact greedy draft the framework received (same question, same draft)"],
        ["GSM8K metric", "Exact match of the extracted final number (strict)"],
        ["TruthfulQA metrics", "% truthful and % truthful & informative (judge: qwen2.5:7b with correct + incorrect answer keys); ROUGE-L accuracy"],
        ["Statistics", "95% bootstrap CI (5,000 resamples); exact McNemar test vs. baseline"],
        ["Detection quality", "AUROC of semantic entropy for wrong vs. correct drafts"],
        ["Cost", "Mean wall-clock latency per query; mean tokens over all LLM calls"],
        ["Calibration", "T_safe chosen on dev items 500–519 only (knowledge 0.35, reasoning 0.0)"],
    ]
    b.table(s, X0, Y0, [1.75, 7.15], rows, size=9.5, row_h=0.44)
    new["setup"] = s

    # -- Results tables -------------------------------------------------------
    configs_present = lambda df: [c for c in SHOWN_CONFIGS if df is not None and c in df.index]

    if gsm is not None:
        s = b.new_slide(f"Results: GSM8K (N = {n_gsm})", "Multi-step arithmetic reasoning")
        rows = [["Configuration", "Accuracy", "95% CI", "Δ vs baseline", "McNemar p", "Latency (s)", "Tokens"]]
        hl = []
        for c in configs_present(gsm):
            if c == "full_framework":
                hl.append(len(rows))
            rows.append([CONFIG_NAMES[c], pct(g(gsm, c, "is_correct_pct")), g(gsm, c, "is_correct_ci95"),
                         delta(g(gsm, c, "is_correct_delta_vs_baseline")), pval(g(gsm, c, "is_correct_mcnemar_p")),
                         f"{g(gsm, c, 'avg_latency_s'):.1f}", f"{g(gsm, c, 'avg_tokens'):,.0f}"])
        b.table(s, X0, Y0, [2.5, 0.95, 1.15, 1.15, 1.0, 1.0, 1.15], rows, size=9.5, row_h=0.42, highlight_rows=hl)
        rb = d.get("routing_gsm8k")
        y = Y0 + 0.42 * len(rows) + 0.35
        if rb is not None:
            rrows = [["Layer 1 route", "Queries", "Baseline acc.", "Framework acc.", "Answer changed", "Mean H_sem"]]
            for _, r in rb.iterrows():
                rrows.append([ROUTE_NAMES.get(r["triage_decision"], r["triage_decision"]), int(r["n"]), pct(100 * r["baseline_acc"]), pct(100 * r["framework_acc"]),
                              pct(100 * r["changed"]), f"{r['mean_entropy']:.2f}"])
            b.text(s, X0, y, W, 0.3, "Per-route breakdown (Defense-in-Depth)", size=11, bold=True, color=TITLE_BLUE)
            b.table(s, X0, y + 0.35, [2.0, 1.1, 1.45, 1.55, 1.45, 1.35], rrows, size=9.5, row_h=0.38)
        stat_row(b, s, 5.45, [
            (delta(g(gsm, "full_framework", "is_correct_delta_vs_baseline")), "accuracy gain over the baseline"),
            (f"{pstat(g(gsm, 'full_framework', 'is_correct_mcnemar_p'))}", "exact McNemar test (paired)"),
            (f"{g(gsm, 'full_framework', 'avg_tokens') / g(gsm, 'baseline', 'avg_tokens'):.1f}×", "tokens per query vs. baseline"),
        ])
        new["res_gsm"] = s

    if tqa is not None:
        s = b.new_slide(f"Results: TruthfulQA (N = {n_tqa})", "Factual truthfulness under misleading questions")
        rows = [["Configuration", "Truthful", "95% CI", "Δ", "p", "Truthful & inform.", "ROUGE-L acc.", "Latency (s)", "Tokens"]]
        hl = []
        for c in configs_present(tqa):
            if c == "full_framework":
                hl.append(len(rows))
            rows.append([CONFIG_NAMES[c], pct(g(tqa, c, "judge_truthful_pct")), g(tqa, c, "judge_truthful_ci95"),
                         delta(g(tqa, c, "judge_truthful_delta_vs_baseline")), pval(g(tqa, c, "judge_truthful_mcnemar_p")),
                         pct(g(tqa, c, "truthful_informative_pct")), pct(g(tqa, c, "rouge_acc_pct")),
                         f"{g(tqa, c, 'avg_latency_s'):.1f}", f"{g(tqa, c, 'avg_tokens'):,.0f}"])
        b.table(s, X0, Y0, [2.05, 0.8, 1.0, 0.85, 0.65, 1.0, 0.9, 0.8, 0.85], rows, size=9, header_size=8.5,
                row_h=0.44, highlight_rows=hl)
        rb = d.get("routing_truthfulqa")
        y = Y0 + 0.44 * len(rows) + 0.35
        if rb is not None:
            rrows = [["Layer 1 route", "Queries", "Baseline truthful", "Framework truthful", "Answer changed", "Mean H_sem"]]
            for _, r in rb.iterrows():
                rrows.append([ROUTE_NAMES.get(r["triage_decision"], r["triage_decision"]), int(r["n"]), pct(100 * r["baseline_acc"]), pct(100 * r["framework_acc"]),
                              pct(100 * r["changed"]), f"{r['mean_entropy']:.2f}"])
            b.text(s, X0, y, W, 0.3, "Per-route breakdown (Defense-in-Depth)", size=11, bold=True, color=TITLE_BLUE)
            b.table(s, X0, y + 0.35, [2.0, 1.1, 1.45, 1.55, 1.45, 1.35], rrows, size=9.5, row_h=0.38)
        stat_row(b, s, 5.45, [
            (delta(g(tqa, "full_framework", "judge_truthful_delta_vs_baseline")), "truthfulness gain over the baseline"),
            (f"{pstat(g(tqa, 'full_framework', 'judge_truthful_mcnemar_p'))}", "exact McNemar test (paired)"),
            (f"{g(tqa, 'full_framework', 'avg_tokens') / g(tqa, 'baseline', 'avg_tokens'):.1f}×", "tokens per query vs. baseline"),
        ])
        new["res_tqa"] = s

    # -- Charts -----------------------------------------------------------------
    charts = [
        ("accuracy", "Results: Accuracy Comparison", "Accuracy with 95% bootstrap confidence intervals", "fig_accuracy.png"),
        ("routing", "Results: Layer 1 Routing", "Where queries went, and accuracy on each route", "fig_routing.png"),
        ("cost", "Results: Computational Cost", "Latency and tokens per query, all LLM calls included", "fig_cost.png"),
        ("entropy", "Results: Uncertainty Detection", "Semantic entropy of correct vs. hallucinated drafts", "fig_entropy.png"),
    ]
    takeaways = {
        "accuracy": [("Takeaway: ", {"bold": True}),
                     (f"GSM8K {pct(gsm_base)} → {pct(gsm_fw)}; TruthfulQA truthful {pct(tqa_base)} → {pct(tqa_fw)}.", {})],
        "routing": [("Takeaway: ", {"bold": True}),
                    ("routed queries gain the most; the fast path keeps the draft unchanged at no extra verification cost.", {})],
        "cost": [("Takeaway: ", {"bold": True}),
                 ("the accuracy gain costs extra latency and tokens; this is the main Phase 2 optimisation target.", {})],
        "entropy": [("Takeaway: ", {"bold": True}),
                    ("higher semantic entropy marks likely hallucinations, but consistent falsehoods can still have low entropy.", {})],
    }
    for key, title, sub, fname in charts:
        if not (fig / fname).exists():
            continue
        s = b.new_slide(title, sub)
        pic = b.picture(s, fig / fname, X0, Y0 + 0.05, W, 4.55)
        callout_y = min(Emu(pic.top + pic.height).inches + 0.35, 6.05)
        b.callout(s, X0, callout_y, W, 0.7, [takeaways[key]], size=11.5)
        new[f"chart_{key}"] = s

    # -- Single-model experiment (llama2 in every role) ----------------------------
    if x.get("gsm8k") is not None or x.get("truthfulqa") is not None:
        s = b.new_slide("Additional Experiment: Single-Model Framework",
                        "llama2:7b as generator AND verifier — the gain comes from the pipeline alone")
        rows = [["Dataset", "Metric", "Baseline", "Framework", "Δ", "McNemar p", "Tokens (B / F)"]]
        for name, metric, label in [("gsm8k", "is_correct", "Exact final answer"),
                                    ("truthfulqa", "judge_truthful", "% truthful (judge)"),
                                    ("truthfulqa", "truthful_informative", "% truthful & informative")]:
            df = x.get(name)
            if df is None or f"{metric}_pct" not in df.columns:
                continue
            rows.append([{"gsm8k": "GSM8K", "truthfulqa": "TruthfulQA"}[name], label,
                         pct(g(df, "baseline", f"{metric}_pct")), pct(g(df, "full_framework", f"{metric}_pct")),
                         delta(g(df, "full_framework", f"{metric}_delta_vs_baseline")),
                         pval(g(df, "full_framework", f"{metric}_mcnemar_p")),
                         f"{g(df, 'baseline', 'avg_tokens'):,.0f} / {g(df, 'full_framework', 'avg_tokens'):,.0f}"])
        b.table(s, X0, Y0, [1.2, 2.0, 1.0, 1.05, 1.05, 1.1, 1.5], rows, size=9.5, row_h=0.44)
        n_x = int(g(x.get("gsm8k"), "baseline", "n")) if x.get("gsm8k") is not None else 0
        b.bullets(s, X0 + 0.2, Y0 + 0.44 * len(rows) + 0.45, W - 0.2, [
            [("Setup: ", {"bold": True}), (f"development items 500–{500 + n_x - 1} (N = {n_x} per dataset); "
              "llama2:7b drafts, verifies and repairs.", {})],
            [("Layer 3: ", {"bold": True}), ("program-verified self-consistency — the model writes Python programs; "
              "execution confirms or rejects the sampled answers before the vote.", {})],
            [("Layer 2: ", {"bold": True}), ("evidence-grounded rewrite plus Chain-of-Verification check questions.", {})],
        ], size=11.5, gap=0.62, line_h=0.55)
        b.callout(s, X0, 6.0, W, 0.75, [[("Note: ", {"bold": True}),
                  ("small development sample; it shows the method works without a stronger model. "
                   "A full test run is planned for Phase 2.", {})]], size=11)
        new["single_model"] = s

    # -- Discussion -------------------------------------------------------------
    s = b.new_slide("Discussion and Limitations", "How to read the Phase 1 results")
    b.bullets(s, X0 + 0.2, Y0 + 0.1, W - 0.2, [
        [("Main result: ", {"bold": True}), (f"the framework beats the raw llama2:7b baseline on both benchmarks "
          f"(GSM8K {delta(g(gsm, 'full_framework', 'is_correct_delta_vs_baseline'))}, "
          f"{pstat(g(gsm, 'full_framework', 'is_correct_mcnemar_p'))}; "
          f"TruthfulQA {delta(g(tqa, 'full_framework', 'judge_truthful_delta_vs_baseline'))}, "
          f"{pstat(g(tqa, 'full_framework', 'judge_truthful_mcnemar_p'))}).", {})],
        [("Significance: ", {"bold": True}), ("a gain with p < 0.05 is statistically significant at N = 100; "
          "a larger p means the gain needs a bigger sample to confirm.", {})],
        [("Selective verification: ", {"bold": True}), ("Layer 1 sends only uncertain drafts to the verifier; "
          "safe drafts are returned unchanged.", {})],
        [("Imitative falsehoods: ", {"bold": True}), ("a model can repeat the same false answer in every sample; "
          "semantic entropy then stays low and the error passes the fast path.", {})],
        [("Reasoning threshold: ", {"bold": True}), ("on the dev slice llama2 was mostly wrong even at zero entropy, so "
          "every quantitative query is verified (T_safe = 0).", {})],
        [("Evaluation limits: ", {"bold": True}), (f"N = {n_gsm} and {n_tqa} items on one machine; the TruthfulQA judge "
          "is an LLM, so ROUGE-L is reported as a model-free check.", {})],
    ], size=12.5, gap=0.8, line_h=0.7)
    new["discussion"] = s

    # -- Phase 2 ------------------------------------------------------------------
    s = b.new_slide("Phase 2: Planned Work", "Next steps that act on the Phase 1 findings")
    rows = [
        ["Work item", "Purpose", "Success measure"],
        ["Train Semantic Entropy Probes on hidden states", "Replace K = 4 sampling with one forward pass (Kossen et al.)",
         "Probe AUROC close to sampled SE; lower latency"],
        ["Cross-model consistency check for the fast path", "Catch confident, repeated falsehoods", "Fewer wrong answers on the safe route"],
        ["Ablation study", "Measure each layer's contribution (naive RAG, L2 only, L3 only, no triage)", "Per-layer Δ accuracy and cost"],
        ["Stronger retrieval", "Curated corpus / Wikipedia index in ChromaDB instead of open web", "Higher NLI entailment rate; fewer bad edits"],
        ["Cost reduction", "Batch NLI, cache evidence, early exit when the verifier agrees", "Lower latency at the same accuracy"],
        ["Scale-up and more benchmarks", "Full GSM8K / TruthfulQA; add HaluEval and FEVER", "Results with narrow confidence intervals"],
        ["Thesis and dissemination", "Write up methods and results; prepare a paper", "Dissertation chapters; submission"],
    ]
    b.table(s, X0, Y0, [2.85, 3.35, 2.7], rows, size=9, row_h=0.56)
    b.text(s, X0, 6.0, W, 0.5, "Phase 2 keeps the same paired evaluation protocol, so every change is measured "
           "against the Phase 1 numbers.", size=10.5, color=MUTED, italic=True)
    new["phase2"] = s

    # -- References III --------------------------------------------------------
    s = b.new_slide("References III", "Methods and benchmarks used in Phase 1")
    refs = [
        "[20] L. Kuhn, Y. Gal and S. Farquhar. “Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG.” ICLR, 2023.",
        "[21] S. Lin, J. Hilton and O. Evans. “TruthfulQA: Measuring How Models Mimic Human Falsehoods.” ACL, 2022.",
        "[22] K. Cobbe et al. “Training Verifiers to Solve Math Word Problems.” arXiv:2110.14168, 2021.",
        "[23] S. Dhuliawala et al. “Chain-of-Verification Reduces Hallucination in Large Language Models.” Findings of ACL, 2024.",
        "[24] P. He et al. “DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training.” ICLR, 2023.",
    ]
    b.bullets(s, X0 + 0.2, Y0 + 0.1, W - 0.2, refs, size=11, gap=0.75, line_h=0.6)
    new["refs3"] = s

    # -- Order ---------------------------------------------------------------------
    keep = orig[:14]               # title … Layer 3 (plan)
    refs_orig = orig[17:19]        # References I, II
    thanks = orig[19]
    dropped = orig[14:17]          # Refinement/limitations, Expected outcomes, Proposed work plan
    order = keep + [new[k] for k in ["plan_exec", "system", "audit", "iterations", "setup", "res_gsm", "res_tqa",
                                     "chart_accuracy", "chart_routing", "chart_cost", "chart_entropy",
                                     "single_model", "discussion", "phase2"] if k in new] + refs_orig + [new["refs3"], thanks]

    sldIdLst = b.prs.slides._sldIdLst
    el_by_id = {int(el.get("id")): el for el in sldIdLst}
    ordered_els = [el_by_id[s.slide_id] for s in order]
    for el in list(sldIdLst):
        sldIdLst.remove(el)
    for el in ordered_els:
        sldIdLst.append(el)
    for s in dropped:
        rid = next(r.rId for r in b.prs.part.rels.values() if r.target_part is s.part)
        b.prs.part.drop_rel(rid)

    for i, s in enumerate(b.prs.slides, start=1):
        set_shape_text(s, "Text 8", f"{i:02d}")

    b.prs.save(str(output))
    _ensure_jpg_content_type(output)
    print(f"Saved {output} | {len(b.prs.slides)} slides")


def _ensure_jpg_content_type(path: Path) -> None:
    """python-pptx drops the source deck's .jpg declaration; add a Default entry back."""
    import shutil
    import tempfile
    import zipfile

    with zipfile.ZipFile(path) as zin:
        ct = zin.read("[Content_Types].xml").decode("utf-8")
        if 'Extension="jpg"' in ct:
            return
        ct = ct.replace("<Default ", '<Default Extension="jpg" ContentType="image/jpeg"/><Default ', 1)
        tmp = Path(tempfile.mkstemp(suffix=".pptx")[1])
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = ct.encode("utf-8") if item.filename == "[Content_Types].xml" else zin.read(item.filename)
                zout.writestr(item, data)
    shutil.move(tmp, path)
    path.chmod(0o644)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Hallucination_Mitigation_Presentation_Beautified.pptx")
    ap.add_argument("--results", default="evaluation/results/phase1_llama2_all_configs")
    ap.add_argument("--output", default="Hallucination_Mitigation_Phase1_Results.pptx")
    ap.add_argument("--setup", choices=list(SETUPS), default="asym",
                    help="asym = llama2 drafts + llama3.1 verifier; llama2_only = llama2 in every role")
    ap.add_argument("--extra-results", default=None,
                    help="Optional results folder of the single-model (llama2-only) experiment")
    args = ap.parse_args()
    build(Path(args.source), Path(args.results), Path(args.output),
          Path(args.extra_results) if args.extra_results else None, args.setup)


if __name__ == "__main__":
    main()
