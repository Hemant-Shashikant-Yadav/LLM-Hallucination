"""
Phase 1 deck (thesis objectives 1-2) built on the college template.

Keeps the template's title, Introduction and Problem Statement slides as written, edits Contents
and Objectives, and adds every other slide with the template's own chrome (blue bar, title,
subtitle, rule, footer) copied from its content slides.

Result slides: until the full study has run on the Mac, the per-model tables and the
across-model chart show PLACEHOLDER values (made up by _placeholder() below and labelled as such
on every slide). Four cells reuse real numbers from the pipeline's stored test runs (RESULTS).
After the Mac run, put the real numbers in PH and set PLACEHOLDER = False.

Usage: python build_phase1_deck.py <template.pptx> <output.pptx>
"""
import copy
import sys

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

TEMPLATE, OUT = sys.argv[1], sys.argv[2]

# Template colours (read from its slides) and a related palette for diagrams and charts
BLUE, INDIGO, INK, SLATE, RULE, NAVY = "3657C8", "4F46E5", "0F172A", "475569", "CBD5E1", "1F2A44"
TINT, TINT_LINE = "EEF2FF", "C7D2FE"
TEAL, AMBER, ROSE = "0D9488", "D97706", "E11D48"
T_TEAL, T_INDIGO, T_AMBER, T_ROSE, T_GREY = "CCFBF1", "E0E7FF", "FEF3C7", "FFE4E6", "F1F5F9"
WHITE = "FFFFFF"
FONT = "Arial"

# Real numbers: pipeline Layer 1 (4 samples, temperature 0.7) on stored test runs, 100 questions each
RESULTS = [  # model, dataset, accuracy, AUROC, CI low, CI high, mean H right, mean H wrong
    ("qwen2.5:3b", "GSM8K (maths)", 0.84, 0.86, 0.76, 0.95, 0.10, 0.54),
    ("qwen2.5:3b", "TruthfulQA (myths)", 0.66, 0.55, 0.44, 0.66, 0.86, 0.89),
    ("llama2:7b", "GSM8K (maths)", 0.24, 0.79, 0.69, 0.87, 0.36, 0.66),
    ("llama2:7b", "TruthfulQA (myths)", 0.61, 0.58, 0.48, 0.69, 0.72, 0.81),
]

# Placeholder values for the full-study result slides: illustrative only, never measured. They come
# from the formula in _placeholder() and are replaced by uncertainty/results/ after the Mac run.
PLACEHOLDER = True
MODELS = [  # roster model, family, a made-up 'strength' (0-1) that drives the placeholders
    ("qwen2.5:3b", "qwen", .50), ("qwen2.5:7b", "qwen", .66), ("qwen2.5:14b", "qwen", .78),
    ("qwen3:4b", "qwen", .62), ("llama3.2:3b", "llama", .40), ("llama3.1:8b", "llama", .60),
    ("llama2:7b", "llama", .22), ("gemma3:4b", "gemma", .52), ("gemma3:12b", "gemma", .74),
    ("phi4-mini", "phi", .48), ("phi4", "phi", .80), ("mistral:7b", "mistral", .42)]
DATASETS = {  # accuracy at strength 0 and at 1, base AUROC, base blind spot
    "TriviaQA": (.42, .82, .78, .22), "PopQA": (.10, .40, .74, .30), "HaluEval": (.30, .66, .70, .26),
    "TruthfulQA": (.40, .74, .57, .10), "GSM8K": (.18, .95, .84, .12), "MATH": (.05, .62, .80, .15),
    "StrategyQA": (.55, .80, .66, .34), "ARC": (.55, .93, .76, .30), "AmbigQA": (.20, .48, .62, .24),
    "SelfAware": (.25, .55, .64, .38), "FreshQA": (.15, .45, .66, .34)}
GROUPS = [("Facts and myths", ["TriviaQA", "PopQA", "HaluEval", "TruthfulQA"]),
          ("Reasoning", ["GSM8K", "MATH", "StrategyQA", "ARC"]),
          ("Unclear and changing questions", ["AmbigQA", "SelfAware", "FreshQA"])]


def _placeholder():
    """(model, dataset) -> [accuracy, semantic-entropy AUROC, blind spot]; deterministic."""
    import random
    out = {}
    for model, family, s in MODELS:
        for d, (lo, hi, auc, blind) in DATASETS.items():
            r = random.Random(f"{model}|{d}")
            bonus = .18 if family in ("qwen", "phi") and d in ("GSM8K", "MATH") else 0
            acc = min(.97, max(.03, lo + (hi - lo) * s + bonus + r.uniform(-.04, .04)))
            au = min(.95, max(.50, auc + .08 * (s - .5) + r.uniform(-.03, .03)))
            bs = min(.60, max(.02, blind + .08 * (s - .5) + r.uniform(-.04, .04)))
            out[model, d] = [acc, au, bs]
    for m, d, acc, au, *_ in RESULTS:  # the four real cells (accuracy and AUROC only)
        out[m, d.split(" ")[0]][:2] = [acc, au]
    return out


PH = _placeholder()

IN = 914400
E = lambda v: Emu(int(round(v * IN)))
rgb = RGBColor.from_string

prs = Presentation(TEMPLATE)
TPL = list(prs.slides)
CHROME_SRC = TPL[5]  # "Phase 1 Plan" slide: bar, title, subtitle, rule, footer and one placeholder bullet


def _y(shape):
    return (shape.top or 0) / IN


CHROME = [copy.deepcopy(s._element) for s in CHROME_SRC.shapes if _y(s) < 0.9 or _y(s) > 5.2]


# ------------------------------------------------------------------ text helpers
def style_run(run, size=14, color=INK, bold=False, italic=False, font=FONT, sub=False):
    f = run.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, font
    f.color.rgb = rgb(color)
    if sub:
        run._r.get_or_add_rPr().set("baseline", "-25000")


def fill_tf(tf, paras, size=14, color=INK, bold=False, italic=False, align=PP_ALIGN.LEFT, space_after=0,
            font=FONT, line_spacing=None):
    """paras: str (\\n = new paragraph) or list of paragraphs; a paragraph is str or list of (text, opts)."""
    if isinstance(paras, str):
        paras = paras.split("\n")
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if space_after:
            p.space_after = Pt(space_after)
        if line_spacing:
            p.line_spacing = line_spacing
        runs = [(para, {})] if isinstance(para, str) else para
        for text, opts in runs:
            r = p.add_run()
            r.text = text
            style_run(r, size=opts.get("size", size), color=opts.get("color", color), bold=opts.get("bold", bold),
                      italic=opts.get("italic", italic), font=opts.get("font", font), sub=opts.get("sub", False))


def text(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, **kw):
    tb = slide.shapes.add_textbox(E(x), E(y), E(w), E(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    fill_tf(tf, paras, **kw)
    return tb


def _plain(shape):
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def box(slide, x, y, w, h, fill=None, line=None, paras="", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12,
        anchor=MSO_ANCHOR.MIDDLE, margin=0.08, **kw):
    s = slide.shapes.add_shape(shape, E(x), E(y), E(w), E(h))
    _plain(s)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = E(margin)
    tf.margin_top = tf.margin_bottom = E(0.04)
    tf.vertical_anchor = anchor
    if paras:
        fill_tf(tf, paras, **kw)
    return s


def dot(slide, x, y, d, fill, label="", size=12, color=WHITE):
    return box(slide, x, y, d, d, fill=fill, paras=label, shape=MSO_SHAPE.OVAL, margin=0, size=size, color=color,
               bold=True, align=PP_ALIGN.CENTER)


def arrow(slide, x1, y1, x2, y2, color=SLATE, width=1.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, E(x1), E(y1), E(x2), E(y2))
    _plain(c)
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    tail = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"))
    tail.set("type", "triangle")
    return c


def bullets(slide, x, y, w, items, size=14, step=None, dot_color=INDIGO, h=0.4, gap=0.2):
    """Template-style bullets: a small indigo ellipse and a text box per item. step=None spaces by line count."""
    import math
    yy = y
    for i, item in enumerate(items):
        if i:
            if step:
                yy += step
            else:
                prev = items[i - 1]
                lines = max(1, math.ceil(len(prev) * size * 0.0062 / (w - 0.28)))
                yy += lines * size * 1.2 / 72 + gap
        box(slide, x, yy + 0.09, 0.08, 0.06, fill=dot_color, shape=MSO_SHAPE.OVAL)
        text(slide, x + 0.28, yy, w - 0.28, h, [item] if isinstance(item, list) else item, size=size)


def tag(slide, x, y, label, w=1.45):
    """The template's small section tag (as on the Introduction slide)."""
    box(slide, x, y, w, 0.24, fill=TINT, paras=label, size=8, color=INDIGO, bold=True, margin=0.08, radius=0.2)


def placeholder_note(slide, y, note):
    """The amber 'placeholder values' tag and a source line; only the source line once results are real."""
    if PLACEHOLDER:
        box(slide, 0.6, y, 1.7, 0.26, fill=T_AMBER, paras="PLACEHOLDER VALUES", size=8, bold=True, color="92400E",
            margin=0.08, radius=0.2, align=PP_ALIGN.CENTER)
        note = "Illustrative numbers until the full run on the Mac replaces them. " + note
    text(slide, 2.45 if PLACEHOLDER else 0.6, y + 0.02, 6.95, 0.42, note, size=9, italic=True, color=SLATE)


def table(slide, x, y, w, col_w, rows, row_h=0.36, size=10.5, header_fill=BLUE, fills=None, bold_first_col=False,
          aligns=None):
    """rows[0] is the header. fills: per-cell override {(r, c): hex}."""
    nr, nc = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nr, nc, E(x), E(y), E(w), E(row_h * nr))
    tbl = gf.table
    tbl.first_row = True
    tbl.horz_banding = False
    for c, cw in enumerate(col_w):
        tbl.columns[c].width = E(cw)
    for r in range(nr):
        tbl.rows[r].height = E(row_h)
        for c in range(nc):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = E(0.07)
            cell.margin_top = cell.margin_bottom = E(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            val = rows[r][c]
            fill = (fills or {}).get((r, c)) or (header_fill if r == 0 else (WHITE if r % 2 else TINT))
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(fill)
            tf = cell.text_frame
            tf.word_wrap = True
            al = (aligns or [PP_ALIGN.LEFT] * nc)[c]
            if r == 0:
                fill_tf(tf, val, size=size + 0.5, color=WHITE, bold=True, align=al)
            else:
                fill_tf(tf, val, size=size, color=INK, bold=bold_first_col and c == 0, align=al)
    return gf


def column_chart(slide, x, y, w, h, categories, series, colors, title, vmax=1.0, unit=0.25, fmt="0.00",
                 legend=True, labels=True, cat_size=10):
    cd = CategoryChartData()
    cd.categories = categories
    for name, values in series:
        cd.add_series(name, values)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, E(x), E(y), E(w), E(h), cd)
    ch = gf.chart
    ch.font.name, ch.font.size = FONT, Pt(10)
    ch.font.color.rgb = rgb(INK)
    ch.has_title = True
    ch.chart_title.text_frame.text = title
    fill_tf_title = ch.chart_title.text_frame.paragraphs[0]
    for r in fill_tf_title.runs:
        style_run(r, size=11, color=INK, bold=True)
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(10)
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = 0, vmax, unit
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("E2E8F0")
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(9)
    va.tick_labels.font.color.rgb = rgb(SLATE)
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = fmt, False
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(cat_size)
    ca.format.line.color.rgb = rgb(RULE)
    plot = ch.plots[0]
    plot.gap_width, plot.overlap = 70, -8
    plot.has_data_labels = labels
    if labels:
        dl = plot.data_labels
        dl.number_format, dl.number_format_is_linked = fmt, False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        dl.font.size = Pt(10)
        dl.font.bold = True
    for s, col in zip(plot.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = rgb(col)
    return gf


# ------------------------------------------------------------------ slides
def set_runs(shape, value):
    tf = shape.text_frame
    p = tf.paragraphs[0]
    p.runs[0].text = value
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    for extra in tf.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def chrome_shape(slide, y, x_min=None):
    for s in slide.shapes:
        if s.has_text_frame and abs(_y(s) - y) < 0.06 and (x_min is None or s.left / IN > x_min):
            return s
    return None


def new_slide(title, subtitle):
    s = prs.slides.add_slide(CHROME_SRC.slide_layout)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    tree = s.shapes._spTree
    ext = tree.find(qn("p:extLst"))
    for el in CHROME:
        (ext.addprevious if ext is not None else tree.append)(copy.deepcopy(el))
    set_runs(chrome_shape(s, 0.26), title)
    set_runs(chrome_shape(s, 0.62), subtitle)
    return s


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


ORDER = []

# 1. Title (template slide, edited)
s1 = TPL[0]
for shp in s1.shapes:
    if shp.has_text_frame and "Phase" in shp.text_frame.text and "number" in shp.text_frame.text:
        set_runs(shp, "Phase 1: Measuring and Classifying LLM Uncertainty")
notes(s1, "This is Phase 1 of the thesis. It covers the first two objectives: measuring how uncertain a "
          "language model is, and identifying what kind of uncertainty an answer has.")
ORDER.append(s1)

# 2. Contents
s = new_slide("Contents", "Presentation roadmap")
items = ["Introduction, problem statement and literature review",
         "Research gap, objectives and the Phase 1 plan",
         "Methodology: how uncertainty is measured and classified",
         "Phase 1 execution: plan vs result, implemented system",
         "Evaluation setup and results for every model",
         "Discussion, Phase 2 plan and references"]
for i, it in enumerate(items):
    yy = 1.12 + i * 0.6
    box(s, 0.9, yy + 0.09, 0.08, 0.06, fill=INDIGO, shape=MSO_SHAPE.OVAL)
    text(s, 1.18, yy, 7.8, 0.3, it, size=16)
notes(s, "The talk follows the usual order: the problem, what is missing in current work, our two Phase 1 "
         "objectives, how we built and checked the system, the first results, and the plan for Phase 2.")
ORDER.append(s)

# 3-4. Introduction and Problem Statement: the template's own slides, unchanged
notes(TPL[2], "Language models write fluent answers that can still be wrong. Some errors are about facts and "
              "need evidence; others are about reasoning and need a careful step-by-step check. Our framework "
              "first screens how unsure the model is, and only then verifies.")
notes(TPL[3], "Today's methods work separately, retrieved evidence can be noisy, and checking every answer the "
              "same way is slow and costly. So we want to check only the answers that are uncertain, and check "
              "them in the right way.")
ORDER += [TPL[2], TPL[3]]

# 5. Literature review
s = new_slide("Literature Review", "How uncertainty is measured today")
rows = [["Approach", "Key idea", "Limitation"],
        ["Semantic entropy (Kuhn 2023; Farquhar 2024)", "Sample several answers, group by meaning; more groups = less sure",
         "Needs 5–10 answers; misses errors made while sure"],
        ["Graph methods (Lin 2024)", "Treat the answers as a similarity graph", "Same sampling cost"],
        ["Token probability (Malinin & Gales 2021)", "Low word probability = unsure",
         "Mixes up wording with doubt"],
        ["Self-evaluation (Kadavath 2022; Tian 2023)", "Ask the model to rate its own answer",
         "Small models are over-confident"],
        ["Consistency checks (Manakul 2023)", "Compare each sentence with other answers", "Slow for long answers"],
        ["Uncertainty types (Hou 2024; Yadkori 2024)", "Separate “does not know” from “question is unclear”",
         "Not linked to a fix"]]
table(s, 0.5, 1.02, 8.95, [2.75, 3.4, 2.8], rows, row_h=0.43, size=10.5, bold_first_col=True)
box(s, 0.72, 4.18, 8.5, 0.6, fill=TINT, line=TINT_LINE,
    paras="Most work gives one uncertainty number, on a few models, and does not say what kind of doubt it is.",
    size=12, bold=True, color=NAVY, margin=0.2)
notes(s, "Semantic entropy is the best-known method: ask the same question several times and see whether the "
         "answers agree in meaning. Other methods use word probabilities or ask the model to judge itself. Newer "
         "work separates kinds of uncertainty, but none links the kind of doubt to the fix it needs.")
ORDER.append(s)

# 6. Research gap
s = new_slide("Research Gap", "What is missing, and how Phase 1 answers it")
gaps = [("Few models, few tasks", "Methods are usually compared on one or two models and datasets.",
         "Compare 17 methods on 11 datasets and up to 12 models.", "Objective 1"),
        ("One number, no reason", "A score says how unsure the model is, not why.",
         "Classify the kind of doubt: knowledge, reasoning, aleatoric or temporal.", "Objective 2"),
        ("No link to a fix", "The score does not say which check would help.",
         "Each type names its own fix.", "Bridge to Phase 2")]
for i, (head, gap, ans, obj) in enumerate(gaps):
    x = 0.6 + i * 3.0
    box(s, x, 1.05, 2.75, 1.2, fill=T_ROSE, anchor=MSO_ANCHOR.TOP, margin=0.15,
        paras=[[("GAP", {"size": 9, "bold": True, "color": ROSE})], [(head, {"size": 13, "bold": True})],
               [(gap, {"size": 11, "color": SLATE})]])
    arrow(s, x + 1.375, 2.3, x + 1.375, 2.62)
    box(s, x, 2.68, 2.75, 1.35, fill=TINT, line=TINT_LINE, anchor=MSO_ANCHOR.TOP, margin=0.15,
        paras=[[("PHASE 1 ANSWER", {"size": 9, "bold": True, "color": INDIGO})], [(ans, {"size": 12.5})],
               [(obj, {"size": 10.5, "bold": True, "color": BLUE})]])
notes(s, "Three things are missing. First, methods are compared on very few models and tasks, so we compare 17 "
         "methods widely. Second, an uncertainty score does not explain itself, so we classify the type of doubt. "
         "Third, the score is not linked to a fix; our types are, and Phase 2 will use them.")
ORDER.append(s)

# 7. Objectives (template slide, typos fixed, phase legend added)
s7 = TPL[4]
fixed = ["To measure an AI model's baseline uncertainty using an existing algorithm.",
         "To develop a framework for identifying and classifying uncertainty in AI-generated responses.",
         "To design an algorithm for reducing uncertainty using evidence-based factual verification and "
         "structured logical reasoning.",
         "To test and compare the developed integrated framework with existing LLM models in terms of "
         "faithfulness, hallucination reduction, and computational cost."]
for shp in s7.shapes:
    if shp.has_text_frame and "baseline uncertainty" in shp.text_frame.text:
        k = 0
        for p in shp.text_frame.paragraphs:
            if p.runs and "".join(r.text for r in p.runs).strip():
                p.runs[0].text = fixed[k]
                for r in p.runs[1:]:
                    r._r.getparent().remove(r._r)
                k += 1
box(s7, 0.75, 4.68, 4.15, 0.36, fill=TINT, line=TINT_LINE, paras="Phase 1 (this presentation): objectives 1 and 2",
    size=11, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
box(s7, 5.05, 4.68, 3.2, 0.36, fill=T_GREY, line=RULE, paras="Phase 2: objectives 3 and 4", size=11, bold=True,
    color=SLATE, align=PP_ALIGN.CENTER)
notes(s7, "There are four objectives. Phase 1, presented today, covers the first two: measuring uncertainty "
          "with an existing algorithm, and classifying it. Phase 2 covers reducing it and comparing the full system.")
ORDER.append(s7)

# 8. Phase 1 plan
s = new_slide("Phase 1 Plan: Measure and Classify Uncertainty", "Implementation sequence planned for Phase 1")
steps = [("Freeze the questions", "11 public datasets, 3,367 questions, fixed versions"),
         ("Choose the models", "5 model families, 3B to 14B parameters"),
         ("Ask each question", "1 normal answer, 10 samples and 5 self-checks"),
         ("Mark right or wrong", "rules for maths; a judge model from another family"),
         ("Score uncertainty (Objective 1)", "17 published methods, from the stored answers"),
         ("Label the type (Objective 2)", "re-ask with evidence or with a method hint"),
         ("Train the classifier (Objective 2)", "compare it with the current routing rule"),
         ("Run and report", "one command on a Mac; tables and figures")]
for i, (head, detail) in enumerate(steps):
    x = 0.75 + (i // 4) * 4.45
    y = 1.08 + (i % 4) * 0.98
    dot(s, x, y, 0.44, INDIGO, str(i + 1), size=14)
    text(s, x + 0.6, y - 0.02, 3.6, 0.3, head, size=13.5, bold=True)
    text(s, x + 0.6, y + 0.3, 3.6, 0.5, detail, size=11.5, color=SLATE)
notes(s, "The plan has eight steps. We fix the questions and the models, ask every question eleven times, mark "
         "each answer right or wrong, then score uncertainty for Objective 1 and learn the type of doubt for "
         "Objective 2. Everything runs with one command.")
ORDER.append(s)

# 9. Methodology: architecture
s = new_slide("Methodology: Generate Once, Score Many", "Phase 1 architecture")
row = [("Question sets", "11 datasets · 3,367 questions", T_INDIGO),
       ("LLM under test", "9–12 open models, 3B–14B", T_INDIGO),
       ("Generate", "11 answers + self-checks", T_INDIGO),
       ("Mark right / wrong", "rules or a judge model", T_INDIGO)]
for i, (h, d, f) in enumerate(row):
    x = 0.55 + i * 2.27
    box(s, x, 1.02, 1.98, 0.78, fill=f, line=TINT_LINE, align=PP_ALIGN.CENTER,
        paras=[[(h, {"size": 12, "bold": True})], [(d, {"size": 9.5, "color": SLATE})]])
    if i < 3:
        arrow(s, x + 1.98, 1.41, x + 2.27, 1.41)
arrow(s, 7.37 + 0.99, 1.8, 7.37 + 0.99, 2.08)
box(s, 0.55, 2.08, 8.79, 0.44, fill=BLUE, paras="Stored answers: every method reads the same data", size=12,
    bold=True, color=WHITE, align=PP_ALIGN.CENTER)
lanes = [(0.55, "Objective 1: how unsure is the model?", INDIGO, T_INDIGO,
          [("17 methods", "entropy and others"), ("Metrics", "AUROC, calibration"),
           ("Baseline profile", "per model and dataset")]),
         (5.05, "Objective 2: what kind of doubt?", TEAL, T_TEAL,
          [("Re-ask labels", "evidence or hint"), ("Detector + classifier", "risk and type"),
           ("Report + fix", "recommended action")])]
for x0, head, col, tint, steps2 in lanes:
    arrow(s, x0 + 0.65, 2.52, x0 + 0.65, 3.2)
    text(s, x0 + 0.85, 2.72, 3.5, 0.3, head, size=12, bold=True, color=col)
    for k, (h, d) in enumerate(steps2):
        x = x0 + k * 1.5
        box(s, x, 3.2, 1.3, 0.85, fill=tint, align=PP_ALIGN.CENTER, margin=0.05,
            paras=[[(h, {"size": 11, "bold": True})], [(d, {"size": 9, "color": SLATE})]])
        if k < 2:
            arrow(s, x + 1.3, 3.62, x + 1.5, 3.62)
text(s, 0.55, 4.45, 8.8, 0.5, "Figure 1. Phase 1 architecture. Each question is answered once and stored; Objective 1 "
     "and Objective 2 are both computed from the same stored answers, so every method is compared on identical data.",
     size=10.5, italic=True, color=SLATE)
notes(s, "This is the architecture. Each model answers each question once normally and ten more times, and rates "
         "its own answer. A judge marks the answers right or wrong. Everything is stored, and both objectives are "
         "computed from the same stored answers: Objective 1 scores how unsure the model is, Objective 2 says what kind "
         "of doubt it is and suggests a fix.")
ORDER.append(s)

# 10. Methodology: semantic entropy
s = new_slide("Methodology: Semantic Entropy", "The existing algorithm used for Objective 1 (Farquhar et al., Nature 2024)")
box(s, 0.55, 1.02, 4.3, 0.42, fill=TINT, paras="Q: What is the capital of Australia?", size=12, margin=0.15)
text(s, 0.55, 1.55, 4.3, 0.22, "The model answers 10 times", size=10, color=SLATE)
for i, k in enumerate([1, 1, 0, 1, 1, 0, 1, 1, 0, 1]):
    dot(s, 0.65 + i * 0.42, 1.82, 0.28, TEAL if k else AMBER)
arrow(s, 2.55, 2.18, 2.55, 2.5)
text(s, 2.7, 2.22, 2.0, 0.25, "group by meaning", size=10, color=SLATE)
dot(s, 0.75, 2.58, 1.3, TEAL, "Canberra\n7 of 10", size=12)
dot(s, 2.2, 2.78, 0.9, AMBER, "Sydney\n3 of 10", size=10.5)
box(s, 3.4, 2.7, 1.45, 1.0, fill=BLUE, align=PP_ALIGN.CENTER,
    paras=[[("H = 0.61", {"size": 19, "bold": True, "color": WHITE})], [("0 if all agree", {"size": 9.5, "color": TINT})]])
box(s, 5.2, 1.02, 4.15, 0.95, fill=TINT, margin=0.18,
    paras=[[("H = − Σ p(c) · ln p(c)", {"size": 20, "italic": True, "color": BLUE, "font": "Cambria"})],
           [("p(c) = share of the answers in meaning group c", {"size": 10.5, "color": INK})]])
bullets(s, 5.2, 2.2, 4.15, ["Ask the same question 10 times.",
                            "Group answers that mean the same (an NLI model checks both directions).",
                            "H = 0: all answers agree, the model is sure.",
                            "High H: answers scatter, likely a hallucination."], size=12.5, h=0.5, gap=0.22)
text(s, 0.55, 4.2, 4.3, 0.6, "Worked answers (maths, options, yes/no) are grouped by their final answer.",
     size=10.5, italic=True, color=SLATE)
notes(s, "Semantic entropy is the existing algorithm for Objective 1. We ask the same question ten times and group "
         "answers that mean the same thing. Here seven say Canberra and three say Sydney, so H is minus 0.7 log 0.7 "
         "minus 0.3 log 0.3, which is 0.61. If all ten agreed, H would be zero.")
ORDER.append(s)

# 11. Methodology: the 17 methods
s = new_slide("Methodology: Seventeen Methods", "Objective 1 compares semantic entropy with 16 other published methods")
R = lambda t: (t, {})
SUB = lambda t: (t, {"sub": True})
cards = [("9", "Meaning-level", "How much the answers' meanings spread", TEAL,
          [R("H = −Σ p(c) ln p(c)      EigV = Σ"), SUB("k"), R(" max(0, 1 − λ"), SUB("k"), R(")")],
          "semantic entropy (3 forms), number of groups, graph methods (3), disagreement, SelfCheckGPT"),
         ("5", "Token-level", "The model's own word probabilities", INDIGO,
          [R("NLL = −(1/L) Σ"), SUB("i"), R(" ln p(t"), SUB("i"), R(")")],
          "sequence and average NLL, token entropy (mean, max), predictive entropy"),
         ("1", "Surface", "Word overlap between the answers", AMBER,
          [R("1 − mean ROUGE-L(s"), SUB("i"), R(", s"), SUB("j"), R(")")], "lexical similarity"),
         ("2", "Self-evaluation", "The model judges its own answer", ROSE,
          [R("P(True | question, samples, answer)")], "P(True), stated confidence from 0 to 100")]
for i, (count, head, desc, col, formula, ex) in enumerate(cards):
    x = 0.55 + (i % 2) * 4.45
    y = 1.02 + (i // 2) * 1.95
    box(s, x, y, 4.25, 1.78, fill="F8FAFC", line=RULE)
    dot(s, x + 0.17, y + 0.17, 0.48, col, count, size=15)
    text(s, x + 0.8, y + 0.15, 3.3, 0.3, head, size=14, bold=True)
    text(s, x + 0.8, y + 0.47, 3.3, 0.28, desc, size=11, color=SLATE)
    text(s, x + 0.2, y + 0.88, 3.9, 0.36, [[(t, {**o, "italic": True, "color": BLUE, "font": "Cambria"}) for t, o in formula]],
         size=14)
    text(s, x + 0.2, y + 1.27, 3.9, 0.45, ex, size=9.5, color=SLATE)
notes(s, "We do not test semantic entropy alone. It is compared with sixteen other published methods in four "
         "families: how the meanings spread, the model's own word probabilities, simple word overlap, and asking the "
         "model to judge itself. All of them are computed from the same stored answers.")
ORDER.append(s)

# 12. Methodology: uncertainty types
s = new_slide("Methodology: Types of Uncertainty", "Objective 2: how unsure, and what kind of doubt")
for i, (h, f, out) in enumerate([("Identify: how unsure?", "P(hallucination) from the 17 signals, calibrated",
                                  "→ low · medium · high"),
                                 ("Classify: what kind?", "one of four types of doubt",
                                  "→ the fix that suits that type")]):
    box(s, 0.55 + i * 4.45, 1.02, 4.25, 1.0, fill=BLUE, anchor=MSO_ANCHOR.MIDDLE, margin=0.18,
        paras=[[(h, {"size": 14, "bold": True, "color": WHITE})], [(f, {"size": 11.5, "color": WHITE})],
               [(out, {"size": 11, "color": "FDE68A", "bold": True})]])
rows = [["Type", "What the answers look like", "What fixes it"],
        ["Knowledge", "Answers disagree on names, dates or facts", "Look up evidence"],
        ["Reasoning", "Same facts, different conclusions", "Careful step-by-step reasoning"],
        ["Aleatoric", "The question is unclear, unanswerable or based on a false premise",
         "Ask to clarify, refuse, or correct the premise"],
        ["Temporal", "The true answer changes over time", "Search for fresh information"]]
table(s, 0.55, 2.25, 8.7, [1.55, 4.15, 3.0], rows, row_h=0.46, size=11, bold_first_col=True,
      fills={(1, 0): T_TEAL, (2, 0): T_INDIGO, (3, 0): T_AMBER, (4, 0): T_ROSE})
notes(s, "Objective 2 answers two questions about an answer. First, how likely it is to be wrong. Second, what kind "
         "of doubt it has. Each of the four types needs a different fix: missing knowledge needs evidence, a reasoning "
         "slip needs careful steps, an unclear question needs clarification, and a changing fact needs fresh data.")
ORDER.append(s)

# 13. Methodology: labelling
s = new_slide("Methodology: Labelling the Type", "Objective 2: the type is whichever re-ask removes the doubt")
box(s, 0.55, 1.62, 1.5, 0.95, fill=BLUE, paras="Unsure or wrong answer", size=12, bold=True, color=WHITE,
    align=PP_ALIGN.CENTER)
box(s, 2.45, 1.2, 1.7, 0.78, fill=TINT, align=PP_ALIGN.CENTER,
    paras=[[("Re-ask with the true evidence", {"size": 11})], [("5 answers", {"size": 9.5, "color": SLATE})]])
box(s, 2.45, 2.2, 1.7, 0.78, fill=TINT, align=PP_ALIGN.CENTER,
    paras=[[("Re-ask with a method hint", {"size": 11})], [("5 answers", {"size": 9.5, "color": SLATE})]])
arrow(s, 2.05, 2.1, 2.45, 1.59)
arrow(s, 2.05, 2.1, 2.45, 2.59)
box(s, 4.5, 1.27, 1.45, 0.64, fill=T_TEAL, paras="Fixed → knowledge", size=11, bold=True, align=PP_ALIGN.CENTER)
box(s, 4.5, 2.27, 1.45, 0.64, fill=T_INDIGO, paras="Fixed → reasoning", size=11, bold=True, align=PP_ALIGN.CENTER)
arrow(s, 4.15, 1.59, 4.5, 1.59)
arrow(s, 4.15, 2.59, 4.5, 2.59)
box(s, 0.55, 3.3, 2.6, 0.62, line=TINT_LINE, paras="Dataset labels\n→ aleatoric, temporal", size=11)
box(s, 3.3, 3.3, 2.65, 0.62, line=TINT_LINE, paras="~300 human labels\n→ checked with Cohen's κ", size=11)
text(s, 0.55, 4.15, 5.4, 0.6, "If the evidence fixes the answer, the model lacked knowledge; if the hint fixes it, "
     "the reasoning was the problem.", size=11, italic=True, color=SLATE)
tag(s, 6.35, 1.1, "THE CLASSIFIER")
bullets(s, 6.35, 1.5, 3.05, ["Inputs: the 17 signals, how the answers group, and 5 self-checks",
                             "Compared: the current routing rule, a small decision tree, two learned models",
                             "Tested on unseen datasets and models",
                             "Goal: beat the routing rule clearly (95% CI above zero)"],
        size=11, h=0.72, gap=0.22)
notes(s, "To train the classifier we need labels. For an unsure answer we ask the model again, once with the true "
         "evidence and once with the first half of a worked method. If the evidence fixes it, the doubt was about "
         "knowledge; if the hint fixes it, about reasoning. Unclear and time-dependent questions are labelled by their "
         "datasets, and about 300 answers are checked by hand.")
ORDER.append(s)

# 14. Plan, execution and result
s = new_slide("Phase 1: Plan, Execution and Result", "What we planned, what we built, and what we measured")
DONE, PEND = ("Done", {}), ("Pending", {})
rows = [["Part", "Planned", "Built and checked", "Status"],
        ["Question sets", "11 datasets, fixed versions", "3,367 questions; rebuild gives identical files", "Done"],
        ["Answer generation", "1 answer + 10 samples + self-checks", "Runner that resumes after a crash", "Done"],
        ["Right / wrong labels", "Rules + independent judge", "Second judge to measure agreement", "Done"],
        ["Objective 1", "17 methods, metrics, statistics", "Reports, tables and figures", "Done"],
        ["Objective 2", "Detector + type classifier", "Re-ask labels, 3 classifiers, analyze tool", "Done"],
        ["Full study", "Up to 12 models on a Mac", "One command; about one week of running", "Pending"]]
fills = {(r, 3): (T_TEAL if rows[r][3] == "Done" else T_AMBER) for r in range(1, len(rows))}
table(s, 0.5, 1.02, 8.95, [1.75, 2.75, 3.4, 1.05], rows, row_h=0.4, size=10.5, bold_first_col=True, fills=fills)
box(s, 0.72, 4.0, 8.5, 0.85, fill=TINT, line=TINT_LINE, margin=0.2,
    paras=[[("Result in short: ", {"bold": True}), ("all Phase 1 code is built and tested. Early real data from two "
            "models shows that semantic entropy works on maths (AUROC 0.86) but not on myth questions (0.55).", {})]], size=11.5, color=NAVY)
notes(s, "This table compares plan and execution. Every part of Phase 1 is built and tested; only the full run on "
         "the Mac is pending. Even before that run, stored results from the pipeline give us a first real finding, "
         "shown in the results section.")
ORDER.append(s)

# 15. Implemented system
s = new_slide("Phase 1 Execution: Implemented System", "What was built and what it uses")
mods = [("Question-set builder", "downloads and freezes the 11 sets"),
        ("Answer generator", "11 answers and self-checks"),
        ("Right / wrong labeller", "rules or a cross-family judge"),
        ("Uncertainty scorer", "17 methods from stored answers"),
        ("Objective 1 reports", "AUROC, calibration, figures"),
        ("Objective 2 framework", "detector, classifier, analyze tool"),
        ("One-command runner", "checks, pilot, all models, resume")]
cols = [TEAL, TEAL, INDIGO, INDIGO, INDIGO, AMBER, ROSE]
for i, ((h, d), col) in enumerate(zip(mods, cols)):
    y = 1.02 + i * 0.5
    box(s, 0.55, y, 4.6, 0.42, fill="F8FAFC", line=RULE)
    dot(s, 0.65, y + 0.08, 0.26, col, str(i + 1), size=9)
    text(s, 1.05, y + 0.06, 1.85, 0.3, h, size=11, bold=True)
    text(s, 2.95, y + 0.07, 2.15, 0.3, d, size=9.5, color=SLATE)
tag(s, 5.45, 1.02, "BUILT WITH")
table(s, 5.45, 1.34, 3.95, [1.6, 2.35], [["Tool", "Used for"], ["Ollama", "runs the open LLMs locally"],
                                         ["DeBERTa-v3 NLI", "same-meaning check"],
                                         ["scikit-learn", "classifiers and metrics"],
                                         ["pandas, SciPy", "tables and statistics"]], row_h=0.34, size=10)
for i, (num, lab) in enumerate([("3,367", "questions"), ("17", "methods"), ("45", "tests pass"), ("12", "models max")]):
    x = 5.45 + (i % 2) * 2.0
    y = 3.2 + (i // 2) * 0.82
    box(s, x, y, 1.9, 0.7, fill=TINT, align=PP_ALIGN.CENTER,
        paras=[[(num, {"size": 18, "bold": True, "color": BLUE})], [(lab, {"size": 9.5, "color": SLATE})]])
notes(s, "The system has seven parts, from building the question sets to a single command that runs everything. "
         "It uses Ollama to run open models locally, a DeBERTa model to check whether two answers mean the same, and "
         "standard Python libraries for statistics. All 45 automatic tests pass.")
ORDER.append(s)

# 16. Evaluation setup
s = new_slide("Evaluation Setup", "Datasets, configurations, metrics and statistics")
tag(s, 0.55, 1.0, "DATASETS")
table(s, 0.55, 1.3, 4.3, [1.75, 1.85, 0.7], [["Doubt probed", "Datasets", "Qs"],
                                            ["Missing knowledge", "TriviaQA, PopQA, HaluEval", "900"],
                                            ["Common myths", "TruthfulQA", "817"],
                                            ["Reasoning", "GSM8K, MATH, StrategyQA, ARC", "950"],
                                            ["Unclear / unanswerable", "AmbigQA, SelfAware", "500"],
                                            ["Changing facts", "FreshQA", "200"]],
      row_h=0.3, size=9, aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT])
tag(s, 5.1, 1.0, "CONFIGURATIONS")
bullets(s, 5.1, 1.32, 4.3, ["9–12 open LLMs, 3B–14B, run locally", "1 normal answer + 10 samples (temperature 1)",
                            "Tune on dev (30%), report on test (70%)", "Judge: a model from another family"],
        size=10.5, step=0.36, h=0.32)
tag(s, 0.55, 3.25, "METRICS")
bullets(s, 0.55, 3.57, 4.3, ["AUROC: are wrong answers ranked as more unsure?", "AURAC: accuracy when unsure answers are refused",
                             "ECE: is a stated 70% right 70% of the time?", "Macro-F1: quality of the type classifier"],
        size=10.5, step=0.36, h=0.32)
tag(s, 5.1, 3.25, "STATISTICS")
bullets(s, 5.1, 3.57, 4.3, ["95% bootstrap confidence intervals", "DeLong test to compare two methods",
                            "Grouped 5-fold cross-validation", "Leave one dataset, model or family out"],
        size=10.5, step=0.36, h=0.32)
notes(s, "We use eleven public datasets, chosen so every kind of doubt appears. Each model answers each question once "
         "normally and ten more times. Thresholds are tuned on the dev part only. The main metric is AUROC: the "
         "chance that a wrong answer gets a higher uncertainty than a right one. Differences are tested statistically.")
ORDER.append(s)

# 17-19. Results: one table per group of datasets, one row per model
REAL_NOTE = {"Facts and myths": " Real: qwen2.5:3b and llama2:7b accuracy and AUROC on TruthfulQA (earlier stored runs).",
             "Reasoning": " Real: qwen2.5:3b and llama2:7b accuracy and AUROC on GSM8K (earlier stored runs)."}
GROUP_NOTES = {
    "Facts and myths": "Each row is one model; each cell lists the values in the order of the datasets column. Fact "
                       "questions show a useful AUROC, well above 0.5. TruthfulQA, the myth questions, stays close to "
                       "0.5 for every model: entropy cannot separate right from wrong there.",
    "Reasoning": "Maths and reasoning give the clearest signal: when the model is wrong, its ten answers scatter, so "
                 "AUROC is high. Accuracy grows with model size inside each family.",
    "Unclear and changing questions": "These sets hold ambiguous, unanswerable and time-dependent questions. The "
                                      "blind spot is largest here: models often give the same wrong answer every "
                                      "time, so sampling alone misses it. This is why Objective 2 classifies the type."}
for k, (group, sets) in enumerate(GROUPS, start=1):
    s = new_slide(f"Results: Baseline Uncertainty per Model ({k}/3)",
                  f"{group}: accuracy, semantic-entropy AUROC and blind spot for every model")
    names = ", ".join(sets)
    rows = [["Model", "Datasets", "Accuracy", "AUROC (semantic entropy)", "Blind spot"]]
    for model, _, _ in MODELS:
        v = [PH[model, d] for d in sets]
        rows.append([model, names, ", ".join(f"{x[0]:.0%}" for x in v), ", ".join(f"{x[1]:.2f}" for x in v),
                     ", ".join(f"{x[2]:.0%}" for x in v)])
    table(s, 0.6, 1.0, 8.8, [1.3, 2.6, 1.55, 1.85, 1.5], rows, row_h=0.27, size=8.5, bold_first_col=True)
    placeholder_note(s, 4.66, "Each cell lists values in the order of the Datasets column. AUROC: 0.5 = guessing, "
                              "1 = perfect. Blind spot: share of errors made while sure." + REAL_NOTE.get(group, ""))
    notes(s, GROUP_NOTES[group] + (" These values are placeholders until the full run." if PLACEHOLDER else ""))
    ORDER.append(s)

# 20. Results: maths vs myths across every model
s = new_slide("Results: Maths vs Myths Across Models", "Semantic-entropy AUROC on GSM8K (maths) and TruthfulQA (myths)")
cats = [m for m, _, _ in MODELS]
maths = [PH[m, "GSM8K"][1] for m in cats]
myths = [PH[m, "TruthfulQA"][1] for m in cats]
column_chart(s, 0.5, 0.95, 8.95, 2.85, cats, [("GSM8K (maths)", tuple(round(v, 2) for v in maths)),
                                              ("TruthfulQA (myths)", tuple(round(v, 2) for v in myths))],
             [INDIGO, AMBER], "How well entropy spots errors (AUROC; 0.5 = guessing)", labels=False, cat_size=8.5)
better = sum(a > b for a, b in zip(maths, myths))
kpis = [(f"{sum(maths) / len(maths):.2f}", "average AUROC on maths: a strong signal"),
        (f"{sum(myths) / len(myths):.2f}", "average AUROC on myths: close to guessing"),
        (f"{better} of {len(cats)}", "models where entropy works better on maths")]
for i, (num, lab) in enumerate(kpis):
    box(s, 0.6 + i * 2.95, 3.9, 2.8, 0.66, fill=TINT, anchor=MSO_ANCHOR.MIDDLE, margin=0.15,
        paras=[[(num + "  ", {"size": 17, "bold": True, "color": BLUE}), (lab, {"size": 9.5, "color": INK})]])
placeholder_note(s, 4.66, "qwen2.5:3b and llama2:7b bars are real (earlier stored runs).")
notes(s, "This compares maths and myth questions for every model. On maths, wrong answers scatter across the ten "
         "samples, so entropy spots them. On myth questions, right and wrong answers look equally unsure, so entropy "
         "is close to guessing. This is why Objective 2 looks at the kind of doubt, not only its size."
         + (" Values are placeholders until the full run; the qwen2.5:3b and llama2:7b bars are real." if PLACEHOLDER
            else ""))
ORDER.append(s)

# 21. Results: readiness and pending outputs
s = new_slide("Results: Readiness of the Full Study", "Checks on the code (not model results) and what the full run adds")
tag(s, 0.55, 1.0, "VERIFIED NOW", w=1.6)
checks = [("45", "automatic tests pass"), ("6 / 6", "run stages finished in a 12-model test run"),
          ("5 of 30", "questions redone after a simulated crash"), ("> 80%", "of type labels matched in a simulated test")]
for i, (num, lab) in enumerate(checks):
    y = 1.32 + i * 0.82
    box(s, 0.55, y, 3.6, 0.7, fill=T_TEAL, anchor=MSO_ANCHOR.MIDDLE, margin=0.15,
        paras=[[(num, {"size": 16, "bold": True, "color": TEAL})], [(lab, {"size": 10.5})]])
tag(s, 4.45, 1.0, "PENDING: FULL RUN", w=1.8)
rows = [["Output", "Objective", "Status"],
        ["AUROC of 17 methods × 11 datasets × up to 12 models", "1", "Pending"],
        ["Blind spot and calibration per model", "1", "Pending"],
        ["How many samples are needed (2 to 10)", "1", "Pending"],
        ["Detector vs semantic entropy", "2", "Pending"],
        ["Type classifier vs routing rule (macro-F1)", "2", "Pending"],
        ["Agreement with ~300 human labels (κ)", "2", "Pending"]]
table(s, 4.45, 1.32, 4.95, [3.1, 0.85, 1.0], rows, row_h=0.42, size=10,
      fills={(r, 2): T_AMBER for r in range(1, 7)}, aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
notes(s, "The left side shows what is already verified about the code, using a "
         "simulated model. The right side lists the results the full run will produce; they will replace this slide.")
ORDER.append(s)

# 22. Discussion and limitations
s = new_slide("Discussion and Limitations", "How to read the Phase 1 results")
bullets(s, 0.75, 1.1, 8.6, [
    "Result tables show placeholder values; only qwen2.5:3b and llama2:7b on GSM8K and TruthfulQA are real so far.",
    "Entropy works when answers are short and checkable (maths); on open questions it is high for right and wrong alike.",
    "Some errors are made while the model is sure (blind spot); sampling cannot see them, so self-checks are also tested.",
    "A judge model marks free-text answers; a second judge and 100 human checks test it.",
    "Type labels come from re-asking; about 300 human labels check them.",
    "Only small open models (3B–14B) are tested; larger models may behave differently."], size=12.5, h=0.55, gap=0.26)
notes(s, "A few cautions. The result tables are placeholders until the full run; the few real numbers are small. Entropy is weak on "
         "open-ended questions. Some confident errors cannot be caught by sampling. Judges and our type labels are "
         "checked against humans. And we only test small open models.")
ORDER.append(s)

# 23. Phase 2 plan
s = new_slide("Phase 2 Plan: Reduce Uncertainty and Compare", "Implementation sequence planned for Phase 2")
p2 = [("Route by type", "use the Phase 1 type to pick the fix for each unsure answer"),
      ("Evidence-based verification", "check factual claims against retrieved passages (Objective 3)"),
      ("Structured reasoning", "plan, worked samples and small programs, then vote (Objective 3)"),
      ("Compare with base LLMs", "faithfulness, hallucination reduction and cost (Objective 4)")]
for i, (h, d) in enumerate(p2):
    y = 1.08 + i * 0.75
    dot(s, 0.75, y, 0.44, INDIGO, str(i + 1), size=14)
    text(s, 1.35, y - 0.02, 7.9, 0.3, h, size=13.5, bold=True)
    text(s, 1.35, y + 0.3, 7.9, 0.3, d, size=11.5, color=SLATE)
box(s, 0.72, 4.15, 8.5, 0.85, fill=TINT, line=TINT_LINE, margin=0.2,
    paras=[[("Head start: ", {"bold": True}), ("a first pipeline prototype already exists. On 100 test questions it "
            "raised GSM8K accuracy from 84% to 88%, but TruthfulQA did not improve (66% → 63%). Phase 1's uncertainty "
            "types will tell it where to act.", {})]], size=11, color=NAVY)
notes(s, "Phase 2 uses what Phase 1 learns. Each unsure answer gets the fix that suits its type: evidence for missing "
         "knowledge, careful reasoning for slips. Then we compare the whole system with the base models. A prototype "
         "already helps on maths, but not yet on myth questions, which is exactly where Phase 1's types should help.")
ORDER.append(s)

# 24. References
s = new_slide("References", "Main sources")
refs = ["S. Farquhar, J. Kossen, L. Kuhn, Y. Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 2024.",
        "L. Kuhn, Y. Gal, S. Farquhar. Semantic uncertainty: linguistic invariances for uncertainty estimation in NLG. ICLR, 2023.",
        "Z. Lin, S. Trivedi, J. Sun. Generating with confidence: uncertainty quantification for black-box LLMs. TMLR, 2024.",
        "A. Malinin, M. Gales. Uncertainty estimation in autoregressive structured prediction. ICLR, 2021.",
        "S. Kadavath et al. Language models (mostly) know what they know. arXiv, 2022.",
        "K. Tian et al. Just ask for calibration. EMNLP, 2023.",
        "M. Xiong et al. Can LLMs express their uncertainty? ICLR, 2024.",
        "P. Manakul, A. Liusie, M. Gales. SelfCheckGPT: zero-resource hallucination detection. EMNLP, 2023.",
        "B. Hou et al. Decomposing uncertainty for LLMs through input clarification ensembling. ICML, 2024.",
        "Y. Abbasi-Yadkori et al. To believe or not to believe your LLM. arXiv, 2024.",
        "X. Wang et al. Self-consistency improves chain of thought reasoning in language models. ICLR, 2023.",
        "S. Lin, J. Hilton, O. Evans. TruthfulQA: measuring how models mimic human falsehoods. ACL, 2022."]
for i, r in enumerate(refs):
    x = 0.6 + (i // 6) * 4.45
    y = 1.05 + (i % 6) * 0.62
    text(s, x, y, 0.35, 0.3, f"[{i + 1}]", size=10, bold=True, color=BLUE)
    text(s, x + 0.4, y, 3.95, 0.58, r, size=9.5)
notes(s, "These are the main papers behind the methods compared in Phase 1.")
ORDER.append(s)

# ------------------------------------------------------------------ order, page numbers, cleanup
sld = prs.slides._sldIdLst
by_id = {el.get("id"): el for el in sld}
keep = [str(sl.slide_id) for sl in ORDER]
for el in list(sld):
    sld.remove(el)
for sid in keep:
    sld.append(by_id[sid])
for sid, el in by_id.items():
    if sid not in keep:
        prs.part.drop_rel(el.get(qn("r:id")))
for n, sl in enumerate(ORDER, start=1):
    num = chrome_shape(sl, 5.38, x_min=9.0)
    if num is not None:
        set_runs(num, f"{n:02d}")
prs.save(OUT)
print(f"wrote {OUT} with {len(ORDER)} slides")
