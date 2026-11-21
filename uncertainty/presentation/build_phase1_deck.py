"""
Phase 1 deck (thesis objectives 1-2) built on the college template.

Keeps the template's title, Introduction and Problem Statement slides as written, edits Contents
and Objectives, and adds every other slide with the template's own chrome (blue bar, title,
subtitle, rule, footer) copied from its content slides.

Result slides: until the full study has run on the Mac, the per-model tables and the
across-model chart show EXPECTED values (published benchmarks and published semantic-entropy
results, see EXP below), labelled "EXPECTED VALUES" on every slide. They are not measurements of
this study. Four cells are real numbers from the pipeline's stored test runs (RESULTS).
After the Mac run, put the measured numbers in EXP and set EXPECTED = False.

Usage: python build_phase1_deck.py <template.pptx> <output.pptx>
"""
import copy
import math
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

# Expected values for the full-study result slides. NOT measured in this study; every slide that shows
# them carries the "EXPECTED VALUES" tag. Accuracy: the published score where a model card or report gives
# one (list and links in Phase1_Presentation_Script.md, Part D), otherwise an estimate from the model's
# size, family and those published scores. AUROC: the level semantic entropy reaches per kind of dataset
# in published work (Farquhar et al., 2024: 0.79 on average) and in our stored runs, nudged by model
# strength. The four RESULTS cells are real. After the Mac run: measured numbers in EXP, EXPECTED = False.
EXPECTED = True
MODELS = [  # roster model, family, rough strength (0-1) used only for the small AUROC adjustment
    ("qwen2.5:3b", "qwen", .50), ("qwen2.5:7b", "qwen", .66), ("qwen2.5:14b", "qwen", .78),
    ("qwen3:4b", "qwen", .62), ("llama3.2:3b", "llama", .40), ("llama3.1:8b", "llama", .60),
    ("llama2:7b", "llama", .22), ("gemma3:4b", "gemma", .52), ("gemma3:12b", "gemma", .74),
    ("phi4-mini", "phi", .48), ("phi4", "phi", .80), ("mistral:7b", "mistral", .42)]
# The models shown on the result slides (the study itself runs the whole roster above)
SHOWN = ["qwen2.5:3b", "qwen2.5:7b", "qwen3:4b", "llama3.2:3b", "llama2:7b", "llama3.1:8b"]
SETS = ["TriviaQA", "PopQA", "HaluEval", "TruthfulQA", "GSM8K", "MATH", "StrategyQA", "ARC", "AmbigQA",
        "SelfAware", "FreshQA"]
ACCURACY = {  # expected accuracy, in SETS order
    "qwen2.5:3b":  [.55, .18, .30, .66, .84, .60, .64, .78, .30, .42, .24],
    "qwen2.5:7b":  [.62, .22, .36, .70, .92, .76, .70, .87, .34, .46, .28],
    "qwen2.5:14b": [.70, .27, .42, .74, .95, .80, .75, .92, .38, .50, .32],
    "qwen3:4b":    [.52, .18, .32, .68, .91, .80, .70, .86, .32, .46, .27],
    "llama3.2:3b": [.60, .20, .30, .58, .78, .48, .64, .79, .30, .38, .22],
    "llama3.1:8b": [.74, .27, .40, .64, .85, .52, .70, .83, .36, .44, .28],
    "llama2:7b":   [.64, .22, .28, .61, .24, .04, .60, .50, .28, .34, .18],
    "gemma3:4b":   [.66, .22, .33, .62, .89, .76, .68, .80, .32, .42, .25],
    "gemma3:12b":  [.78, .28, .42, .70, .94, .84, .76, .89, .38, .50, .31],
    "phi4-mini":   [.48, .15, .28, .62, .89, .64, .68, .84, .30, .44, .22],
    "phi4":        [.66, .23, .40, .74, .93, .80, .78, .93, .38, .52, .30],
    "mistral:7b":  [.70, .25, .34, .60, .48, .12, .64, .66, .32, .38, .22],
}
AUROC_LEVEL = {"TriviaQA": .80, "PopQA": .77, "HaluEval": .74, "TruthfulQA": .57, "GSM8K": .83, "MATH": .80,
               "StrategyQA": .66, "ARC": .74, "AmbigQA": .62, "SelfAware": .64, "FreshQA": .65}
GROUPS = [("Facts and myths", ["TriviaQA", "PopQA", "HaluEval", "TruthfulQA"]),
          ("Reasoning", ["GSM8K", "MATH", "StrategyQA", "ARC"]),
          ("Unclear and changing questions", ["AmbigQA", "SelfAware", "FreshQA"])]


def _expected():
    """(model, dataset) -> [accuracy, semantic-entropy AUROC]; the four real cells override."""
    out = {}
    for model, _, s in MODELS:
        for d, acc in zip(SETS, ACCURACY[model]):
            out[model, d] = [acc, round(min(.95, max(.50, AUROC_LEVEL[d] + .06 * (s - .5))), 2)]
    for m, d, acc, au, *_ in RESULTS:
        out[m, d.split(" ")[0]] = [acc, au]
    return out


EXP = _expected()

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


def basis_note(slide, y, note):
    """The amber 'expected values' tag and a one-line basis; only the note once results are measured."""
    if EXPECTED:
        box(slide, 0.6, y, 1.55, 0.24, fill=T_AMBER, paras="EXPECTED VALUES", size=8, bold=True, color="92400E",
            margin=0.06, radius=0.2, align=PP_ALIGN.CENTER)
        note = "Not yet measured here; based on published results. " + note
    text(slide, 2.3 if EXPECTED else 0.6, y + 0.03, 7.15, 0.22, note, size=8.5, italic=True, color=SLATE)


def table(slide, x, y, w, col_w, rows, row_h=0.36, size=10.5, header_fill=BLUE, fills=None, bold_first_col=False,
          aligns=None, pad=0.03):
    """rows[0] is the header. fills: per-cell override {(r, c): hex}. pad: top/bottom cell margin (dense tables)."""
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
            cell.margin_top = cell.margin_bottom = E(pad)
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
                 legend=True, labels=True, cat_size=10, label_size=10, title_size=11, legend_size=10):
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
        style_run(r, size=title_size, color=INK, bold=True)
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(legend_size)
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
        dl.font.size = Pt(label_size)
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
         "Implementation and evaluation setup",
         "Objective 1: measuring uncertainty (methods, then results)",
         "Objective 2: classifying uncertainty (method, then result)",
         "Plan vs execution, discussion, Phase 2 plan and references"]
for i, it in enumerate(items):
    yy = 1.12 + i * 0.6
    box(s, 0.9, yy + 0.09, 0.08, 0.06, fill=INDIGO, shape=MSO_SHAPE.OVAL)
    text(s, 1.18, yy, 7.8, 0.3, it, size=16)
notes(s, "The talk follows the usual order: the problem, what is missing in current work, our two Phase 1 "
         "objectives and the setup. Then each objective comes with its own results: first how we measure "
         "uncertainty and what we find, then how we classify it. We close with plan versus execution, the "
         "discussion and the plan for Phase 2.")
ORDER.append(s)

# 3-4. Introduction and Problem Statement: the template's own slides, unchanged
notes(TPL[2], "Language models write fluent answers that can still be wrong. Some errors are about facts and "
              "need evidence; others are about reasoning and need a careful step-by-step check. Our framework "
              "first screens how unsure the model is, and only then verifies.")
notes(TPL[3], "Today's methods work separately, retrieved evidence can be noisy, and checking every answer the "
              "same way is slow and costly. So we want to check only the answers that are uncertain, and check "
              "them in the right way.")
ORDER += [TPL[2], TPL[3]]

# 5. Literature review: the papers studied, one row each, in time order (slide 11 compares the methods themselves)
s = new_slide("Literature Review",
              "Key papers 2021–2024: from word probabilities, to the meaning of the samples, to the kind of doubt")
LR_PAPERS = [  # year, "[n] authors", venue, what they did, key finding
    ("2021", "[4] Malinin & Gales", "ICLR", "Token-level entropy for text generation",
     "Word probabilities show how unsure the model is"),
    ("2022", "[5] Kadavath et al.", "arXiv", "P(True): the model grades its own answer",
     "Large models can judge if their answer is true"),
    ("2022", "[12] Lin, Hilton & Evans", "ACL", "TruthfulQA benchmark of common myths",
     "Models repeat myths; larger ones often less truthful"),
    ("2023", "[2] Kuhn et al.", "ICLR", "Semantic entropy over meaning groups", "Counting meanings beats counting wordings"),
    ("2023", "[8] Manakul et al.", "EMNLP", "SelfCheckGPT: compare the samples", "Inconsistent samples reveal hallucinations"),
    ("2023", "[11] Wang et al.", "ICLR", "Self-consistency: vote over samples", "The most common answer is more often right"),
    ("2023", "[6] Tian et al.", "EMNLP", "Ask the model for its confidence", "Stated confidence can be better calibrated"),
    ("2024", "[1] Farquhar et al.", "Nature", "Semantic entropy to detect hallucinations",
     "Flags confabulations across 30 task–model pairs"),
    ("2024", "[3] Lin, Trivedi & Sun", "TMLR", "Graph methods on the sampled answers", "Works for black-box models"),
    ("2024", "[7] Xiong et al.", "ICLR", "Study of stated confidence", "Models over-state their confidence"),
    ("2024", "[9] Hou et al.", "ICML", "Clarify the question, then compare", "Splits doubt: unclear question vs missing knowledge"),
    ("2024", "[10] Abbasi-Yadkori et al.", "arXiv", "Iterative prompting", "Tells missing knowledge from natural randomness")]
rows = [["Year", "Paper", "Venue", "What they did", "Key finding"]]
for i, (year, paper, venue, did, finding) in enumerate(LR_PAPERS):
    first = i == 0 or LR_PAPERS[i - 1][0] != year  # show each year once, on its first row
    rows.append([[[(year if first else "", {"bold": True, "color": BLUE})]], [[(paper, {"bold": True})]], venue, did,
                 finding])
table(s, 0.5, 1.0, 8.95, [0.55, 1.85, 0.7, 2.55, 3.3], rows, row_h=0.285, size=8.5)
notes(s, "These are the papers I studied on how to measure a language model's uncertainty, in time order. In 2021 "
         "Malinin and Gales measured uncertainty from the model's own word probabilities. In 2022 Kadavath and colleagues "
         "showed that large models can judge their own answers, and TruthfulQA showed that models repeat common myths. In "
         "2023 the field moved to the meaning of several sampled answers: semantic entropy by Kuhn and colleagues, "
         "SelfCheckGPT and self-consistency; Tian showed that simply asking for a confidence can work. In 2024 semantic "
         "entropy reached Nature, graph methods made it work for black-box models, Xiong found that stated confidence is "
         "over-confident, and Hou and Abbasi-Yadkori began to separate kinds of doubt. So the trend is from word "
         "probabilities, to meaning, to the kind of doubt; what is still missing is linking the kind of doubt to the fix "
         "it needs, which is the research gap on the next slide.")
ORDER.append(s)

# 6. Research gap
s = new_slide("Research Gap", "What is missing, and how Phase 1 answers it")
gaps = [("Few models, few tasks", "Methods are usually compared on one or two models and datasets.",
         "Compare 17 methods on 11 datasets and 6 models (Qwen, Llama).", "Objective 1"),
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
               *([[("", {"size": 12.5})]] * {0: 0, 1: 0, 2: 2}[i]),
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

# 8. Phase 1 plan and architecture: the plan's eight steps, numbered on the architecture diagram
s = new_slide("Phase 1 Plan and Architecture", "Generate once, score many: eight steps, and every method reads the same stored answers")


def step_no(slide, x, y, n):
    dot(slide, x, y, 0.27, NAVY, str(n), size=9)


row = [("Question sets", ["11 datasets · 3,367 questions", "fixed versions"]),
       ("LLMs under test", ["6 open models", "Qwen and Llama · 3B–8B"]),
       ("Generate", ["1 answer + 10 samples", "+ self-checks"]),
       ("Mark right / wrong", ["rules for maths; a judge", "from another family"])]
for i, (h, d) in enumerate(row):
    x = 0.55 + i * 2.27
    box(s, x, 1.02, 1.98, 0.86, fill=T_INDIGO, line=TINT_LINE, align=PP_ALIGN.CENTER, margin=0.12,
        paras=[[(h, {"size": 12, "bold": True})]] + [[(line, {"size": 9, "color": SLATE})] for line in d])
    step_no(s, x - 0.1, 0.92, i + 1)
    if i < 3:
        arrow(s, x + 1.98, 1.45, x + 2.27, 1.45)
arrow(s, 7.37 + 0.99, 1.88, 7.37 + 0.99, 2.1)
box(s, 0.55, 2.1, 8.79, 0.42, fill=BLUE, paras="Stored answers: every method reads the same data", size=12,
    bold=True, color=WHITE, align=PP_ALIGN.CENTER)
lanes = [(0.55, "Objective 1: how unsure is the model?", INDIGO, T_INDIGO,
          [("17 methods", "score uncertainty", 5), ("Metrics", "AUROC, calibration", None),
           ("Baseline profile", "per model and dataset", None)]),
         (5.05, "Objective 2: what kind of doubt?", TEAL, T_TEAL,
          [("Re-ask labels", "evidence or hint", 6), ("Detector + classifier", "vs the routing rule", 7),
           ("Report + fix", "recommended action", None)])]
for x0, head, col, tint, steps2 in lanes:
    arrow(s, x0 + 0.65, 2.52, x0 + 0.65, 3.18)
    text(s, x0 + 0.85, 2.7, 3.5, 0.3, head, size=12, bold=True, color=col)
    for k, (h, d, n) in enumerate(steps2):
        x = x0 + k * 1.5
        box(s, x, 3.18, 1.3, 0.85, fill=tint, align=PP_ALIGN.CENTER, margin=0.05,
            paras=[[(h, {"size": 11, "bold": True})], [(d, {"size": 9, "color": SLATE})]])
        if n:
            step_no(s, x - 0.1, 3.08, n)
        if k < 2:
            arrow(s, x + 1.3, 3.6, x + 1.5, 3.6)
box(s, 0.55, 4.3, 8.79, 0.5, fill=TINT, line=TINT_LINE, margin=0.5,
    paras=[[("Run and report: ", {"bold": True}),
            ("one command runs every step, resumes after a crash, and writes the tables and figures.", {})]],
    size=11, color=NAVY)
step_no(s, 0.68, 4.415, 8)
notes(s, "This slide shows the Phase 1 plan as one picture; the numbers are the eight steps. One: we freeze 11 "
         "datasets, 3,367 questions. Two: we test 6 open models from two families, Qwen and Llama. Three: each question gets one "
         "normal answer, ten samples and a few self-checks. Four: rules or a judge from another family mark the "
         "answer right or wrong. Everything is stored once, so every method reads the same data. Five: Objective 1 "
         "scores uncertainty with 17 methods. Six and seven: Objective 2 re-asks unsure answers to get labels, then "
         "trains a detector and a type classifier and compares it with the current routing rule. Eight: one command "
         "runs it all and writes the tables and figures.")
ORDER.append(s)

# 9. Implementation and evaluation setup (shared by both objectives, so it comes before them)
s = new_slide("Implementation and Evaluation Setup", "Datasets, tools and metrics used in Phase 1")
tag(s, 0.55, 1.0, "DATASETS")
table(s, 0.55, 1.3, 4.3, [1.6, 2.1, 0.6], [["Doubt probed", "Datasets", "Qs"],
                                            ["Missing knowledge", "TriviaQA, PopQA, HaluEval", "900"],
                                            ["Common myths", "TruthfulQA", "817"],
                                            ["Reasoning", "GSM8K, MATH, StrategyQA, ARC", "950"],
                                            ["Unclear / unanswerable", "AmbigQA, SelfAware", "500"],
                                            ["Changing facts", "FreshQA", "200"]],
      row_h=0.3, size=9, aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT])
for i, (num, lab) in enumerate([("3,367", "questions"), ("17", "methods"), ("6", "Qwen, Llama models")]):
    box(s, 0.55 + i * 1.46, 3.45, 1.38, 0.75, fill=TINT, align=PP_ALIGN.CENTER,
        paras=[[(num, {"size": 18, "bold": True, "color": BLUE})], [(lab, {"size": 9, "color": SLATE})]])
tag(s, 5.1, 1.0, "BUILT WITH")
table(s, 5.1, 1.3, 4.3, [1.7, 2.6], [["Tool", "Used for"], ["Ollama", "runs the open LLMs locally"],
                                     ["DeBERTa-v3 NLI", "same-meaning check"],
                                     ["scikit-learn", "classifiers and metrics"],
                                     ["pandas, SciPy", "tables and statistics"]], row_h=0.3, size=9.5)
tag(s, 5.1, 3.0, "METRICS")
bullets(s, 5.1, 3.32, 4.3, ["AUROC: are wrong answers ranked as more unsure?",
                            "AURAC: accuracy when unsure answers are refused",
                            "ECE: is a stated 70% right 70% of the time?",
                            "Macro-F1: quality of the type classifier"], size=10.5, step=0.36, h=0.32)
notes(s, "We use eleven public datasets, chosen so every kind of doubt appears: missing knowledge, common myths, "
         "reasoning, unclear or unanswerable questions, and changing facts. In total 3,367 questions, 17 methods and "
         "6 models from the Qwen and Llama families. Ollama runs the open models locally, DeBERTa checks whether two answers mean the same, and "
         "scikit-learn, pandas and SciPy give the classifiers, metrics and statistics. The main metric is AUROC: the "
         "chance that a wrong answer gets a higher uncertainty than a right one.")
ORDER.append(s)

# ================================================================== Objective 1: method, then its results
# 10. Objective 1 method: semantic entropy
s = new_slide("Objective 1: Semantic Entropy", "The existing algorithm used to measure baseline uncertainty [1], [2]")
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

# 11. Objective 1 methods: all 17, grouped where they share an idea, with pros, cons and sources
FAMILY_FILL = {"meaning": T_TEAL, "token": T_INDIGO, "surface": T_AMBER, "self-eval": T_ROSE}
METHOD_ROWS = [  # family, method (how many of the 17), key idea (pro), limitation (con), references
    ("meaning", "Semantic entropy (3)", "Groups answers by meaning, not wording", "Needs 5–10 samples; misses sure errors",
     "[1], [2]"),
    ("meaning", "Number of meaning groups (1)", "Simple count of different meanings", "Ignores how big each group is",
     "[3]"),
    ("meaning", "Graph: EigV, Degree, Ecc. (3)", "Soft similarity graph, no hard grouping",
     "Same sampling cost; hard to explain", "[3]"),
    ("meaning", "Disagreement with answer (1)", "Share of samples that differ from it", "Relies on one main answer",
     "[11]"),
    ("meaning", "SelfCheckGPT-NLI (1)", "Checks the answer against every sample", "Many NLI checks; slow on long text",
     "[8]"),
    ("token", "Sequence NLL, normalised (2)", "Word probabilities; one answer is enough", "Mixes up wording with doubt",
     "[4]"),
    ("token", "Token entropy: mean, max (2)", "Finds the single most unsure word", "A rare but right word looks unsure",
     "[4]"),
    ("token", "Predictive entropy (1)", "Entropy over the sampled answers", "Different wordings count as doubt", "[2]"),
    ("surface", "Lexical similarity (1)", "Plain word overlap; no extra model", "Misses paraphrases", "[3]"),
    ("self-eval", "P(True) (1)", "The model rates its own answer", "Small models are over-confident", "[5]"),
    ("self-eval", "Verbalised confidence (1)", "Easy to read: “I am 80% sure”", "Badly calibrated; ranks errors poorly",
     "[6], [7]")]
s = new_slide("Objective 1: Methods Compared",
              "Semantic entropy and 16 other published methods: key idea, limitation and source")
rows = [["Method (how many of the 17)", "Key idea (pro)", "Limitation (con)", "Ref."]]
rows += [[name, pro, con, [[(ref, {"color": BLUE, "bold": True})]]] for _, name, pro, con, ref in METHOD_ROWS]
table(s, 0.5, 1.0, 8.95, [2.45, 3.0, 2.85, 0.65], rows, row_h=0.29, size=9, bold_first_col=True,
      fills={(r, 0): FAMILY_FILL[fam] for r, (fam, *_) in enumerate(METHOD_ROWS, start=1)})
legend = [("Meaning of the samples", T_TEAL), ("Word probabilities", T_INDIGO), ("Word overlap", T_AMBER),
          ("Self-evaluation", T_ROSE)]
for i, (lab, fill) in enumerate(legend):
    box(s, 0.5 + i * 1.75, 4.62, 0.22, 0.16, fill=fill, line=RULE, radius=0.1)
    text(s, 0.78 + i * 1.75, 4.58, 1.45, 0.22, lab, size=8.5, color=SLATE)
text(s, 7.5, 4.58, 1.95, 0.22, "[n] = References slide", size=8.5, italic=True, color=SLATE, align=PP_ALIGN.RIGHT)
notes(s, "We do not test semantic entropy alone. This table lists all seventeen methods; where several share one idea "
         "they are on one row, with the count in brackets. For each: the key idea, which is its strength, the main "
         "limitation, and the paper it comes from. The colours show the four families: how the meanings of the samples "
         "spread, the model's own word probabilities, simple word overlap, and asking the model to judge itself. All "
         "seventeen are computed from the same stored answers.")
ORDER.append(s)

# 12-14. Objective 1 results: one table per group of datasets, one row per shown model, and a small chart
REAL_NOTE = {"Facts and myths": " Real: qwen2.5:3b and llama2:7b on TruthfulQA.",
             "Reasoning": " Real: qwen2.5:3b and llama2:7b on GSM8K."}
GROUP_NOTES = {
    "Facts and myths": "Each row is one model; each cell lists the values in the order of the datasets column. Fact "
                       "questions show a useful AUROC, well above 0.5. TruthfulQA, the myth questions, stays close to "
                       "0.5 for every model: entropy cannot separate right from wrong there.",
    "Reasoning": "Maths and reasoning give the clearest signal: when the model is wrong, its ten answers scatter, so "
                 "AUROC is high. Accuracy grows with model size inside each family.",
    "Unclear and changing questions": "These sets hold ambiguous, unanswerable and time-dependent questions. "
                                      "Accuracy is lowest here and entropy is weaker: models often give the same "
                                      "wrong answer every time, so sampling alone misses it. This is why Objective 2 "
                                      "classifies the type."}
HIGH, LOW = "15803D", "B91C1C"  # highest / lowest value of a dataset across the models
FORMATS = (lambda x: f"{x:.0%}", lambda x: f"{x:.2f}")  # accuracy, AUROC: as shown, so ties are what the eye sees


def marked_cell(texts, highs, lows):
    """One table cell 'a, b, c, d' as runs: each dataset's highest value green, its lowest red (ties all marked)."""
    runs = []
    for j, t in enumerate(texts):
        if j:
            runs.append((", ", {}))
        style = {"bold": True, "color": HIGH} if t == highs[j] else {"bold": True, "color": LOW} if t == lows[j] else {}
        runs.append((t, style))
    return [runs]


def auroc_panel(slide, x, y, w, n_models):
    """What AUROC stands for and means, and what the table's colours mean."""
    box(slide, x, y, w, 2.42, fill=TINT, line=TINT_LINE, anchor=MSO_ANCHOR.TOP, margin=0.14, space_after=4, paras=[
        [("What is AUROC?", {"size": 12, "bold": True, "color": BLUE})],
        [("Area Under the Receiver Operating Characteristic curve.", {"italic": True, "color": SLATE})],
        "Here: the chance that a wrong answer gets a higher semantic entropy (looks less sure) than a right answer.",
        [("0.5 = guessing  ·  1.0 = perfect", {"bold": True, "color": NAVY})],
        "Example: 0.80 → in 8 of 10 pairs of one wrong and one right answer, the wrong one looks less sure."],
        size=9.5)
    box(slide, x, y + 2.54, w, 0.97, fill=WHITE, line=RULE, anchor=MSO_ANCHOR.TOP, margin=0.14, space_after=2, paras=[
        [("Colours (per dataset)", {"size": 10, "bold": True})],
        [("Green", {"bold": True, "color": HIGH}), (f" = highest of the {n_models} models", {})],
        [("Red", {"bold": True, "color": LOW}), (f" = lowest of the {n_models} models", {})],
        [("Ties are all marked.", {"size": 8.5, "italic": True, "color": SLATE})]], size=9.5)


for k, (group, sets) in enumerate(GROUPS, start=1):
    s = new_slide(f"Objective 1 Result: Baseline Uncertainty ({k}/3)",
                  f"{group}: accuracy and semantic-entropy AUROC for six models")
    names = ", ".join(sets)
    shown = {m: [[FORMATS[i](EXP[m, d][i]) for d in sets] for i in (0, 1)] for m in SHOWN}
    rows = [["Model", "Datasets", "Accuracy", "AUROC"]]
    extremes = []
    for i in (0, 1):  # per metric and dataset: the highest and lowest shown value
        vals = [[EXP[m, d][i] for m in SHOWN] for d in sets]
        extremes.append(([FORMATS[i](max(v)) for v in vals], [FORMATS[i](min(v)) for v in vals]))
    for model in SHOWN:
        rows.append([model, names] + [marked_cell(shown[model][i], *extremes[i]) for i in (0, 1)])
    table(s, 0.6, 1.0, 6.0, [1.05, 2.3, 1.32, 1.33], rows, row_h=0.29, size=8.5, bold_first_col=True)
    # Small chart: per dataset, the six models' mean accuracy next to their mean AUROC
    mean = lambda d, i: round(sum(EXP[m, d][i] for m in SHOWN) / len(SHOWN), 2)
    column_chart(s, 0.5, 3.12, 6.2, 1.5, sets, [("Accuracy", tuple(mean(d, 0) for d in sets)),
                                                ("AUROC", tuple(mean(d, 1) for d in sets))],
                 [AMBER, INDIGO], "Mean of the six models", unit=0.5, labels=True, cat_size=8.5, label_size=8,
                 title_size=8.5, legend_size=8)
    auroc_panel(s, 6.8, 1.0, 2.6, len(SHOWN))
    basis_note(s, 4.7, "Cells follow the Datasets order." + REAL_NOTE.get(group, ""))
    notes(s, GROUP_NOTES[group] + " For each dataset, green marks the highest value of the six models and red the "
             "lowest. AUROC, the area under the receiver operating characteristic curve, is the chance that a wrong "
             "answer gets a higher semantic entropy than a right one: 0.5 is guessing, 1 is perfect. The small chart "
             "puts the six models' mean accuracy next to their mean AUROC for each dataset: a model can be often wrong "
             "and still show when it is wrong."
          + (" These are expected values from published results, not yet measured in this study." if EXPECTED else ""))
    ORDER.append(s)

# 15. Objective 1 result: every method for every shown model (AUROC, mean over the 11 datasets)
# Expected values: semantic entropy (discrete) is each model's mean AUROC from slides 12-14; every other method is
# shifted from it by the gap published for that method: discrete and weighted SE about equal, plain/token entropy
# about 0.10 lower and P(True) about 0.09 lower [1] (0.790 vs 0.691 and 0.698); graph methods level with or
# slightly above SE, number of groups and lexical overlap below it [3]; verbalised confidence near 0.6 [7];
# fewer samples (the pipeline's K = 4) lower AUROC [1], [2].
METHOD_GAP = [  # family, label, gap to semantic entropy (discrete)
    ("meaning", "Semantic entropy (discrete)", 0.0), ("meaning", "Semantic entropy (weighted)", 0.0),
    ("meaning", "Semantic entropy (K = 4, pipeline)", -0.03), ("meaning", "Number of meaning groups", -0.02),
    ("meaning", "Graph: EigV", 0.01), ("meaning", "Graph: Degree", 0.01), ("meaning", "Graph: Eccentricity", 0.0),
    ("meaning", "Disagreement with answer", -0.02), ("meaning", "SelfCheckGPT-NLI", -0.02),
    ("token", "Sequence NLL", -0.10), ("token", "Length-normalised NLL", -0.10),
    ("token", "Mean token entropy", -0.10), ("token", "Max token entropy", -0.10),
    ("token", "Predictive entropy", -0.10), ("surface", "Lexical similarity (ROUGE-L)", -0.05),
    ("self-eval", "P(True)", -0.09), ("self-eval", "Verbalised confidence", -0.17)]
assert len(METHOD_GAP) == 17
base = {m: sum(EXP[m, d][1] for d in SETS) / len(SETS) for m in SHOWN}
cell = {(m, lab): f"{min(.95, max(.50, base[m] + gap)):.2f}" for m in SHOWN for _, lab, gap in METHOD_GAP}
s = new_slide("Objective 1 Result: Methods Across Models",
              "AUROC of all 17 methods for six models (mean over the 11 datasets)")
rows = [["Method"] + SHOWN]
for fam, lab, _ in METHOD_GAP:
    row = [lab]
    for m in SHOWN:
        col = [cell[m, l] for _, l, _ in METHOD_GAP]
        v = cell[m, lab]
        style = {"bold": True, "color": HIGH} if v == max(col) else {"bold": True, "color": LOW} if v == min(col) else {}
        row.append([[(v, style)]])
    rows.append(row)
table(s, 0.5, 1.0, 7.48, [2.2] + [0.88] * len(SHOWN), rows, row_h=0.2, size=8, bold_first_col=True, pad=0.01,
      fills={(r, 0): FAMILY_FILL[fam] for r, (fam, *_) in enumerate(METHOD_GAP, start=1)},
      aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * len(SHOWN))
text(s, 8.1, 1.0, 1.35, 0.25, "Families", size=9.5, bold=True)
for i, (lab, fill) in enumerate(legend):  # the same tints as the method names in the table
    box(s, 8.1, 1.28 + i * 0.32, 1.35, 0.27, fill=fill, paras=lab, size=8, margin=0.06, radius=0.1)
box(s, 8.1, 2.7, 1.35, 1.3, fill=WHITE, line=RULE, anchor=MSO_ANCHOR.TOP, margin=0.1, space_after=3,
    paras=[[("Colours", {"size": 9.5, "bold": True})],
           [("Green", {"bold": True, "color": HIGH}), (" = best", {})],
           [("Red", {"bold": True, "color": LOW}), (" = weakest", {})],
           [("method, for each model", {"italic": True, "color": SLATE})]], size=8.5)
basis_note(s, 4.7, "Semantic entropy = mean of slides 12–14; other rows apply the gaps published in [1], [3], [7].")
notes(s, "This table puts every method next to every model. Each value is the AUROC of that method for that model, "
         "averaged over the eleven datasets. Green marks the best method for each model and red the weakest. Methods "
         "that look at the meaning of the samples, semantic entropy and the graph methods, are expected on top; the "
         "model's own word probabilities are about 0.1 lower, and asking the model how sure it is comes last."
         + (" These are expected values: the semantic-entropy row is the mean of the previous three slides, and the "
            "other rows apply the gaps between methods reported in the published comparisons. They are not yet "
            "measured in this study." if EXPECTED else ""))
ORDER.append(s)

# ================================================================== Objective 2: method, then its result
# 16. Objective 2 method: classifying uncertainty (types of doubt and how each answer is labelled)
s = new_slide("Objective 2: Classifying Uncertainty",
              "How unsure, what kind of doubt, which fix: the Australia example from Objective 1, continued")
# Left: the worked example, step by step, ending in the framework's output (illustrative numbers)
tag(s, 0.5, 1.0, "WORKED EXAMPLE (ILLUSTRATIVE NUMBERS)", w=3.25)
steps = [("INPUT", T_GREY, SLATE, 0.58, "What is the capital of Australia?",
          ["answer: Canberra · samples: 7 Canberra, 3 Sydney · H = 0.61"]),
         ("IDENTIFY", T_INDIGO, INDIGO, 0.58, "How unsure?  →  medium",
          ["P(hallucination) = 0.42, from SE_norm 0.27, P(True) 0.80, …"]),
         ("CLASSIFY", T_TEAL, "0F766E", 0.8, "What kind of doubt?  →  knowledge",
          ["re-ask with the evidence: 5 of 5 say Canberra → fixed ✓",
           "re-ask with a method hint: 3 of 5 agree → not fixed ✗"]),
         ("ACT", T_AMBER, "92400E", 0.58, "Which fix?  →  look up evidence",
          ["Phase 2, Layer 2: retrieve the fact, then check the answer"])]
y = 1.3
for i, (chip, fill, dark, h, head, details) in enumerate(steps):
    box(s, 0.5, y, 4.55, h, fill=fill, margin=0.1)
    box(s, 0.6, y + (h - 0.28) / 2, 0.85, 0.28, fill=WHITE, paras=chip, size=8, bold=True, color=dark, radius=0.2,
        align=PP_ALIGN.CENTER, margin=0.03)
    text(s, 1.55, y + 0.06, 3.45, 0.24, head, size=10.5, bold=True)
    for k, d in enumerate(details):
        text(s, 1.55, y + 0.31 + k * 0.22, 3.45, 0.22, d, size=8.5, color=SLATE)
    if i < len(steps) - 1:
        arrow(s, 2.9, y + h, 2.9, y + h + 0.1)
    y += h + 0.1
text(s, 0.5, 4.22, 4.55, 0.72, [[("How types are labelled: ", {"bold": True, "color": INK}),
                                 ("fixed = the majority answer is right and at least 4 of 5 agree. AmbigQA and "
                                  "SelfAware give aleatoric, FreshQA temporal; ~300 answers are labelled by hand "
                                  "(Cohen's κ). The classifier must beat the current routing rule.", {})]],
     size=8.5, color=SLATE)

# Right, top: the three levels as a gauge (half doughnut), needle at the example's 0.42
LOW_T, HIGH_T, P_EX = 0.2, 0.5, 0.42  # example thresholds; each model's are tuned on its dev questions
tag(s, 5.25, 1.0, "HOW UNSURE: THE THREE LEVELS", w=2.35)
cx, cy, R, band = 7.35, 2.72, 1.1, 0.36


def gauge_xy(v, r):
    """Point on the gauge for value v in [0, 1] (left end 0, right end 1) at radius r."""
    a = math.radians(180 + 180 * v)
    return cx + r * math.cos(a), cy + r * math.sin(a)


for lo, hi, colour, label in [(0, LOW_T, "16A34A", "LOW"), (LOW_T, HIGH_T, AMBER, "MEDIUM"),
                              (HIGH_T, 1, "DC2626", "HIGH")]:
    arc = s.shapes.add_shape(MSO_SHAPE.BLOCK_ARC, E(cx - R), E(cy - R), E(2 * R), E(2 * R))
    _plain(arc)
    arc.adjustments[0] = (180 + 180 * lo) * 0.6  # start angle, clockwise from 3 o'clock (60000ths of a degree)
    arc.adjustments[1] = ((180 + 180 * hi) % 360) * 0.6
    arc.adjustments[2] = band / (2 * R)  # thickness as a share of the size
    arc.fill.solid()
    arc.fill.fore_color.rgb = rgb(colour)
    arc.line.color.rgb = rgb(WHITE)
    arc.line.width = Pt(1.5)
    lx, ly = gauge_xy((lo + hi) / 2, R - band / 2)
    text(s, lx - 0.45, ly - 0.1, 0.9, 0.2, label, size=7.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
for v in (0, LOW_T, HIGH_T, 1):
    tx, ty = gauge_xy(v, R + 0.14)
    text(s, tx - 0.25, ty - (0.0 if v in (0, 1) else 0.12), 0.5, 0.2, f"{v:g}", size=8.5, color=SLATE,
         align=PP_ALIGN.CENTER)
nx, ny = gauge_xy(P_EX, R - band - 0.05)
arrow(s, cx, cy, nx, ny, color=NAVY, width=2.25)
dot(s, cx - 0.07, cy - 0.07, 0.14, NAVY)
text(s, 5.25, cy + 0.13, 4.2, 0.24, [[("P(hallucination) = 0.42  →  ", {"bold": True}),
                                      ("medium", {"bold": True, "color": AMBER})]], size=10, align=PP_ALIGN.CENTER)
text(s, 5.25, cy + 0.4, 4.2, 0.4, "low: answers below it are ≥ 90% right · high: answers above it are ≥ 50% wrong. "
     "Example thresholds 0.2 and 0.5; each model's are tuned on its dev questions.", size=8, italic=True,
     color=SLATE, align=PP_ALIGN.CENTER)
# Right, bottom: the four types and what fixes each
rows = [["Type", "What fixes it"], ["Knowledge", "Look up evidence"], ["Reasoning", "Step-by-step reasoning"],
        ["Aleatoric", "Clarify, refuse or fix the premise"], ["Temporal", "Search for fresh information"]]
table(s, 5.25, 3.62, 4.2, [1.2, 3.0], rows, row_h=0.26, size=9, bold_first_col=True,
      fills={(1, 0): T_TEAL, (2, 0): T_INDIGO, (3, 0): T_AMBER, (4, 0): T_ROSE})
notes(s, "Objective 2 asks two things about an answer: how likely it is wrong, and what kind of doubt it has. We continue "
         "the Australia example from Objective 1; the numbers are illustrative. Ten samples gave seven Canberra and three "
         "Sydney, so semantic entropy is 0.61. Identify: the detector combines the signals, here semantic entropy, "
         "P(True) and the others, into one probability of hallucination, 0.42. The gauge shows the three levels. Low "
         "is the range where answers are at least 90 percent right, high where at least half are wrong, and medium is "
         "in between. The thresholds are tuned for each model on its dev questions; 0.2 and 0.5 are examples, so 0.42 "
         "is medium. Classify: we re-ask five times with the true evidence and five times with a method hint. The "
         "evidence fixes it, five of five say Canberra; the hint does not. So the doubt is knowledge. Act: the fix for "
         "knowledge is to look up evidence, which Phase 2 does in Layer 2. The table lists the four types and their "
         "fixes. Unclear and time-dependent questions are labelled by their datasets, about 300 answers are checked by "
         "hand, and the classifier must beat the current routing rule.")
ORDER.append(s)

# 17. Objective 2 result: what the framework reports, what is checked, what the full run measures.
# No numbers here: Objective 2 has no published baseline to borrow from, so its results wait for the full run.
s = new_slide("Objective 2 Result: Framework Built and Checked",
              "What the framework reports for any answer, what is checked so far, and what the full run measures")
tag(s, 0.55, 1.0, "WHAT IT REPORTS", w=1.6)
flow = [("INPUT", "Question and answer", "the answer, its 10 samples and the self-checks", T_GREY, SLATE),
        ("IDENTIFY", "How unsure?", "hallucination probability → low, medium or high", T_INDIGO, INDIGO),
        ("CLASSIFY", "What kind of doubt?", "knowledge, reasoning, aleatoric or temporal", T_TEAL, "0F766E"),
        ("ACT", "Which fix?", "evidence, step-by-step reasoning, clarify, or fresh search", T_AMBER, "92400E")]
for i, (step, head, detail, fill, dark) in enumerate(flow):
    y = 1.35 + i * 0.83
    box(s, 0.55, y, 5.0, 0.64, fill=fill, margin=0.12)
    box(s, 0.68, y + 0.17, 1.05, 0.3, fill=WHITE, paras=step, size=8.5, bold=True, color=dark, radius=0.2,
        align=PP_ALIGN.CENTER, margin=0.04)
    text(s, 1.9, y + 0.07, 3.55, 0.25, head, size=11.5, bold=True)
    text(s, 1.9, y + 0.33, 3.55, 0.25, detail, size=9.5, color=SLATE)
    if i < 3:
        arrow(s, 3.05, y + 0.64, 3.05, y + 0.83)
box(s, 5.85, 1.0, 3.6, 1.72, fill=T_TEAL, anchor=MSO_ANCHOR.TOP, margin=0.16, space_after=4,
    paras=[[("CHECKED SO FAR", {"size": 9, "bold": True, "color": "0F766E"})],
           [("✓ ", {"bold": True, "color": "0F766E"}), ("Re-ask labels name the planted type in over 80% of answers", {})],
           [("✓ ", {"bold": True, "color": "0F766E"}), ("The type classifier beats the current routing rule", {})],
           [("✓ ", {"bold": True, "color": "0F766E"}), ("The analyze tool reports level, type and fix", {})],
           [("With a simulated model: it checks the code, not real models.", {"size": 8.5, "italic": True,
                                                                            "color": SLATE})]], size=10)
box(s, 5.85, 2.85, 3.6, 1.65, fill="F8FAFC", line=RULE, anchor=MSO_ANCHOR.TOP, margin=0.16, space_after=4,
    paras=[[("MEASURED IN THE FULL RUN", {"size": 9, "bold": True, "color": INDIGO})],
           "Detector: AUROC and calibration (ECE)",
           "Classifier: macro-F1 against the routing rule",
           "Confusion matrix of the four types",
           "Agreement with ~300 human labels (κ)"], size=10)
notes(s, "This is where Objective 2 stands. On the left is what the framework reports for any question: from the answer "
         "and its samples it gives how likely the answer is a hallucination, low, medium or high, then the kind of doubt, "
         "and the fix that suits it. On the right, what is already checked: with a simulated model whose types we "
         "planted, the re-ask labels find the planted type in over 80 percent of answers, the type classifier beats the "
         "current routing rule, and the analyze tool works end to end. That proves the code, not real models. The real "
         "numbers come from the full run: the detector's AUROC and calibration, the classifier's macro-F1 against the "
         "routing rule, its confusion matrix, and agreement with about 300 human labels.")
ORDER.append(s)

# ================================================================== Wrap-up
# 18. Plan, execution and result
s = new_slide("Phase 1: Plan, Execution and Result", "What we planned, what we built, and what we measured")
DONE, PEND = ("Done", {}), ("Pending", {})
rows = [["Part", "Planned", "Built and checked", "Status"],
        ["Question sets", "11 datasets, fixed versions", "3,367 questions; rebuild gives identical files", "Done"],
        ["Answer generation", "1 answer + 10 samples + self-checks", "Runner that resumes after a crash", "Done"],
        ["Right / wrong labels", "Rules + independent judge", "Second judge to measure agreement", "Done"],
        ["Objective 1", "17 methods, metrics, statistics", "Reports, tables and figures", "Done"],
        ["Objective 2", "Detector + type classifier", "Re-ask labels, 3 classifiers, analyze tool", "Done"],
        ["Full study", "6 models: Qwen and Llama", "One command; resumes after a crash", "Pending"]]
fills = {(r, 3): (T_TEAL if rows[r][3] == "Done" else T_AMBER) for r in range(1, len(rows))}
table(s, 0.5, 1.02, 8.95, [1.75, 2.75, 3.4, 1.05], rows, row_h=0.4, size=10.5, bold_first_col=True, fills=fills)
box(s, 0.72, 4.0, 8.5, 0.85, fill=TINT, line=TINT_LINE, margin=0.2,
    paras=[[("Result in short: ", {"bold": True}), ("all Phase 1 code is built and ready to run. Early real data from two "
            "models shows that semantic entropy works on maths (AUROC 0.86) but not on myth questions (0.55).", {})]], size=11.5, color=NAVY)
notes(s, "This table compares plan and execution. Every part of Phase 1 is built; only the full run on "
         "the Mac is pending. Even before that run, stored results from the pipeline give us a first real finding, "
         "shown with the Objective 1 results.")
ORDER.append(s)

# 19. Discussion and limitations: short points in two columns
s = new_slide("Discussion and Limitations", "What the results mean, and what to keep in mind")
cols = [(0.55, "WHAT WE SEE", TEAL, T_TEAL, "0F766E",
         ["Entropy spots maths errors well.", "On myth questions it is close to guessing.",
          "Some errors are made while sure: the blind spot."]),
        (5.0, "LIMITATIONS", AMBER, T_AMBER, "92400E",
         ["Most result numbers are expected values, not yet measured.", "A judge model marks free-text answers; humans check it.",
          "Only 6 small open models (3B–8B), two families."])]
for x, head, col, tint, dark, items in cols:
    box(s, x, 1.05, 4.3, 3.75, fill="F8FAFC", line=RULE)
    box(s, x + 0.25, 1.25, 1.7, 0.32, fill=tint, paras=head, size=10, bold=True, color=dark, radius=0.2,
        align=PP_ALIGN.CENTER)
    for k, it in enumerate(items):
        y = 1.85 + k * 0.95
        dot(s, x + 0.3, y + 0.03, 0.34, col, str(k + 1), size=11)
        text(s, x + 0.8, y, 3.3, 0.8, it, size=14)
notes(s, "What we see: entropy works well on maths, is close to guessing on myth questions, and misses errors the "
         "model makes while sure. Limitations: most result numbers are expected values until the full run, a judge model "
         "marks free-text answers so humans check the judge, and we only test six small open models from two families, Qwen and Llama.")
ORDER.append(s)

# 20. Phase 2 plan
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

# 21. References
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
