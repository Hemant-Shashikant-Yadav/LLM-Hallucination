"""
Shared answer-parsing helpers.

Used by Layer 1 (clustering answers, task routing), Layer 3 (comparing and
voting on answers, exact calculation) and the evaluation harness (GSM8K scoring).
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from fractions import Fraction

# Must start on '-', '$' or a digit (never on whitespace), so the character before
# a match is the real neighbour of the number.
_NUMBER = r"-?(?:\$\s?)?\d[\d,]*(?:\.\d+)?"
_END = r"(?![\d/]|\.\d)"  # a sentence-final '.' may follow; a decimal or '/' may not

# A numeric answer value: LaTeX fraction, mixed number, a/b fraction, or a plain
# number with an optional scale word ("1.5 million").
_VALUE = re.compile(
    r"(?P<frac>\\d?frac\{\s*(?P<fn>-?\d+(?:\.\d+)?)\s*\}\{\s*(?P<fd>\d+(?:\.\d+)?)\s*\})"
    r"|(?P<mixed>(?<![\d.,/])(?P<mw>-?\d+)\s+(?P<mn>\d+)/(?P<md>\d+)" + _END + ")"
    r"|(?P<ratio>(?<![\d.,/])(?P<rn>-?\d+)\s*/\s*(?P<rd>\d+)" + _END + ")"
    r"|(?P<num>" + _NUMBER + r")(?:\s*(?P<scale>thousand|million|billion|trillion)\b)?",
    re.IGNORECASE,
)
_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}

# Explicit answer markers, strongest first. The answer value is parsed from the
# text right after the last occurrence of the strongest marker present.
_MARKERS = [
    re.compile(r"####"),
    re.compile(r"\\boxed\{"),  # LaTeX answer box used by Qwen-style models
    re.compile(r"final answer\s*(?:is|:)?\s*[:=]?", re.IGNORECASE),
    re.compile(r"\banswer\s*(?:is|:)", re.IGNORECASE),
]


def normalize_number(raw: str) -> str | None:
    """Normalize '$1,234.50' → '1234.5', '18.0' → '18'. Returns None if not numeric."""
    cleaned = raw.replace(",", "").replace("$", "").replace(" ", "").rstrip(".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return _format_value(value)


def _format_value(value: float) -> str | None:
    if math.isnan(value) or math.isinf(value):
        return None
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _is_symbolic(text: str, start: int, end: int) -> bool:
    """'2x', 'x^2', '3y' are algebra, not numeric answers."""
    before = text[start - 1] if start > 0 else " "
    after = text[end:end + 2]
    if before.isalpha() or before in "^_":
        return True
    # A single letter glued to the number ('2x') is a variable; units like 'km' are not
    return bool(re.match(r"[a-zA-Z](?![a-zA-Z])", after)) and not re.match(r"[kKmMbB]\b", after)


def _match_value(match: re.Match, text: str) -> str | None:
    if match.group("frac"):
        den = float(match.group("fd"))
        return _format_value(float(match.group("fn")) / den) if den else None
    if match.group("mixed"):
        whole, num, den = int(match.group("mw")), int(match.group("mn")), int(match.group("md"))
        if not den or num >= den:
            return None
        sign = -1 if whole < 0 else 1
        return _format_value(whole + sign * num / den)
    if match.group("ratio"):
        den = int(match.group("rd"))
        return _format_value(int(match.group("rn")) / den) if den else None
    if _is_symbolic(text, match.start("num"), match.end("num")):
        return None
    value = normalize_number(match.group("num"))
    if value is None or not match.group("scale"):
        return value
    return _format_value(float(value) * _SCALES[match.group("scale").lower()])


def _values(text: str) -> list[str]:
    out = []
    for m in _VALUE.finditer(text):
        v = _match_value(m, text)
        if v is not None:
            out.append(v)
    return out


def extract_final_number(text: str) -> str | None:
    """
    Extract the final numeric answer from a free-form response.

    Prefers explicit markers ("#### 18", "\\boxed{18}", "Final answer: 18",
    "the answer is 18"); otherwise falls back to the last number in the text.
    Understands fractions (3/4, \\frac{3}{4}, 2 1/2), scale words (1.5 million)
    and ignores algebra ('2x', 'x^2'). A marker followed by a symbolic answer
    returns None rather than some unrelated number from earlier in the text.
    """
    if not text:
        return None
    for marker in _MARKERS:
        hits = list(marker.finditer(text))
        if not hits:
            continue
        tail = text[hits[-1].end():hits[-1].end() + 80].lstrip(" *:=$\\")
        first = _VALUE.search(tail)
        if first and first.start() <= 3:
            if re.match(r"[\d.,\s]*[*/^+\-]\s*\(?[a-zA-Z]\b", tail[first.start():]):
                return None  # '2 * x', '3 + y': an expression, not a number
            return _match_value(first, tail)
        if re.match(r"\s*[a-zA-Z\\(]", tail) and re.search(r"[a-z]\s*[\^=+*/]|\d\s*[a-z]\b", tail[:30]):
            return None  # symbolic final answer such as '2x' or 'x = 2y + 1'
    numbers = _values(text)
    return numbers[-1] if numbers else None


_MARKERS_PRESENT = re.compile(r"####|\\boxed\{|final answer|\banswer\s*(?:is|:)", re.IGNORECASE)

_SYMBOLIC_MARKER = re.compile(
    r"(?:final answer\s*(?:is|:)?|\\boxed\{)\s*[:=]?\s*\**\s*([^\n]{1,60})", re.IGNORECASE
)


def extract_final_answer(text: str) -> str | None:
    """
    Final answer as a comparable key: the number when there is one, otherwise a
    normalised symbolic string ('2x', 'x=4'), so non-numeric math answers can
    still be clustered and voted on.
    """
    stated = final_expression(text)
    if stated and re.search(r"[a-zA-Z\\πe√]", stated):
        # π/4, e^2, 2x sin x, 18 \text{dollars}: parse the stated expression rather
        # than grabbing a stray digit from it ('4' out of \frac{\pi}{4})
        key = symbolic_key(text)
        if key is not None:
            return key
    elif stated and not _MARKERS_PRESENT.search(text):
        # No explicit marker: the concluding sentence's number beats the last number
        # in the text ('… is ₹946.94, rounded to 2 decimal places' → 946.94, not 2)
        value = normalize_number(stated)
        if value is not None:
            return value
    number = extract_final_number(text)
    if number is not None:
        return number
    hits = _SYMBOLIC_MARKER.findall(text or "")
    if not hits:
        return None
    key = re.sub(r"\\(?:cdot|times)", "", hits[-1])
    key = re.sub(r"[\s*$}{\\]", "", key).rstrip(".").lower()
    return key or None


# =============================================================================
# Multiple-choice questions
# =============================================================================

# "(a) 120", "(B) 60", or an "a) …" / "A. …" option at the start of a line
_OPTION = re.compile(r"(?:(?<=\s)|^)\(([a-eA-E])\)\s*|(?m:^)[ \t]*([a-eA-E])[.)][ \t]+")
_CHOICE_MARK = re.compile(
    r"(?:answer|option|choice)\s*(?:is|:|=)?\s*(?:option\s*)?"
    r"(?:\(([a-eA-E])\)|([A-E])(?![A-Za-z])|([a-e])\))",
    re.IGNORECASE,
)
_BOXED_LETTER = re.compile(r"\\boxed\{\s*(?:\\text\{)?\(?([a-eA-E])\)?\}?\s*\}")


def parse_options(question: str) -> dict[str, str]:
    """
    Options of a multiple-choice question: {'a': '120', 'b': '60', ...}.
    Only a consecutive run a, b, c, ... counts; fewer than two options → {}.
    """
    run: list[tuple[int, int, str]] = []
    expected = "a"
    for m in _OPTION.finditer(question or ""):
        letter = (m.group(1) or m.group(2)).lower()
        if letter == expected:
            run.append((m.start(), m.end(), letter))
            expected = chr(ord(expected) + 1)
    if len(run) < 2:
        return {}
    options = {}
    for i, (_, end, letter) in enumerate(run):
        stop = run[i + 1][0] if i + 1 < len(run) else len(question)
        options[letter] = question[end:stop].strip().rstrip(",;")
    return options


def strip_options(question: str) -> str:
    """The question stem without its answer options."""
    options = parse_options(question)
    if not options:
        return question
    first = next(m for m in _OPTION.finditer(question)
                 if (m.group(1) or m.group(2)).lower() == "a")
    return question[:first.start()].strip()


def extract_choice(answer: str, options: dict[str, str]) -> str | None:
    """
    Which option an answer picked. Small models often compute the right value but
    mis-copy the letter, so a final number that matches exactly one option wins;
    otherwise an explicit letter ("answer is (b)", "\\boxed{B}"); otherwise the
    option text quoted near the end.
    """
    if not answer or not options:
        return None
    value = final_value(answer)
    if value is not None:
        hits = [k for k, text in options.items() if values_close(option_value(text), value)]
        if len(hits) == 1:
            return hits[0]
    marks = _BOXED_LETTER.findall(answer) or [
        next(g for g in m.groups() if g) for m in _CHOICE_MARK.finditer(answer)
    ]
    letters = [m.lower() for m in marks if m.lower() in options]
    if letters:
        return letters[-1]
    tail = answer[-300:].lower()
    found = [(tail.rfind(text.lower()), k) for k, text in options.items()
             if len(text) >= 3 and text.lower() in tail]
    return max(found)[1] if found else None


# =============================================================================
# Safe symbolic maths: LaTeX / plain answer text → sympy, without eval
# =============================================================================

_SYMPY_FUNCS = {
    "sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "log", "ln", "exp", "sqrt", "abs", "Abs",
}
_SYMPY_NAMES = {"x", "y", "z", "t", "n", "k", "a", "b", "c", "r", "pi", "e", "E", "oo"}
_UNIT_WORDS = (
    r"km/h|kmph|m/s|cm\^?2|cm²|cm³|m²|m\^2|sq\.?\s*units?|square\s+units?|units?|cm|mm|km|kg|"
    r"g|m|s|sec|seconds?|minutes?|mins?|hours?|hrs?|h|days?|years?|litres?|liters?|l|ml|"
    r"rupees|rs\.?|inr|dollars?|percent|per\s*cent|degrees?|radians?|ways|people|eggs|balls"
)


def _replace_braced(text: str, command: str, render) -> str:
    """Replace \\command{a}{b}... (balanced braces) using render(args)."""
    nargs = render.__code__.co_argcount
    while True:
        start = text.find(command + "{")
        if start < 0:
            return text
        pos, args = start + len(command), []
        for _ in range(nargs):
            if pos >= len(text) or text[pos] != "{":
                return text
            depth, end = 0, pos
            while end < len(text):
                depth += {"{": 1, "}": -1}.get(text[end], 0)
                if depth == 0:
                    break
                end += 1
            args.append(text[pos + 1:end])
            pos = end + 1
        text = text[:start] + render(*args) + text[pos:]


def _latex_to_plain(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\^\s*\{?\s*\\circ\s*\}?|\\circ|\\degree", " ", t)  # 70.53^\circ → 70.53
    t = re.sub(r"\\(?:text|mathrm|operatorname|textbf|mathbf)\{([^{}]*)\}", r" \1 ", t)
    t = re.sub(r"\\(?:left|right|displaystyle|,|;|!|quad)", " ", t)
    t = t.replace("\\(", " ").replace("\\)", " ").replace("\\[", " ").replace("\\]", " ").replace("$", " ")
    for cmd in ("\\dfrac", "\\tfrac", "\\frac"):
        t = _replace_braced(t, cmd, lambda a, b: f"(({a})/({b}))")
    t = _replace_braced(t, "\\sqrt", lambda a: f"sqrt({a})")
    t = re.sub(r"\\(sin|cos|tan|sec|csc|cot|arcsin|arccos|arctan|sinh|cosh|tanh|ln|log|exp)\b",
               r" \1 ", t)
    t = t.replace("arcsin", "asin").replace("arccos", "acos").replace("arctan", "atan")
    t = re.sub(r"\\(?:cdot|times)", "*", t)
    t = re.sub(r"\\pi\b", " pi ", t)
    t = re.sub(r"\\infty\b", " oo ", t)
    t = t.replace("π", " pi ").replace("∞", " oo ").replace("×", "*").replace("·", "*")
    t = t.replace("÷", "/").replace("−", "-").replace("√", "sqrt").replace("²", "^2").replace("³", "^3")
    t = t.replace("{", "(").replace("}", ")").replace("^", "**").replace("\\", " ")
    return t


def _strip_units(text: str) -> tuple[str, bool]:
    """Drop leading currency and trailing unit words; report whether a % was present."""
    t = text.strip().rstrip(".")
    percent = "%" in t or re.search(r"\bper\s*cent\b|\bpercent\b", t, re.IGNORECASE) is not None
    t = t.replace("%", " ")
    t = re.sub(r"^\s*(?:₹|rs\.?|inr|\$|usd)\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[₹°]", " ", t)
    for _ in range(3):
        t = re.sub(r"\s*\b(?:" + _UNIT_WORDS + r")\s*$", "", t, flags=re.IGNORECASE).strip()
    return t.replace(",", ""), percent


def _insert_multiplication(t: str) -> str:
    funcs = "|".join(sorted(_SYMPY_FUNCS, key=len, reverse=True))
    t = re.sub(rf"\b({funcs})\s+(?!\()([a-zA-Z0-9.]+(?:\*\*[a-zA-Z0-9.]+)?)", r"\1(\2)", t)  # sin x
    t = re.sub(r"(\d)\s*(?=[a-zA-Z(])", r"\1*", t)                  # 2x, 2(…), 2pi
    t = re.sub(r"\)\s*(?=[\w(])", ")*", t)                          # )( , )x
    t = re.sub(rf"(?<![a-zA-Z])(?!(?:{funcs})\b)([a-zA-Z])\s*\(", r"\1*(", t)  # x(…) when x is a variable
    t = re.sub(r"(?<=[\w)])\s+(?=[\w(])", "*", t)                   # x sin(x) → x*sin(x)
    return t


def to_sympy(text: str | None):
    """
    Parse an answer such as 'π/4', 'e^2', '\\frac{12}{17}', '2x sin x + x^2 cos x' or
    '72 km/h' into a sympy expression. Untrusted text is never passed to eval or
    sympify: it is reduced to a whitelisted AST and built node by node.
    Returns None when it is not a maths expression.
    """
    if not text or len(text) > 240:
        return None
    import sympy as sp

    plain, percent = _strip_units(_latex_to_plain(text))
    plain = _insert_multiplication(plain).strip()
    if not plain or re.search(r"[^\w\s+\-*/().]", plain):
        return None
    try:
        tree = ast.parse(plain, mode="eval")
    except SyntaxError:
        return None
    x = sp.Symbol("x")
    names = {n: sp.Symbol(n) for n in _SYMPY_NAMES - {"pi", "e", "E", "oo"}}
    names.update({"x": x, "pi": sp.pi, "e": sp.E, "E": sp.E, "oo": sp.oo})
    funcs = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sec": sp.sec, "csc": sp.csc,
             "cot": sp.cot, "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "sinh": sp.sinh,
             "cosh": sp.cosh, "tanh": sp.tanh, "log": sp.log, "ln": sp.log, "exp": sp.exp,
             "sqrt": sp.sqrt, "abs": sp.Abs, "Abs": sp.Abs}
    ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}

    def build(node):
        if isinstance(node, ast.Expression):
            return build(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return sp.nsimplify(node.value) if isinstance(node.value, float) else sp.Integer(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = build(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, exp_ = build(node.left), build(node.right)
            if exp_.is_number and abs(float(exp_)) > 200:
                raise ValueError("exponent too large")
            return base ** exp_
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](build(node.left), build(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in funcs and len(node.args) == 1 and not node.keywords:
            return funcs[node.func.id](build(node.args[0]))
        raise ValueError("unsupported")

    try:
        return build(tree)  # '20%' parses as 20; answers_equivalent also tries 20/100
    except (ValueError, TypeError, ZeroDivisionError, RecursionError):
        return None


_FINAL_LINE = re.compile(
    r"(?:final answer|the answer|answer)\s*(?:is|:|=)\s*[:=]?\s*\**\s*(.+)", re.IGNORECASE
)


def final_expression(text: str | None) -> str | None:
    """
    The text of the final answer: the last \\boxed{...}, else what follows the last
    "final answer is / answer:" marker on its line. None when neither is present.
    """
    if not text:
        return None
    start = text.rfind("\\boxed{")
    if start >= 0:
        pos, depth = start + len("\\boxed"), 0
        for end in range(pos, len(text)):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            if depth == 0:
                return text[pos + 1:end].strip()
    hits = _FINAL_LINE.findall(text)
    if not hits:
        return _closing_expression(text)
    answer = re.split(r"(?<!\d)\.(?!\d)|;|\s+\(|, (?:which|so|since|because)", hits[-1])[0]
    answer = answer.strip().strip("*").strip()
    # "x = 4", "dy/dx = 2x" → the right-hand side; "≈ 7.39" approximations → the exact part
    answer = re.sub(r"^(?:[a-zA-Z]|dy/dx|f'\(x\)|y')\s*=\s*", "", answer)
    return answer.split("≈")[0].strip() or None


_CLOSING_IS = re.compile(r"\bis\s*(?:\\\(|\$)\s*(.+?)\s*(?:\\\)|\$)\s*\.?\s*$", re.DOTALL)
_LAST_DISPLAY = re.compile(r"(?:\\\[|\$\$)(.+?)(?:\\\]|\$\$)", re.DOTALL)


def _closing_expression(text: str) -> str | None:
    """
    No explicit marker: the answer stated at the very end — "… is \\(e^2\\)." or a
    closing display line "\\[ dy/dx = 2x sin x + x^2 cos x \\]" (its right-hand side).
    Only the last ~300 characters count, so a formula quoted mid-solution is ignored.
    """
    tail = text.strip()[-300:]
    m = _CLOSING_IS.search(tail)
    if m:
        return m.group(1).split("=")[-1].strip() or None
    blocks = _LAST_DISPLAY.findall(tail)
    if blocks and tail.rstrip().endswith(("\\]", "$$")):
        return blocks[-1].split("=")[-1].strip() or None
    return _concluding_number(tail)


_CONCLUSION = re.compile(r"^\W*(?:therefore|thus|so|hence|finally|in total|altogether)\b", re.I)
_STATED_AS = re.compile(
    r"(?:\b(?:is|are|was|were|be|equals|comes to|approximately|about|roughly)|=|≈)\s*"
    r"(?:approximately|about|roughly|≈)?\s*(?:₹|rs\.?|\$|inr)?\s*(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_NOT_RATIO = re.compile(r"(?<![\d:/.,])(-?\d[\d,]*(?:\.\d+)?)(?!\s*[:/]\s*\d)")


def _concluding_number(tail: str) -> str | None:
    """
    'Therefore, the EMI is approximately ₹946.94, rounded to 2 decimal places.' → 946.94
    'Therefore, we need to add 10 litres … to get the ratio 3:2.' → 10
    'So, it would take 5 people 48 hours to paint the house.' → 48
    The number stated after 'is / be / = / ≈ / approximately' wins; otherwise the last
    number that is not part of a ratio.
    """
    clean = tail.replace("\\(", " ").replace("\\)", " ").replace("$", " ")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", clean) if s.strip()]
    for sentence in reversed(sentences[-3:]):
        if not _CONCLUSION.match(sentence):
            continue
        stated = _STATED_AS.findall(sentence)
        if stated:
            return stated[0]
        numbers = _NOT_RATIO.findall(sentence)
        return numbers[-1] if numbers else None
    return None


def final_value(text: str | None) -> float | None:
    """Numeric value of the final answer (π/4 → 0.785…), else the last number."""
    expr = to_sympy(final_expression(text))
    if expr is not None and expr.is_number:
        try:
            return float(expr.evalf())
        except (TypeError, ValueError):
            pass
    number = extract_final_number(text or "")
    return float(number) if number is not None else None


def answers_equivalent(candidate: str | None, key: str, symbolic: bool = False,
                       rel_tol: float = 5e-3) -> bool:
    """
    Compare an answer text with a trusted key (a sympy expression string written in
    this repo). Numbers: relative tolerance (answers rounded to 2–3 places count).
    Expressions: equal after simplification, or at three sample points.
    """
    import sympy as sp

    if candidate is None:
        return False
    truth = sp.sympify(key)  # trusted: keys come from evaluation/hard_sets.py
    got = to_sympy(candidate)
    if got is None:
        value = extract_final_number(candidate)
        got = sp.Float(value) if value is not None else None
    if got is None:
        return False
    if symbolic or truth.free_symbols:
        if not got.free_symbols:
            return False
        try:
            if sp.simplify(got - truth) == 0:
                return True
            x = sp.Symbol("x")
            return all(abs(complex(sp.N((got - truth).subs(x, v)))) < 1e-6 for v in (0.7, 1.3, 2.1))
        except Exception:
            return False
    if got.free_symbols:
        return False
    try:
        g, t = float(got.evalf()), float(truth.evalf())
    except (TypeError, ValueError):
        return False
    # Relative tolerance only (plus a tiny floor): an absolute 0.005 once accepted
    # 187/4165 = 0.0449 for 198/4165 = 0.0475
    close = lambda a, b: abs(a - b) <= max(rel_tol * abs(b), 5e-5)
    return close(g, t) or ("%" in candidate and close(g / 100, t))


def symbolic_key(text: str | None) -> str | None:
    """A comparable key for a symbolic final answer ('2*x*sin(x) + x**2*cos(x)')."""
    import sympy as sp

    expr = to_sympy(final_expression(text))
    if expr is None:
        return None
    if expr.is_number:
        try:
            return _format_value(float(expr.evalf()))
        except (TypeError, ValueError):
            return None
    try:
        return sp.sstr(sp.simplify(expr))
    except Exception:
        return None


def option_value(text: str | None) -> float | None:
    """Numeric value of an option such as '4π', '671/1296' or '154 cm²'."""
    expr = to_sympy(text)
    if expr is not None and expr.is_number:
        try:
            return float(expr.evalf())
        except (TypeError, ValueError):
            pass
    number = extract_final_number(text or "")
    return float(number) if number is not None else None


def values_close(a: float | str | None, b: float | str | None, rel_tol: float = 5e-3) -> bool:
    if a is None or b is None:
        return False
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(rel_tol * abs(b), 5e-4)


def answers_match(a: str | None, b: str | None) -> bool:
    """numbers_match, plus equivalence of symbolic keys ('x*(x*cos(x) + 2*sin(x))')."""
    if numbers_match(a, b):
        return True
    if a is None or b is None or not (re.search(r"[a-z]", a) and re.search(r"[a-z]", b)):
        return False
    ea, eb = to_sympy(a), to_sympy(b)
    if ea is None or eb is None:
        return False
    import sympy as sp

    try:
        return sp.simplify(ea - eb) == 0
    except Exception:
        return False


def numbers_match(a: str | None, b: str | None) -> bool:
    """
    True when two answers agree: exactly, or numerically at the precision of the
    less precise one when both have decimals (142857.14 vs 142857.142857).
    An integer never matches a non-integer ('18' vs '18.4').
    """
    if a is None or b is None:
        return False
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    da = len(a.split(".")[1]) if "." in a else 0
    db = len(b.split(".")[1]) if "." in b else 0
    digits = min(da, db)
    return digits >= 1 and round(x, digits) == round(y, digits)


def canonicalize(answers: list[str | None]) -> list[str | None]:
    """Map every answer onto the first-seen answer it matches (for clustering/voting)."""
    reps: list[str] = []
    out: list[str | None] = []
    for ans in answers:
        if ans is None:
            out.append(None)
            continue
        rep = next((r for r in reps if answers_match(r, ans)), None)
        if rep is None:
            reps.append(ans)
            rep = ans
        out.append(rep)
    return out


# =============================================================================
# Exact calculator for plain arithmetic, conversions and linear equations
# =============================================================================


@dataclass(frozen=True)
class Calculation:
    expression: str   # what was evaluated, e.g. "(15/100)*240"
    value: str        # normalised decimal, comparable with extract_final_number
    display: str      # human form, e.g. "7/8" or "x = 4"


_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUM_WORD = re.compile(
    r"\b(?:(?P<tens>" + "|".join(_TENS) + r")(?:[\s-](?P<unit1>" + "|".join(_UNITS) + r"))?"
    r"|(?P<unit>" + "|".join(_UNITS) + r"))(?:\s+(?P<mult>hundred|thousand))?\b",
    re.IGNORECASE,
)

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod,
}
_COUNTING = {"perm": math.perm, "comb": math.comb, "fact": math.factorial}


def _words_to_digits(text: str) -> str:
    def repl(m: re.Match) -> str:
        if m.group("tens"):
            value = _TENS[m.group("tens").lower()] + _UNITS.get((m.group("unit1") or "").lower(), 0)
        else:
            value = _UNITS[m.group("unit").lower()]
        if m.group("mult"):
            value *= 100 if m.group("mult").lower() == "hundred" else 1000
        return str(value)

    return _NUM_WORD.sub(repl, text)


def _safe_eval(expr: str, x: Fraction | None = None) -> Fraction | float:
    """Evaluate +, -, *, /, %, ** (small exponents), sqrt() and x — nothing else."""

    def ev(node: ast.AST) -> Fraction | float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return Fraction(str(node.value))
        if isinstance(node, ast.Name) and node.id == "x" and x is not None:
            return x
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, exp = ev(node.left), ev(node.right)
            if abs(exp) > 64 or abs(base) > 1e12:
                raise ValueError("exponent too large")
            if isinstance(exp, Fraction) and exp.denominator == 1:
                return base ** int(exp)
            return float(base) ** float(exp)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(left, float) or isinstance(right, float):
                left, right = float(left), float(right)
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "sqrt" and len(node.args) == 1 and not node.keywords:
            v = float(ev(node.args[0]))
            root = math.sqrt(v)
            return Fraction(int(round(root))) if root.is_integer() else root
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _COUNTING and not node.keywords:
            args = [ev(a) for a in node.args]
            if any(not isinstance(a, Fraction) or a.denominator != 1 or not 0 <= a <= 500
                   for a in args):
                raise ValueError("counting functions need small non-negative integers")
            return Fraction(_COUNTING[node.func.id](*(int(a) for a in args)))
        raise ValueError(f"unsupported syntax: {ast.dump(node)[:40]}")

    return ev(ast.parse(expr, mode="eval"))


def _value_strings(value: Fraction | float, prefer_fraction: bool) -> tuple[str, str]:
    as_float = float(value)
    decimal = _format_value(as_float) or str(as_float)
    if isinstance(value, Fraction) and value.denominator != 1 and prefer_fraction \
            and value.denominator <= 1000:
        return decimal, f"{value.numerator}/{value.denominator}"
    return decimal, decimal


_LEADING = re.compile(
    r"^(?:please\s+)?(?:what(?:'s| is| are)|calculate|compute|evaluate|find|work out|"
    r"how much is|simplify|tell me)\s+(?:the\s+(?:value|result)\s+of\s+)?",
    re.IGNORECASE,
)
_ARITH_ONLY = re.compile(r"^(?:[\d.+\-*/()%,\s]|sqrt|perm|comb|fact)+$")


def calculate(query: str) -> Calculation | None:
    """
    Exactly solve a query that is plain arithmetic, a percentage, a square root or
    power, a Fahrenheit/Celsius conversion, or a linear equation in x. Returns None
    for anything else (including word problems, which need reasoning).
    """
    q = re.sub(r"[?.\s]+$", "", strip_options(query))
    q = re.sub(r"(?<=\D)!+$", "", q)  # a trailing '!' is punctuation unless it is n!
    q = _words_to_digits(q.strip().replace(",", ""))
    try:
        return _temperature(q) or _linear_equation(q) or _arithmetic(q)
    except (ValueError, ZeroDivisionError, OverflowError, SyntaxError, TypeError):
        return None


def _temperature(q: str) -> Calculation | None:
    m = re.search(
        r"(-?\d+(?:\.\d+)?)\s*°?\s*(?:degrees?\s*)?(fahrenheit|celsius|f|c)\b.*?\b(?:to|in|into)\s*"
        r"(?:degrees?\s*)?°?\s*(fahrenheit|celsius|f|c)\b",
        q, re.IGNORECASE,
    )
    if not m:
        return None
    value, src, dst = float(m.group(1)), m.group(2)[0].lower(), m.group(3)[0].lower()
    if src == dst:
        return None
    result = (value - 32) * 5 / 9 if src == "f" else value * 9 / 5 + 32
    expr = f"({value:g} - 32) * 5/9" if src == "f" else f"{value:g} * 9/5 + 32"
    dec = _format_value(result) or str(result)
    unit = "°C" if dst == "c" else "°F"
    return Calculation(expr, dec, f"{round(result, 2):g} {unit}")


def _to_expr(text: str) -> str:
    t = text.lower()
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*", r"(\1/100)*", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", r"(\1/100)", t)
    t = re.sub(r"(?:the\s+)?square root of\s*(\(?[\d.]+\)?)", r"sqrt(\1)", t)
    t = re.sub(r"(?:the\s+)?cube of\s*([\d.]+)", r"(\1)**3", t)
    t = re.sub(r"([\d.)]+)\s*squared", r"(\1)**2", t)
    t = re.sub(r"([\d.)]+)\s*cubed", r"(\1)**3", t)
    t = re.sub(r"\s*(?:to the power of|raised to(?: the power of)?)\s*", "**", t)
    t = re.sub(r"\b(?:multiplied by|times)\b", "*", t)
    t = re.sub(r"\b(?:plus|added to)\b", "+", t)
    t = re.sub(r"\bminus\b", "-", t)
    t = re.sub(r"\b(?:divided by|over)\b", "/", t)
    t = t.replace("×", "*").replace("÷", "/").replace("^", "**")
    t = re.sub(r"(\d)\s*x\s*(\d)", r"\1*\2", t)  # '3 x 4' → 3*4
    # nPr, nCr and n! as written in exam papers: 8P3, 7C3, 5!
    t = re.sub(r"\b(\d+)\s*p\s*(\d+)\b", r"perm(\1,\2)", t)
    t = re.sub(r"\b(\d+)\s*c\s*(\d+)\b", r"comb(\1,\2)", t)
    t = re.sub(r"(\d+)\s*!", r"fact(\1)", t)
    return t.strip()


def _arithmetic(q: str) -> Calculation | None:
    body = _LEADING.sub("", q).strip()
    expr = _to_expr(body)
    if not _ARITH_ONLY.match(expr) or not re.search(r"[+\-*/%]|sqrt|perm|comb|fact", expr.lstrip("-")):
        return None
    value = _safe_eval(expr)
    prefer_fraction = "/" in body and "sqrt" not in expr and "." not in body
    dec, display = _value_strings(value, prefer_fraction)
    return Calculation(expr, dec, display)


def _linear_equation(q: str) -> Calculation | None:
    m = re.match(
        r"^(?:solve|find x|solve for x)(?:\s+for\s+x)?\s*[:,]?\s*(?:the equation\s+)?(.+?)=(.+)$",
        q, re.IGNORECASE,
    )
    if not m:
        return None
    lhs, rhs = (re.sub(r"(\d|\))\s*x", r"\1*x", _to_expr(side)) for side in m.groups())
    if re.search(r"[a-wyz]", lhs + rhs) or "x" not in lhs + rhs:
        return None

    def f(x: Fraction) -> Fraction:
        return Fraction(_safe_eval(lhs, x)) - Fraction(_safe_eval(rhs, x))

    f0, f1, f2 = f(Fraction(0)), f(Fraction(1)), f(Fraction(2))
    slope = f1 - f0
    if slope == 0 or f2 - f1 != slope:
        return None  # no unique solution, or not linear
    root = -f0 / slope
    dec, frac = _value_strings(root, prefer_fraction=True)
    return Calculation(f"{lhs} = {rhs}", dec, f"x = {frac}")


# =============================================================================
# Task routing (rule-based layer; the LLM router handles ambiguous queries)
# =============================================================================

_REASONING_CUES = re.compile(
    r"\b(how many|how much|how long|how far|how old|what is the total|total|calculate|"
    r"what percentage|what fraction|in all|altogether|left over|remain(?:ing|s)?|"
    r"each|per)\b",
    re.IGNORECASE,
)
_MATH_WORDS = re.compile(
    r"\b(solve|simplify|calculate|compute|evaluate|derivative|differentiate|integral|"
    r"integrate|square root|squared|cubed|factorial|percent of|divided by|multiplied by|"
    r"sum of|product of|average of|mean of|"
    # counting and geometry (school / competitive-exam wording)
    r"ways|arrange(?:d|ments?)?|permutations?|combinations?|diagonals|handshakes|"
    r"probability|area|perimeter|circumference|radius|radii|diameter|volume|arc|sector|"
    r"chord|tangent|\d+\s*[PC]\s*\d+)\b|[=+*/^%×÷]",
    re.IGNORECASE,
)
_TIME_SENSITIVE = re.compile(
    r"\b(current(?:ly)?|latest|today(?:'s)?|right now|as of|this (?:year|month|week)|"
    r"recent(?:ly)?|newest|upcoming|so far|nowadays|these days|at present|"
    r"still (?:alive|in office|running))\b",
    re.IGNORECASE,
)

TASK_TYPES = ("math", "factual", "time_sensitive")

# Senior-school / competitive-exam maths: the fast path is not trusted here, because
# on hard problems all samples can share one mistake
_ADVANCED_MATH = re.compile(
    r"\b(limit of|lim|derivative|differentiat\w*|dy/dx|integral|integrate|antiderivative|"
    r"matrix|matrices|determinant|inverse of|vectors?|dot product|cross product|probability|"
    r"binomial|bayes|expected value|variance|distance of the point|maximum value|minimum value|"
    r"maxima|minima|local maximum|local minimum|continuous|differentiable|differential equation|"
    r"logarithm|progression|coefficient|expansion|permutations?|combinations?|arrangements?|"
    r"in how many ways|magnitude|entries of|can be formed|without repetition|digit numbers?|"
    r"positive integers|integer solutions|solutions? in|area enclosed|area bounded|"
    r"area of the region|between the curves|shortest distance|between the lines)\b|[∫]"
    # notation without the words: [[2, 1], [1, 2]], a × b, 2i + j - k, A^5
    r"|\[\s*\[|\b[a-z]\s*×\s*[a-z]\b|\d*\s*i\s*[+-]\s*\d*\s*j\s*[+-]\s*\d*\s*k\b|\b[A-Z]\^\d",
    re.IGNORECASE,
)


def is_advanced_math(prompt: str) -> bool:
    return bool(_ADVANCED_MATH.search(prompt or ""))


def classify_task_heuristic(prompt: str) -> tuple[str, bool]:
    """
    Rule-based task type: ('math' | 'factual' | 'time_sensitive', confident).

    Confident only when the signal is unambiguous: an exactly solvable expression,
    a time-sensitivity word, a word problem (2+ quantities plus a quantity
    question), or a math keyword. Everything else (e.g. 'What happened on July
    20, 1969?') is returned as not confident, for the LLM router to decide.
    """
    if calculate(prompt) is not None:
        return "math", True
    if _TIME_SENSITIVE.search(prompt):
        return "time_sensitive", True
    quantities = re.findall(r"\d[\d,]*(?:\.\d+)?", prompt)
    if len(quantities) >= 2 and _REASONING_CUES.search(prompt):
        return "math", True
    if _MATH_WORDS.search(prompt) and (quantities or re.search(r"\b[xyz]\b", prompt)):
        return "math", True
    if len(quantities) == 1 and _REASONING_CUES.search(prompt):
        return "math", False
    return "factual", False


def is_numeric_reasoning_query(prompt: str) -> bool:
    """Rule-based check for quantitative questions (no LLM call)."""
    return classify_task_heuristic(prompt)[0] == "math"


def leading_sentences(text: str, max_sentences: int = 2, max_chars: int = 400) -> str:
    """Return the first sentences of a response (the core claim) for NLI comparison."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    head = " ".join(sentences[:max_sentences])
    return head[:max_chars]
