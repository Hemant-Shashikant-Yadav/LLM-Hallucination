# app/answer_utils.py — reading answers, comparing them, exact maths, routing rules

The most important helper file. It decides **what answer a response gives** (a number, an
expression, an option letter, yes/no), whether two answers are **the same**, solves plain
arithmetic **exactly**, and holds the **rule-based task router**. Layer 1 clusters answers
with it, Layer 3 votes with it, and every evaluation scorer uses it. Most measurement bugs
in this project were here, so every change is checked against all stored results.

Nothing here calls `eval` on model text: maths is parsed into a whitelisted syntax tree.

## Number reading (lines 17–170)

| Lines | What it does |
|---|---|
| 19–20 | `_NUMBER`: a number token (`-`, `$`, digits, commas, decimals). `_END`: a number must not run on into another digit or `/`. |
| 24–31 | `_VALUE`: one regex for every numeric form: `\frac{3}{4}`, mixed `2 1/2`, `3/4`, plain numbers with a scale word (`1.5 million`). |
| 35–40 | `_MARKERS`: where a final answer is announced, strongest first: `####`, `\boxed{`, `final answer`, `answer is`. |
| 43–58 | `normalize_number` / `_format_value`: `$1,234.50` → `1234.5`, `18.0` → `18`; rejects NaN/∞. |
| 61–68 | `_is_symbolic`: `2x`, `x^2` are algebra, not numbers (but `5 km` is a number). |
| 71–98 | `_match_value` turns a `_VALUE` match into a decimal string; `_values` lists every number in a text. |
| 101–132 | **`extract_final_number`** (the GSM8K scorer): the value after the strongest marker; a symbolic answer after a marker gives `None`. With no marker, the number stated in the **concluding sentence** (lines 127–131), else the last number. |
| 135–139 | `_MARKERS_PRESENT`, `_SYMBOLIC_MARKER`: detect markers; capture a symbolic final answer. |
| 142–169 | **`extract_final_answer`** (used for clustering and voting): a symbolic key for answers like `π/4`, `2x sin x` (so `4` isn't grabbed from `\frac{\pi}{4}`); else the concluding-sentence number; else the final number; else a normalised symbolic string. |

## Multiple choice and yes/no (lines 172–309)

| Lines | What it does |
|---|---|
| 177 | `_OPTION`: an option label: `(a)`, `(B)`, or `a)` / `A.` at a line start. |
| 178–193 | Patterns for picking an option: `answer is (b)`, `… is (d) aquifers`, a closing label line, endorsing words (`correct`, `best`) and rejecting words (`not`, `incorrect`). |
| 196–215 | **Yes/no**: `is_yes_no_question` (starts with is/are/do/can/…, but not `Can you …` and not multiple choice); `extract_yes_no` (`Yes, …`, `No. …`, `the answer is no`; `None` if never said). |
| 218–236 | **`parse_options`**: `{'a': '120', 'b': '60', …}`; only a consecutive run a, b, c… counts. |
| 239–246 | `strip_options`: the question without its options. |
| 249–264 | `_boxed_choice`: the option a `\boxed{…}` answer names (a letter, `(c)`, or one option's text); `_NONE_PICKED` recognises `\boxed{None}`. |
| 267–309 | **`extract_choice`**, in order: a final **value** that matches exactly one option (small models compute right but mis-copy the letter); a boxed answer; an explicit letter mark; a closing label line; option text in a sentence that endorses it and doesn't reject it. |

## Safe symbolic maths (lines 312–498)

| Lines | What it does |
|---|---|
| 316–325 | Allowed function names and symbols; unit words stripped from answers (`km/h`, `cm²`, `rupees`, `ways`…). |
| 328–368 | `_replace_braced`, `_latex_to_plain`: LaTeX → plain maths (`\frac{a}{b}` → `((a)/(b))`, `\sqrt`, `\pi`, `^` → `**`, `\cdot` → `*`, degree signs, `\text{}` removed). |
| 371–380 | `_strip_units`: drops currency and trailing units; notes a `%`. |
| 383–390 | `_insert_multiplication`: `2x` → `2*x`, `sin x` → `sin(x)`, `)(` → `)*(`. |
| 393–448 | **`to_sympy`**: answer text → sympy expression by parsing a Python syntax tree and building only whitelisted nodes (numbers, allowed names, + − × ÷, powers up to 200, allowed functions). Anything else → `None`. |
| 451–477 | **`final_expression`**: the text of the final answer: the last `\boxed{…}` (nested braces handled), else what follows the last `answer is` on its line (`x = 4` → `4`), else the closing expression. |
| 480–497 | `_closing_expression`: no marker: `… is \(e^2\).`, or the right-hand side of a closing display line, else the concluding number. |
| 500–536 | **`_concluding_number`**: in a sentence starting `Therefore/Thus/So…`: the result after the last `=`; else the number after `is/equals/approximately`; else the last number that is not a ratio and not a span the question gave (`in 4 weeks`). Fixed the "36 hours in 4 weeks → 4" scoring bug. |

## Comparing answers (lines 539–697)

| Lines | What it does |
|---|---|
| 539–548 | `final_value`: numeric value of the final answer (`π/4` → 0.785…). |
| 551–556 | `_latex_text`: LaTeX as comparable text (no spacing commands, `\text{}`, `$`). |
| 559–604 | **`answers_equivalent(candidate, key)`** (hard-set, ladder, MATH and JEE scorer): exact text match; else parse the key (Python expression, or LaTeX for MATH); keys that aren't maths (tuples, words) compare as text; expressions must simplify to the same thing or agree at three points; numbers within 0.5% relative tolerance (absolute floor 5e-5), with a `%` answer also tried divided by 100. |
| 607–622 | `symbolic_key`: a canonical string for a symbolic answer, for voting. |
| 625–644 | `option_value`, `values_close`: numeric value of an option (`4π`, `671/1296`, `154 cm²`) and the closeness test. |
| 647–681 | `answers_match`, **`numbers_match`**: same answer exactly, or equal to the precision of the less precise one (`142857.14` = `142857.142857`); an integer never matches a decimal (`18` ≠ `18.4`); symbolic keys equal if they simplify the same. |
| 684–697 | **`canonicalize`**: maps each answer onto the first-seen equivalent one, so votes and clusters count `0.5` and `1/2` together. |

## Exact calculator (lines 700–894)

| Lines | What it does |
|---|---|
| 705–709 | `Calculation`: expression evaluated, decimal value, human display (`7/8`, `x = 4`). |
| 712–745 | Spelled-out numbers → digits (`three times seven` → `3 * 7`). |
| 748–788 | `_safe_eval`: exact fraction arithmetic over a whitelisted tree: + − × ÷ %, small powers, `sqrt`, and `perm`/`comb`/`fact` for nPr, nCr, n!. |
| 791–797 | Decimal and display strings (fractions kept as fractions when the question used them). |
| 800–805 | Question openers to strip (`what is`, `calculate`…) and the allowed character set. |
| 808–820 | **`calculate(query)`**: tries a temperature conversion, then a linear equation, then arithmetic; returns `None` for anything else (word problems need reasoning). |
| 823–838 | `_temperature`: °F ↔ °C. |
| 841–860 | `_to_expr`: words → operators (`15% of 240` → `(15/100)*240`, `squared`, `divided by`, `8P3` → `perm(8,3)`, `5!` → `fact(5)`). |
| 863–871 | `_arithmetic`: evaluates a pure expression. |
| 874–894 | `_linear_equation`: `Solve 2x + 3 = 11` by evaluating at x = 0, 1, 2 (must be linear) → `x = 4`. |

When the router finds an exact value and the draft disagrees, the exact value wins (router.py).

## Task routing rules (lines 897–989)

| Lines | What it does |
|---|---|
| 901–923 | Cue patterns: multiples (`half`, `twice`, `dozen`); quantity questions (`how many`, `total`, `each`); maths words (`solve`, `area`, `probability`, `8C3`); **time-sensitive** words (`current`, `latest`, `today`, `as of`…). |
| 929–945 | `_ADVANCED_MATH`, `is_advanced_math`: calculus, matrices, vectors, probability, counting (and notation like `[[1,2],[3,4]]`, `a × b`, `2i + j − k`). These never take the fast path and get a larger sample budget. |
| 948–977 | **`classify_task_heuristic(prompt)`** → `(type, confident)`, in order: exactly calculable → maths; **word problem** (2+ quantities, digits or spelled out, plus a quantity question) → maths, checked before time words so `Today is Dec 31 … how many calories` is maths; time-sensitive words in the question itself (not in quoted text or options) → time_sensitive; maths words with numbers or x/y/z → maths; one quantity + quantity question → maths, not confident; otherwise factual, not confident (the LLM router decides). |
| 980–982 | `is_numeric_reasoning_query`: maths or not, rules only. |
| 985–989 | `leading_sentences`: the first 1–2 sentences of an answer, used for NLI clustering in Layer 1. |
