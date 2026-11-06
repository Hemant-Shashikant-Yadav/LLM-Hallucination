"""
Offline tests for advanced-maths support: the safe symbolic parser, the sympy
sandbox, advanced-maths routing, and adaptive self-consistency sampling.

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from app.answer_utils import (
    answers_equivalent,
    answers_match,
    canonicalize,
    extract_choice,
    extract_final_answer,
    final_expression,
    is_advanced_math,
    parse_options,
    to_sympy,
)
from app.config import settings
from app.layer1_triage.semantic_entropy import SemanticEntropyTriage
from app.layer3_halluclean.program_verifier import ProgramVerifier
from app.models.schemas import TriageDecision


class SymbolicParserTests(unittest.TestCase):
    CASES = [
        (r"\boxed{e^{2}}", "exp(2)", False, True),
        (r"Final answer: 7.39", "exp(2)", False, True),
        (r"\boxed{\frac{\pi}{4}}", "pi/4", False, True),
        (r"\boxed{\frac{\pi}{2}}", "pi/4", False, False),
        (r"\boxed{x(2\sin x + x\cos x)}", "2*x*sin(x) + x**2*cos(x)", True, True),
        (r"Final answer: $2x\sin x + x^2\cos x$", "2*x*sin(x) + x**2*cos(x)", True, True),
        (r"\boxed{2x\cos x}", "2*x*sin(x) + x**2*cos(x)", True, False),
        (r"\boxed{70.53^\circ}", "acos(1/3)*180/pi", False, True),
        (r"\boxed{-\frac{1}{2}}", "-1/2", False, True),
        (r"Final answer: ₹9414.69", "9414.6942", False, True),
        (r"\boxed{12.5\%}", "25/2", False, True),
        (r"Final answer: 72 km/h", "72", False, True),
        (r"Final answer: x = 4", "4", False, True),
    ]

    def test_equivalence(self):
        for text, key, symbolic, want in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(answers_equivalent(final_expression(text), key, symbolic), want)

    def test_small_probabilities_are_not_waved_through(self):
        self.assertFalse(answers_equivalent(r"\frac{187}{4165}", "198/4165"))
        self.assertTrue(answers_equivalent("0.0475", "198/4165"))
        self.assertTrue(answers_equivalent("0.2344", "15/64"))

    def test_closing_expression_without_marker(self):
        self.assertEqual(final_expression(r"Thus, the limit is \(e^2\)."), "e^2")
        self.assertEqual(
            final_expression("So the derivative is:\n\n\\[ \\frac{dy}{dx} = 2x \\sin x + x^2 \\cos x \\]"),
            "2x \\sin x + x^2 \\cos x",
        )
        # A formula quoted mid-solution, followed by more prose, is not the answer
        self.assertIsNone(final_expression("\\[ f(x) = x^2 \\]\nNext we consider the domain of f."))
        # The old parser read '1' here (from 'at x = 1'); the stated answer is e^2
        self.assertEqual(
            extract_final_answer(r"Evaluate at \( x = 1 \): ... the derivative at \( x = 1 \) is \( e^2 \)."),
            "7.389056",
        )

    def test_never_evaluates_code(self):
        for text in ["__import__('os').system('x')", "().__class__", "x.__dict__", "lambda: 1",
                     "open('f')", "[1,2][0]", "2**10**10"]:
            with self.subTest(text=text):
                self.assertIsNone(to_sympy(text))

    def test_keys_and_matching(self):
        self.assertEqual(extract_final_answer(r"\boxed{\frac{\pi}{4}}"), "0.785398")
        self.assertEqual(extract_final_answer(r"\boxed{18 \text{ dollars}}"), "18")
        self.assertTrue(answers_match("x*(x*cos(x) + 2*sin(x))", "x**2*cos(x) + 2*x*sin(x)"))
        self.assertFalse(answers_match("2*x*cos(x)", "x**2*cos(x) + 2*x*sin(x)"))
        self.assertEqual(
            canonicalize(["x*(x*cos(x) + 2*sin(x))", "x**2*cos(x) + 2*x*sin(x)", "2*x"]),
            ["x*(x*cos(x) + 2*sin(x))", "x*(x*cos(x) + 2*sin(x))", "2*x"],
        )

    def test_symbolic_options(self):
        q = "Area of x^2 + y^2 = 16 in the first quadrant?\n(a) 16π\n(b) 8π\n(c) 4π\n(d) 2π"
        options = parse_options(q)
        self.assertEqual(extract_choice(r"The area is \boxed{4\pi}.", options), "c")
        self.assertEqual(extract_choice("So the area is 12.566 square units.", options), "c")


class SandboxTests(unittest.TestCase):
    def run_code(self, code):
        return asyncio.run(ProgramVerifier(None)._run(code))

    def test_sympy_programs(self):
        self.assertEqual(self.run_code(
            "import sympy as sp\ndef solution():\n    x = sp.symbols('x')\n"
            "    return sp.integrate(sp.sin(x)**2, (x, 0, sp.pi/2))\n"), ("0.785398", "ok"))
        ans, status = self.run_code(
            "import sympy as sp\ndef solution():\n    x = sp.symbols('x')\n"
            "    return sp.diff(x**2*sp.sin(x), x)\n")
        self.assertEqual(status, "ok")
        self.assertTrue(answers_match(ans, "2*x*sin(x) + x**2*cos(x)"))
        self.assertEqual(self.run_code(
            "from itertools import permutations\ndef solution():\n"
            "    return len(set(permutations('APPLE')))\n"), ("60", "ok"))

    def test_harmless_modules_allowed_and_import_errors_explained(self):
        self.assertEqual(self.run_code(
            "from collections import Counter\ndef solution():\n"
            "    return Counter('ENGINEERING')['E']\n"), ("3", "ok"))
        ans, status = self.run_code("import numpy as np\ndef solution():\n    return 1\n")
        self.assertIsNone(ans)
        self.assertIn("use only:", status)  # the repair prompt tells the model what is allowed

    def test_blocks_string_parsing(self):
        for code in [
            "import sympy as sp\ndef solution():\n    return sp.sympify('1')\n",
            "from sympy import sympify\ndef solution():\n    return sympify('1')\n",
            "import sympy.parsing.sympy_parser as p\ndef solution():\n    return 1\n",
            "from sympy import *\ndef solution():\n    return S('2')\n",
            "import os\ndef solution():\n    return 1\n",
        ]:
            with self.subTest(code=code):
                ans, status = self.run_code(code)
                self.assertIsNone(ans)
                self.assertTrue(status.startswith("rejected"), status)

    def test_non_finite_result_fails_alone(self):
        # Seen live: a limit mis-computed as oo crashed the whole request with a 500
        for code in ["import sympy as sp\ndef solution():\n    return sp.oo\n",
                     "def solution():\n    return float('nan')\n"]:
            with self.subTest(code=code):
                ans, status = self.run_code(code)
                self.assertIsNone(ans)
                self.assertTrue(status.startswith("non-finite"), status)

    def test_string_result_is_not_parsed(self):
        ans, status = self.run_code("def solution():\n    return \"__import__('os')\"\n")
        self.assertIsNone(ans)
        self.assertIn("must return a number or a sympy expression", status)


class AdvancedRoutingTests(unittest.TestCase):
    def test_detection(self):
        for q in ["Evaluate the definite integral of x e^x dx from 0 to 1.",
                  "Find the determinant of the matrix [[1, 2], [3, 4]].",
                  "In how many ways can 5 people sit around a table?",
                  "What is the probability of getting exactly 4 heads in 6 tosses?"]:
            self.assertTrue(is_advanced_math(q), q)
        for q in ["Janet's ducks lay 16 eggs per day. How much does she make?",
                  "A plane flies 500 miles in 2 hours. What is its speed?",
                  "What is the maximum number of apples Tom can buy with $20 at $3 each?"]:
            self.assertFalse(is_advanced_math(q), q)


class _Agreeing:
    async def generate(self, **_kwargs):
        return SimpleNamespace(text="The answer is 1/2.", done_reason="stop", truncated=False)


class _AgreeNLI:
    is_loaded = True

    async def score_pairs(self, pairs):
        return [{"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05} for _ in pairs]


class AdvancedTriageTests(unittest.IsolatedAsyncioTestCase):
    async def test_advanced_math_never_fast_path(self):
        triage = SemanticEntropyTriage(_Agreeing(), _AgreeNLI(), num_samples=4)
        result, estimate = await triage.triage(
            "Evaluate the definite integral of (ln x)/x dx from x = 1 to x = e.",
            "The answer is 1/2.", task_type="math",
        )
        self.assertEqual(estimate.entropy, 0.0)
        self.assertEqual(result.decision, TriageDecision.REASONING_UNCERTAIN)


class TruncatedSampleTests(unittest.IsolatedAsyncioTestCase):
    async def test_cut_off_maths_samples_do_not_vote(self):
        outputs = iter([("... a × b = i(1·2 − (−1)(−1)) − j(", "length"),
                        ("|a × b| = √35, so the answer is \\boxed{\\sqrt{35}}", "stop")] * 4)

        class Client:
            async def generate(self, **_kw):
                text, reason = next(outputs)
                return SimpleNamespace(text=text, done_reason=reason, truncated=reason == "length")

        triage = SemanticEntropyTriage(Client(), _AgreeNLI(), num_samples=4)
        kept = await triage.sample("Find |a × b|.", None, n=4, seed_base=0)
        self.assertEqual(len(kept), 2)
        self.assertTrue(all("boxed" in k for k in kept))


class AdaptiveSamplingTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, answers: list[str]):
        from app.router import DefenseInDepthRouter

        stream = iter(answers)
        calls = []

        async def sample(prompt, model, n, seed_base, max_tokens=None):
            calls.append(n)
            return [next(stream) for _ in range(n)]

        fake = SimpleNamespace(_se_triage=SimpleNamespace(sample=sample))
        request = SimpleNamespace(prompt="Tom has 2 and 3 apples. How many in total?", model="m")
        draft = ["Final answer: 5"]
        out, drawn, share = await DefenseInDepthRouter._adaptive_samples(fake, request, draft)
        return out, drawn, share, calls

    async def test_clear_vote_stops_at_minimum(self):
        _, drawn, share, calls = await self._run(["Final answer: 5"] * 30)
        self.assertEqual(drawn, settings.l3_extra_samples)
        self.assertEqual(calls, [settings.l3_extra_samples])
        self.assertEqual(share, 1.0)

    async def test_stops_when_samples_come_back_cut_off(self):
        from app.router import DefenseInDepthRouter

        calls = []

        async def sample(prompt, model, n, seed_base, max_tokens=None):
            calls.append(n)
            return ["Final answer: 5"] if not calls[1:] else []  # 1 of 6 usable, then none

        fake = SimpleNamespace(_se_triage=SimpleNamespace(sample=sample))
        request = SimpleNamespace(prompt="Tom has 2 and 3 apples. How many in total?", model="m")
        _, drawn, _ = await DefenseInDepthRouter._adaptive_samples(
            fake, request, ["Final answer: 6", "Final answer: 7"]
        )
        self.assertEqual(calls, [settings.l3_extra_samples])  # no further batches
        self.assertEqual(drawn, settings.l3_extra_samples)

    async def test_close_vote_draws_more(self):
        split = ["Final answer: 5", "Final answer: 6"] * 15
        _, drawn, share, calls = await self._run(split)
        self.assertEqual(drawn, settings.l3_max_extra_samples)
        self.assertGreater(len(calls), 1)
        self.assertLess(share, settings.l3_stop_agreement)


if __name__ == "__main__":
    unittest.main()
