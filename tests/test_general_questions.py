"""
Offline tests for general-question handling (doc/IMPROVEMENT_ROADMAP.md §4):
task routing, answer parsing, exact calculation and the freshness rule.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from app.answer_utils import (
    calculate,
    canonicalize,
    classify_task_heuristic,
    extract_choice,
    extract_final_answer,
    extract_final_number,
    numbers_match,
    parse_options,
)
from app.config import settings
from app.layer1_triage.semantic_entropy import SemanticEntropyTriage
from app.layer1_triage.task_router import TaskRouter
from app.layer3_halluclean.program_verifier import ProgramVerifier
from app.models.schemas import TriageDecision
from app.telemetry import degradations, request_trace

# (question, expected route, exact answer or None). Mirrors the roadmap smoke list.
SMOKE = [
    ("What is the capital of Australia?", "factual", None),
    ('Who wrote "Pride and Prejudice"?', "factual", None),
    ("What happened on July 20, 1969?", "factual", None),
    ("What was India's population in 2011?", "factual", None),
    ("Which is taller, Mount Everest (8849 m) or K2 (8611 m)?", "factual", None),
    ("Who is the current Prime Minister of the UK?", "time_sensitive", None),
    ("Explain how vaccines work.", "factual", None),
    ("What is 15% of 240?", "math", "36"),
    ("What is three times seven?", "math", "21"),
    ("What is the square root of 144?", "math", "12"),
    ("Solve 2x + 3 = 11.", "math", "4"),
    ("Simplify 3/4 + 1/8.", "math", "0.875"),
    ("Convert 100 °F to Celsius.", "math", "37.777778"),
    ("What is the derivative of x^2?", "math", None),
    ("A train travels 60 km/h for 2.5 hours. How far does it go?", "math", None),
    ("Tom has 3 boxes of 12 eggs and breaks 5. How many eggs are left?", "math", None),
    ("What is 7 squared?", "math", "49"),
    ("What is 1000000 divided by 7?", "math", "142857.142857"),
]


class RoutingTests(unittest.TestCase):
    def test_smoke_list_routes(self):
        for question, expected, _ in SMOKE:
            with self.subTest(question=question):
                label, confident = classify_task_heuristic(question)
                # Rules must never be confidently wrong; unsure cases go to the LLM
                if confident:
                    self.assertEqual(label, expected)
                else:
                    self.assertEqual(expected, "factual", "math must be caught by rules")

    def test_gsm8k_style_word_problem_is_math(self):
        q = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
             "and bakes muffins with four. She sells the rest for $2 per egg. "
             "How much does she make every day?")
        self.assertEqual(classify_task_heuristic(q), ("math", True))


class CalculatorTests(unittest.TestCase):
    def test_smoke_list_exact_answers(self):
        for question, _, exact in SMOKE:
            with self.subTest(question=question):
                calc = calculate(question)
                if exact is None:
                    self.assertIsNone(calc)
                else:
                    self.assertIsNotNone(calc)
                    self.assertEqual(calc.value, exact)

    def test_displays(self):
        self.assertEqual(calculate("Simplify 3/4 + 1/8").display, "7/8")
        self.assertEqual(calculate("Solve 2x + 3 = 11").display, "x = 4")
        self.assertEqual(calculate("Solve for x: 3x - 2 = x + 5").display, "x = 7/2")

    def test_rejects_unsafe_or_nonlinear(self):
        for q in ["What is __import__('os')?", "Solve x^2 = 4", "What is 2 ** 1000000?",
                  "What is 5?", "Solve 2y + 3 = 11"]:
            with self.subTest(q=q):
                self.assertIsNone(calculate(q))


class ParsingTests(unittest.TestCase):
    CASES = [
        ("The answer is 3/4.", "0.75"), ("The result is 7/8", "0.875"),
        ("1.5 million people", "1500000"), ("Final answer: 2x", None),
        ("about 142,857.14", "142857.14"), (r"\boxed{\frac{7}{8}}", "0.875"),
        ("so he needs 2 1/2 cups", "2.5"), (r"So the total is \boxed{\$1,250}.", "1250"),
        ("The answer is 18. She sells 9 eggs.", "18"), ("costs $3 each. #### 54", "54"),
        ("Speed is 60 km/h for 2.5 hours = 150 km", "150"), ("It is -40 degrees.", "-40"),
    ]

    def test_extract_final_number(self):
        for text, want in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(extract_final_number(text), want)

    def test_symbolic_answer_key(self):
        self.assertEqual(extract_final_answer("Final answer: 2x"), "2*x")
        self.assertEqual(extract_final_answer("Final answer: 2 * x."), "2*x")

    def test_tolerant_matching(self):
        self.assertTrue(numbers_match("142857.14", "142857.142857"))
        self.assertTrue(numbers_match("0.875", "0.88"))
        self.assertFalse(numbers_match("18", "18.4"))
        self.assertEqual(canonicalize(["37.8", "37.777778", "38", None]), ["37.8", "37.8", "38", None])


class MCQTests(unittest.TestCase):
    Q = ("In how many different ways can the letters of the word APPLE be arranged?\n"
         "(a) 120\n(b) 60\n(c) 24\n(d) 30")
    OPTS = {"a": "120", "b": "60", "c": "24", "d": "30"}

    def test_parse_options(self):
        self.assertEqual(parse_options(self.Q), self.OPTS)
        self.assertEqual(parse_options("Pick one: A. Paris B. Rome"), {})  # not line-start
        self.assertEqual(parse_options("A. Paris\nB. Rome\nC. Oslo"), {"a": "Paris", "b": "Rome", "c": "Oslo"})
        self.assertEqual(parse_options("What is (a) here?"), {})

    def test_extract_choice(self):
        cases = [
            ("5!/2! = 60. Final answer: (b) 60", "b"), ("So the answer is (b).", "b"),
            ("Answer: B", "b"), (r"\boxed{B}", "b"), ("Therefore there are 60 ways.", "b"),
            ("The correct option is (a) 60", "b"),  # value beats a mis-copied letter
            ("the answer is a number equal to 120", "a"), ("None of these", None),
        ]
        for text, want in cases:
            with self.subTest(text=text):
                self.assertEqual(extract_choice(text, self.OPTS), want)

    def test_mcq_routes_to_math_and_strips_options(self):
        self.assertEqual(classify_task_heuristic(self.Q), ("math", True))
        self.assertEqual(calculate("What is the value of 8P3?\n(a) 336\n(b) 56").value, "336")
        self.assertEqual(calculate("Find 7C3").value, "35")
        self.assertEqual(calculate("What is 5!?").value, "120")

    def test_program_vote_on_options(self):
        verifier = ProgramVerifier(None, num_programs=3)
        progs = iter(["def solution():\n    return 60\n"] * 2 + ["def solution():\n    return 99\n"])

        async def fake_write(*_a, **_k):
            return next(progs)

        verifier._write_program = fake_write
        draft = "There are 120 arrangements. Final answer: (a) 120"
        samples = ["5!/2! = 60, so (b).", "Answer: B", "It is 120 (a)."]
        old = settings.program_vote_weighting
        settings.program_vote_weighting = "sample_primary"
        try:
            out, _, judgment = asyncio.run(verifier.verify(self.Q, "plan", [draft, *samples], draft))
        finally:
            settings.program_vote_weighting = old
        # b: 2 samples + 2 programs*0.5 = 3 > a: 2 samples; program 99 matches no option
        self.assertTrue(out.endswith("Final answer: (b) 60"))
        self.assertFalse(judgment.is_consistent)

    def _verify_styles(self, by_style: dict[str, str], draft: str, query: str):
        verifier = ProgramVerifier(None, num_programs=5)

        async def fake_write(_q, _plan, seed, style="basic"):
            return by_style[style]

        async def no_repair(_q, code, _err, seed):
            return code

        verifier._write_program, verifier._repair = fake_write, no_repair
        old = settings.program_vote_weighting
        settings.program_vote_weighting = "sample_primary"
        try:
            return asyncio.run(verifier.verify(query, "plan", [draft], draft))
        finally:
            settings.program_vote_weighting = old

    def test_one_kind_of_program_cannot_overturn_draft(self):
        # Seen live (m12): programs of one kind all misread a × b as a dot product
        q = "Find the magnitude of a × b for a = 2i + j - k and b = i - j + 2k."
        draft = r"So |a × b| = \boxed{\sqrt{35}}"
        out, _, judgment = self._verify_styles(
            {"advanced": "def solution():\n    return 1\n",
             "check": "def solution():\n    return undefined_name\n"}, draft, q)
        self.assertEqual(out, draft)  # 3 programs * 0.5 = 1.5 > 1 draft vote, but one method only
        self.assertTrue(judgment.is_consistent)

    def test_two_independent_methods_can_overturn_draft(self):
        # Seen live (h07): every sample was cut off, only the programs got 486
        q = "If A = [[2, 1], [1, 2]], what is the sum of all entries of A^5?"
        draft = r"So the sum is \boxed{585}"
        out, _, judgment = self._verify_styles(
            {"advanced": "def solution():\n    return 486\n",
             "check": "def solution():\n    return 486\n"}, draft, q)
        self.assertTrue(out.endswith("Final answer: 486"))
        self.assertFalse(judgment.is_consistent)

    def test_exact_mcq_override(self):
        q = "What is the value of 8P3?\n(a) 336\n(b) 56\n(c) 512\n(d) 24"
        out, _, judgment = asyncio.run(ProgramVerifier(None).verify(
            q, "", ["8P3 = 56, (b)"], "8P3 = 56, (b)", exact=calculate(q)
        ))
        self.assertTrue(out.endswith("Final answer: (a) 336"))
        self.assertFalse(judgment.is_consistent)


class _FakeJSONClient:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.calls = 0

    async def generate_json(self, **_kwargs):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(text=self.reply)


class TaskRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_rules_skip_llm(self):
        client = _FakeJSONClient('{"type": "factual"}')
        self.assertEqual(await TaskRouter(client).classify("What is 15% of 240?"), "math")
        self.assertEqual(client.calls, 0)

    async def test_llm_decides_ambiguous(self):
        client = _FakeJSONClient('{"type": "factual"}')
        self.assertEqual(await TaskRouter(client).classify("What happened on July 20, 1969?"), "factual")
        self.assertEqual(client.calls, 1)

    async def test_past_year_overrules_time_sensitive(self):
        client = _FakeJSONClient('{"type": "time_sensitive"}')
        self.assertEqual(await TaskRouter(client).classify("What happened on July 20, 1969?"), "factual")

    async def test_llm_failure_falls_back_and_is_recorded(self):
        client = _FakeJSONClient(RuntimeError("boom"))
        with request_trace():
            label = await TaskRouter(client).classify("What happened on July 20, 1969?")
            recorded = degradations()
        self.assertEqual(label, "factual")
        self.assertTrue(any("task_router" in d for d in recorded))


class _SampleClient:
    async def generate(self, **_kwargs):
        return SimpleNamespace(text="Rishi Sunak.")


class _AgreeingNLI:
    is_loaded = True

    async def score_pairs(self, pairs):
        return [{"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05} for _ in pairs]


class FreshnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_time_sensitive_never_fast_path(self):
        from app.llm_clients.ollama_client import OllamaClient

        client = OllamaClient(model="qwen2.5:3b")
        client.generate = _SampleClient().generate  # every sample agrees: H_sem = 0
        triage = SemanticEntropyTriage(client, _AgreeingNLI(), num_samples=4)
        result, estimate = await triage.triage(
            "Who is the current Prime Minister of the UK?", "Rishi Sunak.",
            task_type="time_sensitive",
        )
        self.assertLess(estimate.entropy, 0.01)
        self.assertEqual(result.decision, TriageDecision.KNOWLEDGE_UNCERTAIN)


class ExactVerificationTests(unittest.TestCase):
    def test_exact_overrides_wrong_draft(self):
        exact = calculate("What is 15% of 240?")
        out, traces, judgment = asyncio.run(ProgramVerifier(None).verify(
            "What is 15% of 240?", "", ["It is 34."], "It is 34.", exact=exact
        ))
        self.assertEqual(extract_final_number(out), "36")
        self.assertFalse(judgment.is_consistent)
        self.assertIn("[Calculator]", traces[0])

    def test_exact_keeps_matching_draft(self):
        exact = calculate("What is 1000000 divided by 7?")
        draft = "1000000 / 7 is about 142,857.14."
        out, _, judgment = asyncio.run(ProgramVerifier(None).verify(
            "What is 1000000 divided by 7?", "", [draft], draft, exact=exact
        ))
        self.assertEqual(out, draft)
        self.assertTrue(judgment.is_consistent)


if __name__ == "__main__":
    unittest.main()
