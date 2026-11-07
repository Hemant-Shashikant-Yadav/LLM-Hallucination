"""
Offline tests for general-question handling (smoke list in doc/06_roadmap.md):
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

    def test_today_in_a_word_problem_is_not_current_events(self):
        q = ("Andy wants to lose 30 lbs by his birthday, July 19th. Today is December 31st. "
             "If he must burn 3500 calories to lose a pound, how many calories a day must he burn?")
        self.assertEqual(classify_task_heuristic(q), ("math", True))
        self.assertEqual(classify_task_heuristic("Who is the current Prime Minister of the UK?"),
                         ("time_sensitive", True))

    def test_spelled_out_word_problem_is_math(self):
        q = ("Greg has an alarm set to ring three times a day. The first time it went off today, "
             "it rang four times. The second time it rang for three times as long as the first "
             "time. The third time, it rang for half as long as the second time. How many times "
             "did the alarm ring in all?")
        self.assertEqual(classify_task_heuristic(q), ("math", True))

    def test_today_in_a_quoted_premise_is_not_current_events(self):
        q = ('Given the major premise "Either I do well on this test or I am mad", which minor '
             'premise leads to a valid conclusion?\n(a) "I won\'t do well"\n(b) "My birthday is today"')
        self.assertNotEqual(classify_task_heuristic(q)[0], "time_sensitive")
        self.assertEqual(classify_task_heuristic('What is the latest "Star Wars" film?')[0],
                         "time_sensitive")

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
        # Concluding sentences without a marker (GSM8K test items 18, 36, 65 were misread)
        ("Therefore, Claire will eat 7 dozens of eggs in 4 weeks.", "7"),
        ("Therefore, Jim spends 36 hours on TV and reading in 4 weeks.", "36"),
        ("Therefore, Terry spends $75.00 on yogurt over 30 days.", "75"),
        ("Therefore, Jill's annual salary is $35,000 + $22,500 = $57,500.", "57500"),
        ("So, it would take 5 people 48 hours to paint the house.", "48"),
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

    def test_extract_choice_text_options(self):
        lake = {"a": "colder", "b": "deeper", "c": "larger", "d": "smaller"}
        water = {"a": "lakes", "b": "rivers", "c": "oceans", "d": "aquifers"}
        cases = [
            (lake, "Therefore, the correct answer is:\n\n(d) smaller", "d"),
            (lake, "The lake will become smaller. The other options—colder, deeper, "
                   "and larger—are not affected.", "d"),  # never the rejected options
            (lake, "Water use grows.\n\n(d) Smaller.", "d"),
            (water, "The largest source is (c) oceans. However, oceans are salty. "
                    "The largest drinkable source is (d) aquifers.", "d"),  # self-correction
            (water, "The best conclusion is:\n\n(a) lakes\n\nThis is because water "
                    "collects there over a long time.", "a"),
            (water, r"So option (d) is incorrect. Therefore: \boxed{\text{None}}", None),
            (water, r"Groundwater is stored underground. \boxed{\text{(d) aquifers}}", "d"),
            (water, r"\boxed{\text{Aquifers}}", "d"),
        ]
        for opts, text, want in cases:
            with self.subTest(text=text):
                self.assertEqual(extract_choice(text, opts), want)

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

    async def test_given_quantities_overrule_time_sensitive(self):
        client = _FakeJSONClient('{"type": "time_sensitive"}')
        q = ("The length of the year on the planet Mars is exactly 697 days. If Mars has a calendar "
             "with a 12-day week, and year 0 begins on the first day of the week, what is the next "
             "year which will begin on the first day of the week?")
        self.assertEqual(await TaskRouter(client).classify(q), "math")
        self.assertEqual(await TaskRouter(client).classify("Who is the president of France?"),
                         "time_sensitive")

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


class ChoiceVoteTests(unittest.IsolatedAsyncioTestCase):
    """Non-maths multiple choice: the option most answers pick wins; ties keep the draft."""

    Q = ("Which is Earth's largest source of drinkable water?\n"
         "(a) lakes\n(b) rivers\n(c) oceans\n(d) aquifers")

    async def _vote(self, draft: str, samples: list[str], extra: list[str]):
        from app.models.schemas import QueryRequest, TriageResult
        from app.router import DefenseInDepthRouter

        async def sample(*_args, **_kwargs):
            return extra

        router = DefenseInDepthRouter.__new__(DefenseInDepthRouter)
        router._se_triage = SimpleNamespace(sample=sample)
        triage = TriageResult(
            decision=TriageDecision.KNOWLEDGE_UNCERTAIN, semantic_entropy=0.9,
            hallucination_probability=0.5, error_classification=None,
            threshold_used=settings.t_safe, draft_response=draft, latency_ms=0.0,
        )
        estimate = SimpleNamespace(responses=[draft, *samples])
        draft_result = SimpleNamespace(total_tokens=0, latency_ms=0.0)
        return await router._process_choice_vote(
            QueryRequest(prompt=self.Q, include_audit=True), draft, draft_result, triage, 0.0, estimate
        )

    async def test_majority_overturns_draft(self):
        res = await self._vote("The answer is (c) oceans.", ["The answer is (d) aquifers."] * 2,
                               ["The answer is (d) aquifers."] * 4 + ["The answer is (c) oceans."])
        self.assertIn("(d) aquifers", res.answer)
        self.assertFalse(res.pipeline_result.layer3_result.judgment.is_consistent)

    async def test_tie_keeps_draft(self):
        res = await self._vote("The answer is (c) oceans.", ["The answer is (d) aquifers."],
                               ["The answer is (c) oceans.", "The answer is (d) aquifers."])
        self.assertEqual(res.answer, "The answer is (c) oceans.")


class FreshSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_question_search_retries_when_web_returns_nothing(self):
        from unittest import mock

        from app.layer2_nli_rag.search_retriever import SearchRetriever
        from app.models.schemas import SearchResult

        hit = SearchResult(title="BBC", url="https://bbc.example/pm", snippet="X is PM.",
                           relevance_score=0.7)
        replies = [[], [hit]]  # throttled once, then results

        async def ddg(_query, _limit):
            return replies.pop(0)

        retriever = SearchRetriever(backend="duckduckgo")
        with mock.patch.object(retriever, "_search_duckduckgo", ddg), \
             mock.patch("app.layer2_nli_rag.search_retriever.asyncio.sleep", new=mock.AsyncMock()):
            found = await retriever.search("Who is the PM?", retries=2)
        self.assertEqual([r.url for r in found], ["https://bbc.example/pm"])

    async def test_results_that_do_not_answer_keep_draft_with_note(self):
        from app.layer2_nli_rag.reviser import EvidenceReviser
        from app.models.schemas import SearchResult

        class _NotFound:
            async def generate(self, **_kwargs):
                return SimpleNamespace(text="NOT_FOUND")

        ev = [SearchResult(title="t", url="u", snippet="unrelated", relevance_score=0.7)]
        with request_trace():
            out = await EvidenceReviser(_NotFound())._answer_from_evidence("Who is the PM?", "X.", ev)
            recorded = degradations()
        self.assertTrue(out.startswith("X."))
        self.assertIn("may be out of date", out)
        self.assertTrue(any("freshness" in d for d in recorded))


class YesNoTests(unittest.IsolatedAsyncioTestCase):
    def test_reader_and_question_shape(self):
        from app.answer_utils import extract_yes_no, is_yes_no_question

        self.assertTrue(is_yes_no_question("Could a llama birth twice during the War in Vietnam?"))
        self.assertFalse(is_yes_no_question("Can you explain how vaccines work?"))
        self.assertFalse(is_yes_no_question("Who wrote Hamlet?"))
        self.assertFalse(is_yes_no_question("Is it (a) or (b)?\n(a) x\n(b) y"))
        for text, want in [("Yes, he did.", "yes"), ("**No**, it cannot.", "no"),
                           ("So the answer is no.", "no"), ("He died in 322 BC.", None)]:
            with self.subTest(text=text):
                self.assertEqual(extract_yes_no(text), want)

    def test_layer1_clusters_by_verdict(self):
        draft = "No. Llama gestation is about 11 months, and the war lasted years longer."
        samples = ["No, because of gestation length.", "Yes, it could.", "No."]
        self.assertTrue(SemanticEntropyTriage._yes_no_readable("Could a llama birth twice?", [draft, *samples]))
        self.assertFalse(SemanticEntropyTriage._yes_no_readable(
            "Could a llama birth twice?", ["Llamas gestate for 11 months.", *samples]))

    async def test_vote_over_verdicts(self):
        from app.answer_utils import extract_yes_no
        from app.models.schemas import QueryRequest, TriageResult
        from app.router import DefenseInDepthRouter

        async def sample(*_args, **_kwargs):
            return ["No, it could not."] * 4

        router = DefenseInDepthRouter.__new__(DefenseInDepthRouter)
        router._se_triage = SimpleNamespace(sample=sample)
        draft = "Yes, a llama could give birth twice."
        triage = TriageResult(
            decision=TriageDecision.KNOWLEDGE_UNCERTAIN, semantic_entropy=0.9,
            hallucination_probability=0.5, error_classification=None,
            threshold_used=settings.t_safe, draft_response=draft, latency_ms=0.0,
        )
        res = await router._process_choice_vote(
            QueryRequest(prompt="Could a llama birth twice during the War in Vietnam?", include_audit=True),
            draft, SimpleNamespace(total_tokens=0, latency_ms=0.0), triage, 0.0,
            SimpleNamespace(responses=[draft, "No.", "Yes."]), key=extract_yes_no,
        )
        self.assertEqual(extract_yes_no(res.answer), "no")


class FullSetScoringTests(unittest.TestCase):
    """MATH keys are LaTeX; JEEBench has multi-correct MCQs."""

    def test_latex_keys(self):
        from app.answer_utils import answers_equivalent

        bs = chr(92)
        cases = [(f"{bs}boxed{{3{bs}sqrt{{13}}}}", f"3{bs}sqrt{{13}}", True),
                 ("The answer is 10.8167", f"3{bs}sqrt{{13}}", True),
                 (f"{bs}boxed{{p - q}}", "p - q", True),
                 (f"{bs}boxed{{{bs}left( 3, {bs}frac{{{bs}pi}}{{2}} {bs}right)}}", f"{bs}left( 3, {bs}frac{{{bs}pi}}{{2}} {bs}right)", True),
                 (f"{bs}boxed{{{bs}text{{Evelyn}}}}", f"{bs}text{{Evelyn}}", True),
                 (f"{bs}boxed{{-{bs}frac{{1}}{{8}}}}", f"-{bs}frac{{1}}{{8}}", True),
                 (f"{bs}boxed{{{bs}frac{{1}}{{8}}}}", f"-{bs}frac{{1}}{{8}}", False)]
        for answer, key, want in cases:
            with self.subTest(answer=answer, key=key):
                from app.answer_utils import final_expression
                self.assertEqual(answers_equivalent(final_expression(answer) or answer, key, symbolic=True), want)

    def test_multi_correct_letters(self):
        from evaluation.hard_eval import is_correct

        bs = chr(92)
        row = {"kind": "multi", "key": "abd", "question": ""}
        self.assertTrue(is_correct(row, f"So {bs}boxed{{ABD}}"))
        self.assertTrue(is_correct(row, "Final answer: (A), (B) and (D)"))
        self.assertFalse(is_correct(row, "Final answer: (A) and (B)"))
