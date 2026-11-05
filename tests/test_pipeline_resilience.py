"""
Offline regression tests for the Stage 1 → 2 → 3 failure modes.

No Ollama, GPU or network needed: the Ollama SDK client and the NLI model are
replaced with fakes. Run with:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from types import SimpleNamespace

from ollama import ResponseError

from app.config import settings
from app.layer1_triage.semantic_entropy import SemanticEntropyTriage
from app.layer2_nli_rag.proposition_parser import PropositionParser
from app.layer3_halluclean.program_verifier import ProgramVerifier
from app.llm_clients.ollama_client import ModelUnavailableError, OllamaClient
from app.models.schemas import TriageDecision
from app.telemetry import current_trace_id, degradations, record_degradation, request_trace


def _response(text: str = "ok", done_reason: str = "stop", prompt_tokens: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        response=text, model="fake", prompt_eval_count=prompt_tokens, eval_count=5,
        total_duration=1, eval_duration=1, done_reason=done_reason,
    )


class FakeSDK:
    """Stands in for ollama.AsyncClient: scripted results, records calls."""

    def __init__(self, script: list, models: tuple[str, ...] = ("llama2:7b",)) -> None:
        self.script = list(script)
        self.models = models
        self.calls: list[dict] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model=m) for m in self.models])

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            item = self.script.pop(0) if self.script else _response()
            if isinstance(item, Exception):
                raise item
            return item
        finally:
            self.in_flight -= 1


def _client(sdk: FakeSDK, model: str = "llama2:7b") -> OllamaClient:
    client = OllamaClient(model=model)
    client._client = sdk
    return client


class FakeNLI:
    is_loaded = True

    async def score_pairs(self, pairs):
        # Everything entails everything → a single meaning cluster → H_sem = 0
        return [{"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05} for _ in pairs]


class SemanticEntropyStage1Tests(unittest.IsolatedAsyncioTestCase):
    async def test_all_samples_failing_does_not_take_fast_path(self):
        """Root cause #1: failed samples left H_sem = 0 → SAFE → pipeline stopped at Layer 1."""
        sdk = FakeSDK([TimeoutError("read timeout")] * 4)
        triage = SemanticEntropyTriage(_client(sdk), FakeNLI(), num_samples=4)
        with request_trace():
            result, estimate = await triage.triage("Why do veins appear blue?", "Because ...")
            recorded = degradations()

        self.assertEqual(estimate.entropy, 0.0)  # the misleading signal is still computed
        self.assertEqual(result.decision, TriageDecision.KNOWLEDGE_UNCERTAIN)
        self.assertTrue(any("H_sem unreliable" in d for d in recorded), recorded)

    async def test_healthy_samples_still_take_fast_path(self):
        sdk = FakeSDK([_response("Because of light scattering.")] * 4)
        triage = SemanticEntropyTriage(_client(sdk), FakeNLI(), num_samples=4)
        with request_trace():
            result, _ = await triage.triage("Why do veins appear blue?", "Because of light.")
            self.assertEqual(degradations(), [])
        self.assertEqual(result.decision, TriageDecision.SAFE)


class OllamaClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_model_fails_validation(self):
        """Root cause #2: a missing judge model passed startup and failed every L2/L3 call."""
        client = _client(FakeSDK([], models=("llama2:7b",)), model="llama3.1:8b")
        with self.assertRaises(ModelUnavailableError):
            await client.validate_models()

    async def test_tag_normalisation(self):
        client = _client(FakeSDK([], models=("llama2:latest",)), model="llama2")
        await client.validate_models()

    async def test_transient_500_is_retried(self):
        crash = ResponseError("llama-server process has terminated: CUDA error", 500)
        sdk = FakeSDK([crash, _response("recovered")])
        old = settings.ollama_max_retries
        settings.ollama_max_retries = 2
        try:
            result = await _client(sdk).generate("hi")
        finally:
            settings.ollama_max_retries = old
        self.assertEqual(result.text, "recovered")
        self.assertEqual(len(sdk.calls), 2)

    async def test_404_is_not_retried(self):
        sdk = FakeSDK([ResponseError("model not found", 404)])
        with self.assertRaises(ModelUnavailableError):
            await _client(sdk).generate("hi")
        self.assertEqual(len(sdk.calls), 1)

    async def test_num_ctx_and_keep_alive_are_sent(self):
        sdk = FakeSDK([_response()])
        await _client(sdk).generate("hi", max_tokens=64)
        self.assertEqual(sdk.calls[0]["options"]["num_ctx"], settings.ollama_num_ctx)
        self.assertEqual(sdk.calls[0]["options"]["num_predict"], 64)
        self.assertEqual(sdk.calls[0]["keep_alive"], settings.ollama_keep_alive)

    async def test_concurrency_gate_limits_in_flight_requests(self):
        sdk = FakeSDK([_response()] * 6)
        client = _client(sdk)
        await asyncio.gather(*(client.generate("hi") for _ in range(6)))
        self.assertLessEqual(sdk.max_in_flight, settings.ollama_max_concurrency)

    async def test_truncation_is_flagged(self):
        sdk = FakeSDK([_response(done_reason="length")])
        result = await _client(sdk).generate("hi")
        self.assertTrue(result.truncated)


class PropositionParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = PropositionParser(ollama_client=None)  # parsing only

    def test_object_shapes(self):
        src = "Paris is in France. It is the capital."
        for raw in (
            '{"propositions": ["Paris is in France.", "Paris is the capital of France."]}',
            '{"claims": ["Paris is in France.", "Paris is the capital of France."]}',
            '{"AP1": "Paris is in France.", "AP2": "Paris is the capital of France."}',
            '["Paris is in France.", "Paris is the capital of France."]',
        ):
            with request_trace():
                props = self.parser._parse_json_response(raw, src)
                self.assertEqual(degradations(), [], raw)
            self.assertEqual(len(props), 2, raw)


class ProgramVerifierTests(unittest.TestCase):
    CODE = "def solution():\n    eggs = 16 - 3 - 4\n    return eggs * 2\n"

    def test_runs_on_default_loop(self):
        verifier = ProgramVerifier(ollama_client=None)
        self.assertEqual(asyncio.run(verifier._run(self.CODE)), ("18", "ok"))

    def setUp(self):
        self._weighting = settings.program_vote_weighting
        settings.program_vote_weighting = "program_primary"

    def tearDown(self):
        settings.program_vote_weighting = self._weighting

    def test_sample_primary_majority_wins(self):
        settings.program_vote_weighting = "sample_primary"
        progs = ["def solution():\n    return 7\n"] * 4  # 4 programs * 0.5 = 2.0
        out, _, judgment = self._verify_with(
            progs, "Final answer: 5", ["Final answer: 5", "Final answer: 5"],
            "Tom has 2 and 3 apples. How many?",
        )  # 5 has 3 sample votes > 2.0
        self.assertIn("Final answer: 5", out)
        self.assertTrue(judgment.is_consistent)

    def test_sample_primary_programs_break_sample_split(self):
        settings.program_vote_weighting = "sample_primary"
        progs = ["def solution():\n    return 7\n"] * 2
        out, _, _ = self._verify_with(
            progs, "Final answer: 5", ["Final answer: 7", "Final answer: 7"],
            "Tom has 2 and 3 apples. How many?",
        )  # 7: 2 + 1.0 beats 5: 1
        self.assertTrue(out.endswith("Final answer: 7"))

    def _verify_with(self, programs: list[str], draft: str, samples: list[str], query: str):
        verifier = ProgramVerifier(ollama_client=None, num_programs=len(programs))
        it = iter(programs)

        async def fake_write(*_a, **_k):
            return next(it)

        async def fake_repair(_q, code, _err, seed):
            return code

        verifier._write_program = fake_write
        verifier._repair = fake_repair
        return asyncio.run(verifier.verify(query, "plan", [draft, *samples], draft))

    def test_single_program_cannot_override_draft(self):
        progs = ["def solution():\n    return 7\n"] + ["def solution():\n    return x\n"] * 4
        out, _, judgment = self._verify_with(
            progs, "Final answer: 5", ["Final answer: 6"], "Tom has 2 and 3 apples. How many?"
        )
        self.assertIn("Final answer: 5", out)
        self.assertTrue(judgment.is_consistent)

    def test_two_programs_override_draft(self):
        progs = ["def solution():\n    return 7\n"] * 2 + ["def solution():\n    return x\n"] * 3
        out, _, judgment = self._verify_with(
            progs, "Final answer: 5", [], "Tom has 2 and 3 apples. How many?"
        )
        self.assertTrue(out.endswith("Final answer: 7"))
        self.assertFalse(judgment.is_consistent)

    def test_tie_keeps_draft(self):
        # 72: 3 programs + 0 samples = 3; 144 (draft): 1 program + draft + 1 sample = 3
        progs = ["def solution():\n    return 72\n"] * 3 + ["def solution():\n    return 144\n"]
        out, _, judgment = self._verify_with(
            progs, "Final answer: 144", ["Final answer: 144"], "He has 12 boxes of 12 eggs. How many?"
        )
        self.assertIn("Final answer: 144", out)
        self.assertTrue(judgment.is_consistent)

    def test_negative_program_answer_rejected(self):
        progs = ["def solution():\n    return -17500\n"] * 3
        out, traces, _ = self._verify_with(
            progs, "Final answer: 525", [], "Andy wants to lose 30 lbs in 200 days. How many?"
        )
        self.assertIn("Final answer: 525", out)
        program_traces = [t for t in traces if t.startswith("[Program")]
        self.assertEqual(len(program_traces), 3)
        self.assertTrue(all("implausible negative" in t for t in program_traces))

    @unittest.skipUnless(sys.platform == "win32", "Selector loop limitation is Windows-only")
    def test_runs_on_windows_selector_loop(self):
        """Root cause #4: uvicorn --reload on Windows uses SelectorEventLoop, where
        asyncio.create_subprocess_exec raises NotImplementedError."""
        verifier = ProgramVerifier(ollama_client=None)
        loop = asyncio.SelectorEventLoop()
        try:
            self.assertEqual(loop.run_until_complete(verifier._run(self.CODE)), ("18", "ok"))
        finally:
            loop.close()


class TelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_trace_and_degradations_cross_gather(self):
        async def child(i: int) -> str:
            record_degradation("test", f"child {i}")
            return current_trace_id()

        with request_trace("abc123"):
            ids = await asyncio.gather(*(child(i) for i in range(3)))
            self.assertEqual(ids, ["abc123"] * 3)
            self.assertEqual(len(degradations()), 3)
        self.assertEqual(degradations(), [])


if __name__ == "__main__":
    unittest.main()
