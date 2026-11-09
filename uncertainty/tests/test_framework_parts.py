"""Objective 2 building blocks, gates, the roster, and the frozen question sets."""

import json
import unittest

import numpy as np
from sklearn.metrics import f1_score

from uncertainty.framework.classifier import _boot_macro_f1
from uncertainty.framework.taxonomy import intervention_label, recommended_action
from uncertainty.orchestrate import gates
from uncertainty.roster import focus_models, judge_for, pick_roster, second_judge_for, spec
from uncertainty.sets.load import SETS, load_all, spread
from uncertainty.settings import settings


class ClassifierPartsTest(unittest.TestCase):
    def test_vectorised_macro_f1_matches_sklearn(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 4, 300)
        p = np.where(rng.random(300) < 0.6, y, rng.integers(0, 4, 300))
        idx = rng.integers(0, 300, (40, 300))
        fast = _boot_macro_f1(y, p, idx, 4)
        slow = [f1_score(y[i], p[i], labels=sorted(set(y[i])), average="macro", zero_division=0) for i in idx]
        np.testing.assert_allclose(fast, slow)

    def test_intervention_labels(self):
        self.assertEqual(intervention_label(True, False), "knowledge")
        self.assertEqual(intervention_label(False, True), "reasoning")
        self.assertEqual(intervention_label(True, None), "knowledge")  # only one intervention: it decides
        self.assertEqual(intervention_label(True, True), "mixed")
        self.assertEqual(intervention_label(False, False), "unresolved")
        self.assertEqual(intervention_label(None, None), "not_applicable")

    def test_actions(self):
        self.assertEqual(recommended_action("low", "knowledge", None, False), "accept")
        self.assertEqual(recommended_action("low", "knowledge", None, True), "retrieve_evidence")
        self.assertEqual(recommended_action("high", "aleatoric", "false_premise", False), "correct_the_premise")


class GateTest(unittest.TestCase):
    def test_judge_gate_warns_then_fails(self):
        self.assertIsNone(gates.check_judge({"judge_calls": 40, "judge_failures": 0, "judge_failure_rate": 0.0}, "x"))
        # one failure in a small batch is a warning, not a lost model
        self.assertIsNotNone(gates.check_judge({"judge_calls": 40, "judge_failures": 1, "judge_failure_rate": 0.025}, "x"))
        with self.assertRaises(gates.GateFailure):
            gates.check_judge({"judge_calls": 40, "judge_failures": 20, "judge_failure_rate": 0.5}, "x")

    def test_readability_gate(self):
        rec = lambda text: {"kind": "number", "question": "q", "greedy": {"text": text},
                            "samples": [{"text": text, "done_reason": "stop"}] * 4}
        gates.check_generation([rec("Final answer: 7")] * 5, "ok")
        with self.assertRaises(gates.GateFailure):
            gates.check_generation([rec("I am not sure how to do this.")] * 5, "bad")


class RosterTest(unittest.TestCase):
    def test_roster_by_memory(self):
        self.assertEqual(len(pick_roster(16)), 9)
        self.assertEqual(len(pick_roster(24)), 12)

    def test_judges_never_same_family(self):
        for mem in (16, 24):
            for s in pick_roster(24):
                fams = {s.family, spec(judge_for(s.tag, mem)).family, spec(second_judge_for(s.tag, mem)).family}
                self.assertEqual(len(fams), 3, s.tag)

    def test_focus(self):
        tags = [s.tag for s in pick_roster(16)]
        self.assertEqual(focus_models(tags), ["qwen2.5:3b", "llama3.1:8b", "gemma3:4b"])
        self.assertEqual(focus_models(tags, "qwen3:4b-instruct-2507-q4_K_M")[-1], "qwen3:4b-instruct-2507-q4_K_M")


class QuestionSetsTest(unittest.TestCase):
    def test_frozen_sets_match_the_plan(self):
        manifest = json.loads((settings.sets_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(list(manifest), list(SETS))
        full = load_all("extended")
        self.assertEqual(sum(len(v) for v in full.values()), 3367)
        self.assertEqual(sum(len(v) for v in load_all("core").values()), 1410)
        ids = [it.id for its in full.values() for it in its]
        self.assertEqual(len(ids), len(set(ids)))

    def test_dry_run_sample_keeps_every_stratum(self):
        fresh = spread(load_all("extended", ["freshqa"])["freshqa"], 24)
        self.assertEqual({it.gold_type for it in fresh}, {"false_premise", "temporal", None})


if __name__ == "__main__":
    unittest.main()
