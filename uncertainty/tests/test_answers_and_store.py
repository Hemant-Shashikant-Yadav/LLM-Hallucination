"""Answer reading, correctness rules, log-prob reading, and crash-safe storage."""

import json
import tempfile
import unittest
from pathlib import Path

from uncertainty.answers import read_answer, rule_correct
from uncertainty.generate.backends import Generation
from uncertainty.generate.runner import binary_prob, read_verbal
from uncertainty.label.correctness import alias_match
from uncertainty.store import append_jsonl, finalize, read_jsonl, repair_partial

MCQ = "Which is a gas?\n(a) Ice\n(b) Steam\n(c) Rock\n(d) Iron"


class AnswersTest(unittest.TestCase):
    def test_read_answer_by_kind(self):
        self.assertEqual(read_answer("number", "Step 1: 16 - 3 - 4 = 9.\nFinal answer: 18", "q"), "18")
        self.assertEqual(read_answer("choice", "Steam is water vapour.\nAnswer: B", MCQ), "b")
        self.assertEqual(read_answer("yesno", "No, at first glance... but yes.\nAnswer: yes", "Is it?"), "yes")
        self.assertIsNone(read_answer("short", "Paris.", "q"))

    def test_rule_correct(self):
        self.assertTrue(rule_correct("number", "Final answer: 18", "q", "18"))
        self.assertFalse(rule_correct("number", "Final answer: 19", "q", "18"))
        self.assertTrue(rule_correct("expr", "Final answer: \\frac{1}{2}", "q", "\\frac{1}{2}"))
        self.assertTrue(rule_correct("choice", "Answer: (b)", MCQ, "b"))
        self.assertIsNone(rule_correct("yesno", "It depends on the context.", "Is it?", "yes"))

    def test_alias_match(self):
        self.assertTrue(alias_match("I think it was The Beatles.", ["Beatles"]))
        self.assertFalse(alias_match("Beatlemania", ["Beatles"]))
        self.assertIsNone(alias_match("anything", []))

    def test_binary_prob_from_logprobs_and_text(self):
        gen = Generation("A", top_logprobs=[[[" A", -0.2], ["B", -1.8], ["(", -5.0]]])
        p = binary_prob(gen, ("A",), ("B",))
        self.assertEqual(p["source"], "logprobs")
        self.assertAlmostEqual(p["p"], 0.8320, places=3)
        self.assertEqual(binary_prob(Generation("No."), ("yes",), ("no",))["p"], 0.0)
        self.assertIsNone(binary_prob(Generation("maybe"), ("yes",), ("no",))["p"])

    def test_read_verbal(self):
        self.assertEqual(read_verbal("85"), 85.0)
        self.assertEqual(read_verbal("Confidence: 100%"), 100.0)
        self.assertIsNone(read_verbal("high"))


class StoreTest(unittest.TestCase):
    def test_crash_mid_write_and_finalize(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial, final = Path(tmp) / "s.partial.jsonl", Path(tmp) / "s.jsonl.gz"
            append_jsonl(partial, {"id": "b", "x": 1})
            append_jsonl(partial, {"id": "a", "x": 2})
            with partial.open("a", encoding="utf-8") as fh:
                fh.write('{"id": "c", "x"')  # cut off
            self.assertEqual(len(read_jsonl(partial)), 2)
            repair_partial(partial)
            append_jsonl(partial, {"id": "c", "x": 3})
            finalize(partial, final, ["a", "b", "c"])
            self.assertFalse(partial.exists())
            self.assertEqual([r["id"] for r in read_jsonl(final)], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
