"""Uncertainty methods on hand-worked examples."""

import math
import unittest

from uncertainty import methods as M
from uncertainty.methods.nli import FakeNLI, nli_matrix
from uncertainty.methods.view import View


def keyed(keys, logprobs=None, kind="number"):
    """A View whose answers have the given read keys (greedy first)."""
    lps = logprobs or [[-0.1]] * len(keys)
    return View(kind=kind, question="q", greedy={"text": keys[0], "logprobs": lps[0]},
                samples=[{"text": k, "logprobs": lp} for k, lp in zip(keys[1:], lps[1:])], keys=list(keys))


class SemanticEntropyTest(unittest.TestCase):
    def test_discrete(self):
        v = keyed(["a", "a", "a", "b", "b"])  # samples a, a, b, b
        self.assertAlmostEqual(M.se_discrete(v), math.log(2))
        self.assertAlmostEqual(M.se_normalized(v), math.log(2) / math.log(4))
        self.assertEqual(M.num_clusters(v), 2)

    def test_all_agree_is_zero(self):
        self.assertEqual(M.se_discrete(keyed(["7"] * 5)), 0.0)

    def test_pipeline_variant_matches_layer1(self):
        # Layer 1: draft + samples [a, a, a, b, b], scaled by ln 5
        v = keyed(["a", "a", "a", "b", "b"])
        h = -(0.6 * math.log(0.6) + 0.4 * math.log(0.4))
        self.assertAlmostEqual(M.se_pipeline(v), h / math.log(5))

    def test_weighted_uses_likelihoods(self):
        # the two 'b' samples are far more likely → weighted entropy below discrete
        v = keyed(["a", "a", "a", "b", "b"], [[-0.1], [-3.0], [-3.0], [-0.05], [-0.05]])
        self.assertLess(M.se_weighted(v), M.se_discrete(v))

    def test_unreadable_answers_are_singletons(self):
        v = keyed(["a", None, None, "a", "a"])
        self.assertEqual(M.num_clusters(v), 3)

    def test_free_text_needs_nli(self):
        v = View(kind="short", question="q", greedy={"text": "x"}, samples=[{"text": "x"}, {"text": "y"}])
        self.assertIsNone(M.se_discrete(v))

    def test_nli_clusters(self):
        texts = ["The answer is Paris.", "The answer is Paris.", "I believe the answer is Paris.",
                 "The answer is Lyon."]
        v = View(kind="short", question="q", greedy={"text": texts[0]}, samples=[{"text": t} for t in texts[1:]],
                 nli=nli_matrix(FakeNLI(), texts))
        self.assertEqual(v.clusters(), [0, 0, 1])


class GraphTest(unittest.TestCase):
    def test_eigv_and_degree(self):
        same = keyed(["a"] * 5)
        diff = keyed(["a", "b", "c", "d", "e"])
        self.assertAlmostEqual(M.eigv(same), 1.0)
        self.assertAlmostEqual(M.eigv(diff), 4.0)
        self.assertAlmostEqual(M.degree(same), 0.0)
        self.assertAlmostEqual(M.degree(diff), 0.75)
        self.assertGreaterEqual(M.eccentricity(diff), M.eccentricity(same))


class TokenAndConsistencyTest(unittest.TestCase):
    def test_token_entropy(self):
        v = keyed(["a", "a", "a"])
        v.greedy["top_logprobs"] = [[["x", math.log(0.5)], ["y", math.log(0.5)]]]
        self.assertAlmostEqual(M.token_entropy_mean(v), math.log(2))
        self.assertAlmostEqual(M.nll_norm(v), 0.1)

    def test_predictive_entropy(self):
        v = keyed(["a", "a", "a"], [[-0.1], [-1.0, -3.0], [-2.0]])
        self.assertAlmostEqual(M.predictive_entropy(v), (2.0 + 2.0) / 2)

    def test_disagreement_with_greedy(self):
        self.assertAlmostEqual(M.disagree_greedy(keyed(["a", "a", "a", "b", "b"])), 0.5)

    def test_lexical_similarity(self):
        self.assertAlmostEqual(M.lexical_sim(keyed(["x", "same words", "same words"], kind="number")), 0.0)

    def test_subset(self):
        v = keyed(["a", "a", "b", "b", "c"])
        self.assertEqual(v.subset(2).m, 2)
        self.assertEqual(M.se_discrete(v.subset(2)), math.log(2))


class MethodFilesTest(unittest.TestCase):
    def test_one_numbered_file_per_method_in_registry_order(self):
        from pathlib import Path

        from uncertainty.methods.registry import METHODS

        self.assertEqual(len(METHODS), 17)
        folder = Path(M.__file__).parent
        for i, m in enumerate(METHODS, start=1):
            self.assertTrue(m.file.startswith(f"{i:02d}_"), (i, m.name, m.file))
            self.assertTrue((folder / f"{m.file}.py").exists(), m.file)
        self.assertEqual([m.file for m in METHODS], list(M.METHOD_FILES))

    def test_every_file_documents_its_formula_and_example(self):
        from importlib import import_module

        for stem in M.METHOD_FILES:
            doc = import_module(f"uncertainty.methods.{stem}").__doc__
            for part in ("Formula", "Symbols", "Worked example", "Reading the score"):
                self.assertIn(part, doc, (stem, part))

    def test_every_method_file_has_its_note(self):
        from pathlib import Path

        folder = Path(M.__file__).parent
        for stem, names in M.METHOD_FILES.items():
            note = (folder / f"{stem}.README.md").read_text(encoding="utf-8")
            self.assertIn("## Formula", note, stem)
            self.assertIn(names[0], note, stem)


if __name__ == "__main__":
    unittest.main()
