"""Uncertainty methods on hand-worked examples."""

import math
import unittest

from uncertainty.methods import consistency, graph, semantic_entropy as se, token_level
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
        self.assertAlmostEqual(se.se_discrete(v), math.log(2))
        self.assertAlmostEqual(se.se_normalized(v), math.log(2) / math.log(4))
        self.assertEqual(se.num_clusters(v), 2)

    def test_all_agree_is_zero(self):
        self.assertEqual(se.se_discrete(keyed(["7"] * 5)), 0.0)

    def test_pipeline_variant_matches_layer1(self):
        # Layer 1: draft + samples [a, a, a, b, b], scaled by ln 5
        v = keyed(["a", "a", "a", "b", "b"])
        h = -(0.6 * math.log(0.6) + 0.4 * math.log(0.4))
        self.assertAlmostEqual(se.se_pipeline(v), h / math.log(5))

    def test_weighted_uses_likelihoods(self):
        # the two 'b' samples are far more likely → weighted entropy below discrete
        v = keyed(["a", "a", "a", "b", "b"], [[-0.1], [-3.0], [-3.0], [-0.05], [-0.05]])
        self.assertLess(se.se_weighted(v), se.se_discrete(v))

    def test_unreadable_answers_are_singletons(self):
        v = keyed(["a", None, None, "a", "a"])
        self.assertEqual(se.num_clusters(v), 3)

    def test_free_text_needs_nli(self):
        v = View(kind="short", question="q", greedy={"text": "x"}, samples=[{"text": "x"}, {"text": "y"}])
        self.assertIsNone(se.se_discrete(v))

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
        self.assertAlmostEqual(graph.eigv(same), 1.0)
        self.assertAlmostEqual(graph.eigv(diff), 4.0)
        self.assertAlmostEqual(graph.degree(same), 0.0)
        self.assertAlmostEqual(graph.degree(diff), 0.75)
        self.assertGreaterEqual(graph.eccentricity(diff), graph.eccentricity(same))


class TokenAndConsistencyTest(unittest.TestCase):
    def test_token_entropy(self):
        v = keyed(["a", "a", "a"])
        v.greedy["top_logprobs"] = [[["x", math.log(0.5)], ["y", math.log(0.5)]]]
        self.assertAlmostEqual(token_level.token_entropy_mean(v), math.log(2))
        self.assertAlmostEqual(token_level.nll_norm(v), 0.1)

    def test_predictive_entropy(self):
        v = keyed(["a", "a", "a"], [[-0.1], [-1.0, -3.0], [-2.0]])
        self.assertAlmostEqual(token_level.predictive_entropy(v), (2.0 + 2.0) / 2)

    def test_disagreement_with_greedy(self):
        self.assertAlmostEqual(consistency.disagree_greedy(keyed(["a", "a", "a", "b", "b"])), 0.5)

    def test_lexical_similarity(self):
        self.assertAlmostEqual(consistency.lexical_sim(keyed(["x", "same words", "same words"], kind="number")), 0.0)

    def test_subset(self):
        v = keyed(["a", "a", "b", "b", "c"])
        self.assertEqual(v.subset(2).m, 2)
        self.assertEqual(se.se_discrete(v.subset(2)), math.log(2))


if __name__ == "__main__":
    unittest.main()
