"""Metrics checked against scikit-learn and hand-worked cases."""

import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from uncertainty import metrics


class MetricsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.wrong = rng.integers(0, 2, 200)
        self.unc = self.wrong * 0.8 + rng.normal(0, 0.6, 200)

    def test_auroc_and_auprc_match_sklearn(self):
        self.assertAlmostEqual(metrics.auroc(self.unc, self.wrong), roc_auc_score(self.wrong, self.unc))
        self.assertAlmostEqual(metrics.auprc(self.unc, self.wrong), average_precision_score(self.wrong, self.unc))

    def test_auroc_needs_both_classes(self):
        self.assertIsNone(metrics.auroc([0.1, 0.2], [1, 1]))

    def test_nan_scores_are_dropped(self):
        unc = np.array([0.1, np.nan, 0.9, 0.2])
        self.assertEqual(metrics.auroc(unc, [0, 1, 1, 0]), 1.0)

    def test_rejection_curve_by_hand(self):
        # most certain first: right, right, wrong → accuracies 1, 1, 2/3
        acc = metrics.accuracy_curve([0.1, 0.2, 0.9], [1, 1, 0])
        np.testing.assert_allclose(acc, [1, 1, 2 / 3])
        self.assertAlmostEqual(metrics.aurac([0.1, 0.2, 0.9], [0, 0, 1]), (1 + 1 + 2 / 3) / 3)

    def test_ties_are_averaged(self):
        # all tied: every retention level has the overall accuracy in expectation
        np.testing.assert_allclose(metrics.accuracy_curve([0.5] * 4, [1, 0, 1, 0]), [0.5, 0.5, 0.5, 0.5])

    def test_prr_oracle_and_random(self):
        self.assertAlmostEqual(metrics.prr(self.wrong.astype(float), self.wrong), 1.0)
        self.assertAlmostEqual(metrics.prr(np.zeros(200), self.wrong), 0.0)

    def test_ece_and_brier(self):
        self.assertAlmostEqual(metrics.ece([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0]), 0.0)
        self.assertAlmostEqual(metrics.ece([0.8] * 10, [1] * 8 + [0] * 2), 0.0, places=6)
        self.assertAlmostEqual(metrics.brier([0.5, 0.5], [1, 0]), 0.25)

    def test_bootstrap_auroc_brackets_the_estimate(self):
        lo, hi = metrics.bootstrap_auroc(self.unc, self.wrong, n_boot=300)
        point = metrics.auroc(self.unc, self.wrong)
        self.assertLess(lo, point)
        self.assertGreater(hi, point)

    def test_delong(self):
        same = metrics.delong(self.unc, self.unc, self.wrong)
        self.assertAlmostEqual(same["p"], 1.0)
        noise = np.random.default_rng(2).normal(size=200)
        diff = metrics.delong(self.unc, noise, self.wrong)
        self.assertAlmostEqual(diff["auc_a"], roc_auc_score(self.wrong, self.unc))
        self.assertLess(diff["p"], 0.01)

    def test_holm(self):
        self.assertEqual(metrics.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()
