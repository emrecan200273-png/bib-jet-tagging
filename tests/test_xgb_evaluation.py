"""Small checks for the working-point convention and clustered intervals."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.evaluation.xgb_evaluation import choose_threshold, evaluate_scores


class EvaluationTests(unittest.TestCase):
    def test_ties_keep_the_whole_group_and_use_minimum_cut(self):
        labels = [0, 0, 0, 0, 1, 1]
        scores = np.array([0.1, 0.2, 0.2, 0.8, 0.3, 0.9])
        threshold = choose_threshold(labels, scores, target_llp_keep=0.5)
        self.assertEqual(threshold, np.nextafter(0.2, np.inf))
        self.assertGreaterEqual(np.mean(scores[:4] < threshold), 0.5)
        self.assertLess(np.mean(scores[:4] < np.nextafter(threshold, -np.inf)), 0.5)

    def test_target_one_keeps_a_score_of_one(self):
        threshold = choose_threshold([0, 0, 1], [0.2, 1.0, 1.0], 1.0)
        self.assertGreater(threshold, 1.0)

    def test_float32_tie_is_kept_after_threshold_conversion(self):
        scores = np.array([0.2, 0.2, 0.9], dtype=np.float32)
        threshold = choose_threshold([0, 0, 1], scores, 0.95)
        frame = pd.DataFrame({"label": [0, 0, 1], "event_group": ["a", "b", "c"]})
        result = evaluate_scores(frame, scores, threshold, bootstrap_repeats=0)
        self.assertEqual(result["llp_keep_fraction"], 1.0)
        self.assertEqual(result["bib_tag_fraction"], 1.0)

    def test_decimal_target_does_not_add_extra_jet(self):
        scores = np.arange(100, dtype=float)
        threshold = choose_threshold(np.zeros(100), scores, 0.07)
        self.assertEqual(np.sum(scores < threshold), 7)

    def test_reject_nonfinite_scores(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            choose_threshold([0, 1], [0.1, np.nan])

    def test_cluster_bootstrap_resamples_events_together(self):
        # Two events each have two perfectly correlated jets. Whole-event
        # resampling allows rates 0, 1/2 or 1; independent-jet resampling differs.
        frame = pd.DataFrame({"label": [0, 0, 0, 0, 1, 1],
                              "event_group": ["a", "a", "b", "b", "c", "c"]})
        scores = [0.1, 0.1, 0.9, 0.9, 0.9, 0.9]
        result = evaluate_scores(frame, scores, 0.5, bootstrap_repeats=500, seed=42)
        self.assertEqual(result["n_events"], 3)
        self.assertEqual(result["llp_keep_fraction"], 0.5)
        self.assertEqual(result["llp_keep_fraction_ci95"], [0.0, 1.0])
        self.assertIsNone(result["bib_tag_fraction_ci95"])
        self.assertEqual(result["confusion"]["LLP_tagged_BIB"], 2)
        json.dumps(result, allow_nan=False)
        self.assertEqual(list(frame.columns), ["label", "event_group"])

    def test_boundary_rates_keep_counts_and_omit_misleading_intervals(self):
        frame = pd.DataFrame({"label": [0, 0, 1, 1],
                              "event_group": ["a", "b", "c", "d"]})
        for scores, bib_rate, llp_rate in [([.1, .1, .1, .1], 0., 1.),
                                           ([.9, .9, .9, .9], 1., 0.)]:
            with self.subTest(bib_rate=bib_rate):
                result = evaluate_scores(frame, scores, .5, bootstrap_repeats=100)
                self.assertEqual(result["bib_tag_fraction"], bib_rate)
                self.assertEqual(result["llp_keep_fraction"], llp_rate)
                self.assertEqual(result["class_counts"], {"BIB": 2, "LLP": 2})
                self.assertIsNone(result["bib_tag_fraction_ci95"])
                self.assertIsNone(result["llp_keep_fraction_ci95"])
                self.assertIn("cannot bound unseen outcomes", result["bootstrap"]["boundary_policy"])
                json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
