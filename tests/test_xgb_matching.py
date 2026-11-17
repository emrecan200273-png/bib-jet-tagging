"""Kinematic balancing must preserve event splits and use training-only support."""
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.evaluation.xgb_matching import match_kinematics


class KinematicMatchingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {"label": label, "sample": "LLP" if label == 0 else "unpaired",
             "split": split, "event_group": f"{split}:{label}:{bin_index}:{jet // 2}",
             "source_file": f"sample{label}.root", "jet_index": jet % 2,
             "row_index": jet // 2, "jet_pt_GeV": pt, "jet_abs_eta": eta,
             "jet_time_ATLAS": jet * (1 + label), "model_score": label * .9}
            for split in ["train", "validation", "test"] for label in [0, 1]
            for bin_index, (pt, eta) in enumerate([(40., .1), (400., .8)])
            for jet in range((500 if split == "train" else 250) + label * 70)
        ])

    def test_balances_every_bin_and_preserves_splits(self):
        original = self.frame.groupby("event_group")["split"].first()
        matched, audit = match_kinematics(self.frame)
        counts = matched.groupby(["split", "pt_bin", "eta_bin", "label"]).size().unstack("label")
        self.assertTrue(counts[0].eq(counts[1]).all())
        self.assertTrue(matched.groupby("event_group")["split"].nunique().eq(1).all())
        self.assertTrue(matched["split"].eq(matched["event_group"].map(original)).all())
        self.assertFalse(matched.duplicated(["event_group", "jet_index"]).any())
        self.assertEqual(audit["pairs"], len(matched) // 2)
        self.assertEqual(audit["support_defined_on"], "train only")

    def test_reproducible_when_rows_are_reordered(self):
        first, _ = match_kinematics(self.frame, seed=42)
        reordered, _ = match_kinematics(self.frame.sample(frac=1, random_state=17), seed=42)
        pd.testing.assert_frame_equal(first, reordered)
        other, _ = match_kinematics(self.frame, seed=43)
        self.assertNotEqual(first.event_group.tolist(), other.event_group.tolist())

    def test_timing_and_model_scores_do_not_change_selection(self):
        first, _ = match_kinematics(self.frame)
        changed = self.frame.copy()
        changed["jet_time_ATLAS"] = changed["jet_time_ATLAS"] * -10 - 100
        changed["model_score"] = 1 - changed["model_score"]
        other, _ = match_kinematics(changed)
        pd.testing.assert_frame_equal(first.drop(columns=["jet_time_ATLAS", "model_score"]),
                                      other.drop(columns=["jet_time_ATLAS", "model_score"]))

    def test_validation_test_cannot_add_support(self):
        unsupported = self.frame.copy()
        unsupported["jet_pt_GeV"] = 1000.
        unsupported["jet_abs_eta"] = 1.5
        unsupported["event_group"] = "extra:" + unsupported["event_group"]
        train = unsupported[unsupported.split == "train"].groupby("label").head(2)
        unsupported = pd.concat([train, unsupported[unsupported.split != "train"]], ignore_index=True)
        matched, audit = match_kinematics(pd.concat([self.frame, unsupported], ignore_index=True))
        self.assertFalse(matched.event_group.str.startswith("extra:").any())
        self.assertGreater(sum(row["excluded_unsupported"] for row in audit["split_summary"]), 1000)

    def test_small_held_out_samples_are_rejected(self):
        small = pd.concat([
            self.frame[self.frame.split != "validation"],
            self.frame[self.frame.split == "validation"].groupby("label").head(100),
        ], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "too few validation jets"):
            match_kinematics(small)

    def test_original_event_leakage_is_rejected(self):
        bad = self.frame.copy()
        bad.loc[0, "event_group"] = bad.loc[bad.split == "test", "event_group"].iloc[0]
        with self.assertRaisesRegex(ValueError, "more than one original split"):
            match_kinematics(bad)


if __name__ == "__main__":
    unittest.main()
