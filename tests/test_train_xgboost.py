"""Saved-model checks independent of private ROOT data."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from train_xgboost import balanced_weights, predict_model, score_saved_model
from xgb_data import PHYSICS_FEATURES


def _jets(labels):
    labels = np.asarray(labels)
    return pd.DataFrame({
        "jet_pt_GeV": np.full(len(labels), 50.),
        "jet_eta": np.full(len(labels), .5),
        "jet_time_ATLAS": 2. + 6. * labels,
        "jet_time_spread": 2. + 5. * labels,
        "jet_time_sigma": np.full(len(labels), 1.),
        "jet_n_clusters": 20. - 16. * labels,
        "jet_TileEnergyFrac": np.full(len(labels), .5),
    })


class SavedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory(prefix="ppp-xgb-inference-")
        cls.output = Path(cls.directory.name)
        labels = np.tile([0, 1], 20)
        cls.frame = _jets(labels)
        cls.model = XGBClassifier(n_estimators=8, max_depth=1, n_jobs=1,
                                  random_state=42, objective="binary:logistic")
        cls.model.fit(cls.frame[PHYSICS_FEATURES], labels)
        cls.model.save_model(cls.output / "xgboost.ubj")
        cls.cut = np.nextafter(predict_model(cls.model, _jets([0]), PHYSICS_FEATURES)[0], np.inf)
        schema = {"features": PHYSICS_FEATURES,
                  "selection": {"min_pt_GeV": 30., "max_abs_eta": 1.7},
                  "threshold": float(cls.cut)}
        (cls.output / "feature_schema.json").write_text(json.dumps(schema))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_saved_roundtrip_and_float32_cut(self):
        frame = _jets([0, 1])
        frame.index = ["LLP", "BIB"]
        result = score_saved_model(frame, self.output)
        np.testing.assert_array_equal(result.bib_score, predict_model(self.model, frame, PHYSICS_FEATURES))
        self.assertEqual(result.index.tolist(), ["LLP", "BIB"])
        self.assertTrue(result.in_selection.all())
        self.assertEqual(result.tagged_as_bib.tolist(), [False, True])

    def test_scope_rejects_nonfinite_momentum_and_exclusive_boundaries(self):
        frame = _jets([0] * 8)
        frame.loc[0, "jet_pt_GeV"] = 29.99
        frame.loc[1, "jet_pt_GeV"] = np.inf
        frame.loc[2, "jet_eta"] = 1.7
        frame.loc[3, "jet_time_ATLAS"] = np.nan
        frame.loc[4, "jet_TileEnergyFrac"] = 0.
        frame.loc[5, "jet_TileEnergyFrac"] = 1.
        frame.loc[6, "jet_eta"] = -np.inf
        frame.loc[7, "jet_pt_GeV"] = 30.
        result = score_saved_model(frame, self.output)
        self.assertEqual(result.in_selection.tolist(), [False] * 7 + [True])
        self.assertTrue(result.bib_score.iloc[:7].isna().all())
        self.assertTrue(result.tagged_as_bib.iloc[:7].isna().all())

    def test_missing_secondary_features_do_not_remove_jet(self):
        frame = _jets([0, 0])
        frame.loc[0, "jet_time_sigma"] = -999999.
        frame.loc[1, "jet_time_sigma"] = np.nan
        result = score_saved_model(frame, self.output)
        self.assertTrue(result.in_selection.all())
        self.assertEqual(result.bib_score.iloc[0], result.bib_score.iloc[1])

    def test_physical_invalids_use_same_missing_rules_as_training(self):
        frame = _jets([0, 0, 0, 0])
        frame.loc[0, "jet_time_sigma"] = 0.
        frame.loc[1, "jet_time_spread"] = -1.
        frame.loc[2, "jet_n_clusters"] = 0.
        frame.loc[3, "jet_n_clusters"] = 1.5
        with patch("train_xgboost.predict_model", return_value=np.array([.2] * 4)) as predict:
            result = score_saved_model(frame, self.output)
            seen = predict.call_args.args[1]
        self.assertTrue(result.in_selection.all())
        for row, column in [(0, "jet_time_sigma"), (1, "jet_time_spread"),
                            (2, "jet_n_clusters"), (3, "jet_n_clusters")]:
            self.assertTrue(np.isnan(seen.loc[row, column]), (row, column))

    def test_weights_depend_on_training_counts(self):
        actual = balanced_weights([0, 1], [0, 0, 0, 1])
        np.testing.assert_allclose(actual, [2 / 3, 2.])

    def _matched_schema(self, directory):
        directory = Path(directory)
        self.model.save_model(directory / "xgboost.ubj")
        schema = json.loads((self.output / "feature_schema.json").read_text())
        schema["kinematic_support"] = {
            "pt_edges_GeV": [30., 60., 100.],
            "abs_eta_edges": [0., .5, 1.],
            "supported_bins": [{"pt_bin": 0, "eta_bin": 0},
                               {"pt_bin": 1, "eta_bin": 1}],
        }
        (directory / "feature_schema.json").write_text(json.dumps(schema))
        return directory

    def test_matched_scope_checks_bin_support_and_final_inclusive_edges(self):
        frame = _jets([0] * 10)
        frame["jet_pt_GeV"] = [30., 59.99, 60., 100., 60., 100.01, 45., np.nan, np.inf, 29.99]
        frame["jet_eta"] = [0., .49, -.5, 1., .49, .7, 1.01, .2, .7, .2]
        frame.index = [f"row_{i * 3}" for i in range(len(frame))]
        with TemporaryDirectory(prefix="ppp-xgb-matched-scope-") as directory:
            result = score_saved_model(frame, self._matched_schema(directory))
        self.assertEqual(result.in_selection.tolist(), [True] * 4 + [False] * 6)
        self.assertEqual(result.index.tolist(), frame.index.tolist())
        self.assertTrue(result.bib_score.iloc[4:].isna().all())
        self.assertTrue(result.tagged_as_bib.iloc[4:].isna().all())

    def test_matched_scope_accepts_empty_input_without_scoring(self):
        frame = _jets([])
        with TemporaryDirectory(prefix="ppp-xgb-empty-scope-") as directory:
            with patch("train_xgboost.predict_model") as predict:
                result = score_saved_model(frame, self._matched_schema(directory))
        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), ["in_selection", "bib_score", "tagged_as_bib"])
        predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
