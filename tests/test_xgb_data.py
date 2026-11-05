"""Checks for event leakage and the producer's missing-value convention."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import awkward as ak
import numpy as np
import pandas as pd
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from xgb_data import DATA_GROUPS, assign_event_splits, prepare_training_data, sanitize_values


class EventSplitTests(unittest.TestCase):
    def setUp(self):
        # Deliberately different jet multiplicities: stratify events, not jets.
        self.frame = pd.DataFrame([
            {"event_group": f"sample{label}:42:{event}", "label": label, "jet_index": jet}
            for label in [0, 1] for event in range(100)
            for jet in range(1 + event % 4)
        ])

    def test_jets_from_one_event_stay_together(self):
        result = assign_event_splits(self.frame)
        self.assertEqual(result.groupby("event_group")["split"].nunique().max(), 1)
        event_counts = result.drop_duplicates("event_group").groupby(["split", "label"]).size()
        for label in [0, 1]:
            self.assertEqual(event_counts["train", label], 60)
            self.assertEqual(event_counts["validation", label], 20)
            self.assertEqual(event_counts["test", label], 20)
        self.assertNotIn("split", self.frame)

    def test_seed_is_reproducible_and_row_order_does_not_matter(self):
        first = assign_event_splits(self.frame, seed=42).groupby("event_group")["split"].first()
        reordered = assign_event_splits(self.frame.sample(frac=1, random_state=9), seed=42)
        pd.testing.assert_series_equal(first, reordered.groupby("event_group")["split"].first())
        other = assign_event_splits(self.frame, seed=43).groupby("event_group")["split"].first()
        self.assertTrue((first != other).any())

    def test_conflicting_labels_for_one_event_are_rejected(self):
        bad = self.frame.copy()
        bad.loc[0, "event_group"] = "sample1:42:0"
        with self.assertRaisesRegex(ValueError, "more than one label"):
            assign_event_splits(bad)

    def test_missing_class_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "both labels"):
            assign_event_splits(self.frame[self.frame.label == 0])


class MissingValueTests(unittest.TestCase):
    def test_sentinels_and_infinities_become_missing_but_real_negative_times_stay(self):
        raw = np.array([-999999., -7., 0., 1., np.nan, np.inf, -np.inf])
        result = sanitize_values(raw)
        np.testing.assert_equal(result, [np.nan, -7., 0., 1., np.nan, np.nan, np.nan])
        self.assertEqual(raw[0], -999999.)

    def test_masked_values_are_missing(self):
        result = sanitize_values(np.ma.array([1., 2.], mask=[False, True]))
        np.testing.assert_equal(result, [1., np.nan])


class RootExtractionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="ppp-xgb-data-test-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / "DataForEmre"
        self.manifest = self.root / "analysis/provenance/manifest.json"
        self.manifest.parent.mkdir(parents=True)
        self.environment = patch.dict(os.environ, {
            "PPP_DATA_ROOT": str(self.data), "PPP_DATA_MANIFEST": str(self.manifest),
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def make_sources(self, *, duplicate_event=False, unaligned=False):
        records = []
        for group, (sample, label) in DATA_GROUPS.items():
            path = self.data / group / "tiny.root"
            path.parent.mkdir(parents=True)
            pairs = {
                "jet_pt": [50000., 60000.], "jet_eta": [.4, -.6],
                "jet_time_ATLAS": [2., 3.], "jet_time_spread": [1., 1.],
                "jet_time_sigma": [.3, .4], "jet_n_clusters": [2., 3.],
                "jet_TileEnergyFrac": [.4, .5], "jet_BIB_score": [.5, .6],
            }
            payload = {field: [values[:] for _ in range(20)] for field, values in pairs.items()}
            # Missing ancillary features must survive selection as NaN.
            payload["jet_time_sigma"][0][0] = -999999.
            payload["jet_BIB_score"][0][0] = -999999.
            if unaligned:
                payload["jet_time_spread"][0] = [1.]
            payload = {field: ak.Array(values) for field, values in payload.items()}
            events = np.arange(100, 120, dtype=np.uint64)
            if duplicate_event:
                events[1] = events[0]
            payload.update(runNumber=np.full(20, 42 + label, dtype=np.int32), eventNumber=events)
            if sample == "LLP":
                payload["jet_is_llp_matched"] = ak.Array([[1, 0] for _ in range(20)])
            with uproot.recreate(path) as handle:
                handle["JetContent"] = payload
            records.append({"path": str(path.relative_to(self.data)),
                            "size": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.manifest.write_text(json.dumps({"files": records}))
        return records

    def test_extracts_verified_sources_and_keeps_missing_ancillary_features(self):
        self.make_sources()
        output = self.root / "outputs"
        frame, audit = prepare_training_data(self.root, output)
        self.assertEqual(frame.groupby("sample").size().to_dict(), {"LLP": 20, "unpaired": 40})
        self.assertEqual(frame.jet_time_sigma.isna().sum(), 2)
        self.assertEqual(frame.jet_BIB_score.isna().sum(), 2)
        self.assertTrue(frame.jet_pt_GeV.isin([50., 60.]).all())
        self.assertTrue(all(item["hash_verified"] for item in audit["files"]))
        self.assertEqual(frame.groupby("event_group")["split"].nunique().max(), 1)
        for name in ["training_jets.csv.gz", "data_audit.json", "split_summary.csv", "cutflow.csv"]:
            self.assertTrue((output / name).is_file())

    def test_hash_mismatch_is_rejected_before_reading(self):
        records = self.make_sources()
        path = self.data / records[0]["path"]
        content = bytearray(path.read_bytes())
        content[-1] ^= 1  # Same size, changed content: size checks alone cannot catch it.
        path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            prepare_training_data(self.root, self.root / "outputs")

    def test_duplicate_event_rows_are_rejected(self):
        self.make_sources(duplicate_event=True)
        with self.assertRaisesRegex(ValueError, "Duplicate physical event"):
            prepare_training_data(self.root, self.root / "outputs")

    def test_unaligned_jet_vectors_are_rejected(self):
        self.make_sources(unaligned=True)
        with self.assertRaisesRegex(ValueError, "different lengths"):
            prepare_training_data(self.root, self.root / "outputs")


if __name__ == "__main__":
    unittest.main()
