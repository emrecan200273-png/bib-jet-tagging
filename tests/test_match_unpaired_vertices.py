"""Exercise the event join on small ROOT files."""

import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from match_unpaired_vertices import match, write_outputs


class MatchVertexTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="ppp-vertex-join-")
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.cells = self.directory / "cells"
        self.vertices = self.directory / "vertices"
        self.cells.mkdir()
        self.vertices.mkdir()

    def save_cells(self, runs, events, dtype=np.uint64):
        with uproot.recreate(self.cells / "cells.root") as handle:
            handle.mktree("JetContent", {
                "runNumber": np.asarray(runs, dtype=np.int32),
                "eventNumber": np.asarray(events, dtype=dtype),
            })

    def save_vertices(self, runs, events, counts, filename="vertices.root"):
        with uproot.recreate(self.vertices / filename) as handle:
            handle["OtherTree"] = {"unrelated": np.array([1])}
            handle["event/Vertices"] = {
                "runNumber": np.asarray(runs, dtype=np.int64),
                "eventNumber": np.asarray(events, dtype=np.uint64),
                "nPV": np.asarray(counts, dtype=np.int32),
            }

    def test_large_ids_duplicates_and_missing_counts_survive_export(self):
        big = 2**63 + 1
        self.save_cells([34, 34, 34, 35, -1], [big, big + 1, 12, 44, 14])
        self.save_vertices([34, 34, 34, -1], [big + 1, big, 12, 14], [3, 0, -1, 5])
        self.save_vertices([34, 34], [44, big], [1, 0], filename="more_vertices.root")
        report, rows = match(self.cells, self.vertices)
        self.assertEqual(report["duplicate_vertex_rows"], 1)
        self.assertEqual(report["key_matched_events"], 3)
        self.assertEqual(report["matched_valid_vertex_count"], 2)
        self.assertEqual(report["matched_zero_vertex_count"], 1)
        self.assertEqual(report["matched_invalid_vertex_count"], 1)
        self.assertEqual(report["unmatched_cell_events"], 1)
        self.assertEqual(report["vertex_only_event_keys"], 1)
        self.assertEqual(report["invalid_cell_key_rows"], 1)
        self.assertEqual(report["invalid_vertex_key_rows"], 1)
        streamed, streamed_rows = match(self.cells, self.vertices, matched_only=True)
        self.assertEqual(streamed_rows, rows)
        self.assertEqual(streamed["vertex_rows"], report["vertex_rows"])
        self.assertEqual(streamed["invalid_vertex_key_rows"], 1)
        self.assertEqual(streamed["invalid_vertex_count_rows"], 1)
        self.assertEqual(streamed["vertex_only_rows"], 1)
        self.assertIsNone(streamed["vertex_only_event_keys"])
        self.assertIsNone(streamed["unique_vertex_event_keys"])
        output = self.directory / "report"
        write_outputs(output, report, rows)
        self.assertEqual(json.loads((output / "summary.json").read_text()), report)
        with gzip.open(output / "event_mapping.csv.gz", "rt", newline="") as handle:
            mapping = list(csv.DictReader(handle))
        self.assertEqual([row["vertex_count"] for row in mapping], ["0", "3", "", "", ""])
        self.assertEqual(mapping[0]["eventNumber"], str(big))
        self.assertEqual(mapping[1]["eventNumber"], str(big + 1))
        self.assertEqual(mapping[2]["status"], "invalid_vertex_count")
        self.assertEqual(mapping[3]["status"], "unmatched")

    def test_conflicting_vertex_rows_fail(self):
        self.save_cells([34], [99])
        self.save_vertices([34, 34], [99, 99], [0, 1])
        for mode in (False, True):
            with self.subTest(matched_only=mode):
                with self.assertRaisesRegex(ValueError, "Conflicting vertex counts"):
                    match(self.cells, self.vertices, matched_only=mode)

    def test_matched_only_scans_bib_schema_and_checks_luminosity_block(self):
        with uproot.recreate(self.cells / "cells.root") as handle:
            handle.mktree("JetContent", {
                "runNumber": np.array([34, 34, 34], dtype=np.int32),
                "eventNumber": np.array([10, 20, 2**64 - 1], dtype=np.uint64),
                "lumiBlock": np.array([3, 3, 3], dtype=np.int32),
            })
        with uproot.recreate(self.vertices / "vertices.root") as handle:
            handle.mktree("BIB", {
                "runNumber": np.repeat(np.int32(34), 7),
                "eventNumber": np.array([10, 10, 20, 20, 55, 55, -1], dtype=np.int32),
                "lumiBlockNumber": np.array([3, 3, 3, 4, 2, 2, 3], dtype=np.int32),
                "nPrimaryVertices": np.array([0, 0, -1, 5, 1, 2, 7], dtype=np.int32),
            })
        report, rows = match(
            self.cells, self.vertices, matched_only=True, vertex_tree="BIB",
            key_fields=("runNumber", "eventNumber", "lumiBlock"),
            vertex_keys=("runNumber", "eventNumber", "lumiBlockNumber"),
        )
        self.assertEqual(report["vertex_count_field"], "nPrimaryVertices")
        self.assertEqual(report["vertex_rows"], 7)
        self.assertEqual(report["vertex_only_rows"], 3)
        self.assertEqual(report["matched_vertex_rows"], 3)
        self.assertEqual(report["duplicate_vertex_rows"], 1)
        self.assertEqual(report["invalid_vertex_key_rows"], 1)
        self.assertEqual(report["unique_matched_vertex_event_keys"], 2)
        self.assertEqual(report["vertex_duplicate_check_scope"], "matched_cell_keys")
        self.assertEqual(report["matched_zero_vertex_count"], 1)
        self.assertEqual(report["matched_invalid_vertex_count"], 1)
        self.assertEqual(report["unmatched_cell_events"], 1)
        self.assertEqual([row[5] for row in rows], [0, None, None])

    def test_repeated_cell_events_fail(self):
        self.save_cells([34, 34], [99, 99])
        self.save_vertices([34], [99], [0])
        with self.assertRaisesRegex(ValueError, "not one-to-one"):
            match(self.cells, self.vertices)

    def test_float_event_ids_are_rejected(self):
        self.save_cells([34], [99.0], dtype=np.float64)
        self.save_vertices([34], [99], [0])
        with self.assertRaisesRegex(ValueError, "eventNumber must be a scalar integer"):
            match(self.cells, self.vertices)
