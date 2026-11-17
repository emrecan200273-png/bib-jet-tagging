"""Check that cell order and padding do not change jet predictions."""

from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import awkward as ak
import numpy as np
import pandas as pd
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.data.cell_jet_data import extract_cells


class CellTransformerTests(unittest.TestCase):
    def test_cluster_ids_follow_the_retained_cells(self):
        with TemporaryDirectory() as tmp:
            with uproot.recreate(Path(tmp) / "cells.root") as handle:
                handle.mktree("JetContent", {
                    "runNumber": np.array([1]), "eventNumber": np.array([3]),
                    "jet_pt": ak.Array([[40_000, 50_000]]), "jet_eta": ak.Array([[0., 0.]]),
                    "jet_n_clusters": ak.Array([[2, 1]]), "cluster_n_cells": ak.Array([[2, 2, 1]]),
                    "cell_e": ak.Array([[1000., 3000., 4000., 1000., 1000.]]),
                    "cell_time": ak.Array([[1., -999999., 3., 4., 5.]]),
                    **{field: ak.Array([[0.] * 5]) for field in ("cell_x", "cell_y", "cell_z",
                                                                "cell_dEta_jet", "cell_dPhi_jet")},
                })
            jets = pd.DataFrame({"source_file": ["cells.root"] * 2, "row_index": [0, 0], "jet_index": [0, 1],
                                 "event_group": ["LLP:1:3"] * 2, "label": [0, 0], "split": ["test"] * 2,
                                 "jet_eta": [0., 0.]})
            cells, mask, audit, clusters = extract_cells(jets, tmp, None, with_cluster_ids=True)
            reference_cells, reference_mask, reference_audit = extract_cells(jets, tmp, None)
            np.testing.assert_array_equal(cells, reference_cells)
            np.testing.assert_array_equal(mask, reference_mask)
            self.assertEqual(audit, reference_audit)
            self.assertEqual(cells[0, :, 0].tolist(), [3., 1., 4.])
            self.assertEqual(clusters.tolist(), [[1, 0, 1], [0, -1, -1]])

    @unittest.skipUnless(find_spec("torch"), "PyTorch is optional")
    def test_torch_cell_transformers(self):
        # Isolate PyTorch from XGBoost's OpenMP runtime.
        result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def check_torch_models():
    import torch

    from analysis.models.cell_models import build_model

    torch.set_num_threads(1)
    with torch.inference_mode():
        rng = torch.Generator().manual_seed(12)
        jets = torch.randn(2, 5, generator=rng)
        features = torch.randn(2, 5, 6, generator=rng)
        ids = torch.tensor([[0, 1, 0, 2, 1], [0, 0, 1, 1, -1]])
        mask = ids >= 0
        permutation = torch.tensor([4, 1, 2, 0, 3])
        for name in ("set_transformer", "cluster_transformer"):
            grouped = name == "cluster_transformer"
            cells = torch.cat((features, ids[..., None]), dim=-1) if grouped else features
            model = build_model(name, jets.shape[1], cells.shape[-1]).eval()
            permuted = cells[:, permutation].clone()
            if grouped:
                permuted[0, :, -1] = 2 - permuted[0, :, -1]
            permuted = torch.cat((permuted, torch.full_like(permuted[:, :2], torch.nan)), dim=1)
            padded = torch.cat((mask[:, permutation], mask.new_zeros((2, 2))), dim=1)
            torch.testing.assert_close(model(jets, cells, mask), model(jets, permuted, padded))

            width = cells.shape[-1]
            empty = model(jets, torch.zeros(2, 0, width), torch.zeros(2, 0, dtype=torch.bool))
            masked = model(jets, torch.zeros(2, 3, width), torch.zeros(2, 3, dtype=torch.bool))
            assert torch.isfinite(empty).all()
            torch.testing.assert_close(empty, masked)


if __name__ == "__main__":
    check_torch_models()
