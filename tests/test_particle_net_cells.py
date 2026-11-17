"""Regression test for storage-order ties in calorimeter neighbor graphs."""

from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
import unittest


def check_neighbor_ties():
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.models.particle_net_cell_model import ParticleNetLiteCells

    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = ParticleNetLiteCells(5, 6, angular_scale=(0.2, 0.3)).eval()
    jets = torch.randn(2, 5)
    cells = torch.randn(2, 13, 6)
    # Twelve possible neighbors share a direction and only seven are selected.
    cells[..., 4:6] = 0.25
    mask = torch.ones((2, 13), dtype=torch.bool)
    order = torch.tensor([0, 6, 10, 8, 3, 11, 1, 4, 9, 2, 12, 7, 5])
    padding = torch.full((2, 33, 6), float("nan"))
    padding_mask = torch.zeros((2, 33), dtype=torch.bool)
    with torch.inference_mode():
        reference = model(jets, cells, mask)
        reordered = model(jets, cells[:, order], mask[:, order])
        padded = model(
            jets, torch.cat((padding, cells[:, order]), dim=1),
            torch.cat((padding_mask, mask[:, order]), dim=1),
        )
    torch.testing.assert_close(reference, reordered, rtol=0, atol=1e-6)
    torch.testing.assert_close(reference, padded, rtol=0, atol=1e-6)


@unittest.skipUnless(find_spec("torch"), "PyTorch is optional")
class ParticleNetCellsTests(unittest.TestCase):
    def test_direction_ties_at_neighbor_cut_ignore_cell_order_and_padding(self):
        # Isolate PyTorch from the XGBoost OpenMP runtime in this test suite.
        result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    check_neighbor_ties()
