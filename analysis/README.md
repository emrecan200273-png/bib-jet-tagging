# Data and training outputs

## Input ROOT files

The following paths are relative to `DataForEmre/` in the local project.

| Folder | Contents |
| --- | --- |
| `collision/LLP/N1N2_rpv_1400_10ns/` | 2 files containing simulated LLP jets, the stored match flag and calorimeter cells. |
| `non_collision/unpaired/physics_Background_NTuple_unsplit_v2/` | 291 files containing unpaired jets and their cells, all from run 484258. |
| `non_collision/unpaired/user.ezaya.UNPAIRED_nPV_nocells.EXT0/` | 100 files containing reconstructed vertex counts. Run and event numbers link each count to the corresponding cell event. The luminosity block provides an additional check. |

## Training outputs

Each training run writes to a timestamped folder under `analysis/outputs/`.

| Folder under `outputs/` | Contents |
| --- | --- |
| `02_xgboost_baseline/` | Trained trees, test scores and the comparison after matching the samples in transverse momentum and direction. |
| `03_train_and_compare/` | The four trained neural networks, their learning curves and the weighted ROC curve. |

Notebook 03 saves two tables for the same delayed test jets. In `weighted_delayed_test_comparison.csv`, the thresholds are fixed on validation. In `weighted_delayed_test_roc_scan.csv`, the thresholds correspond to 95% weighted LLP retention on the test ROC curve.

The last section of [notebook 04](../notebooks/04_bib_checks.ipynb) reads two earlier studies from their original folders under `outputs/` when those reports are available.
