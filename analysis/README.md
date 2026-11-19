# Where my results go

Each training run gets its own folder, named by date and time. I use these files when I want to check a result without training again.

| Local folder | What I open it for |
| --- | --- |
| `outputs/02_xgboost_baseline/` | The XGBoost trees, their test scores and the comparison at similar jet momenta and directions. |
| `outputs/03_train_and_compare/` | The four trained models, the learning curves and the weighted ROC. |

In a run from notebook 03, `weighted_delayed_test_comparison.csv` gives the test counts with the cuts I chose on validation. `weighted_delayed_test_roc_scan.csv` gives the counts when I choose the 95% point on the test curve itself.

Older runs are still under `outputs/` with their original names. The last part of [notebook 04](../notebooks/04_bib_checks.ipynb) reads two of those local reports if they're available.

## Which ROOT files I used

These folders sit under `DataForEmre/` in my local copy.

| Folder | What it gives me |
| --- | --- |
| `collision/LLP/N1N2_rpv_1400_10ns/` | 2 files with simulated LLP jets, their match flag and their cells. |
| `non_collision/unpaired/physics_Background_NTuple_unsplit_v2/` | 291 files with unpaired jets and their cells, all from run 484258. |
| `non_collision/unpaired/user.ezaya.UNPAIRED_nPV_nocells.EXT0/` | 100 files with vertex counts. I match them to the cells by run and event number and check the luminosity block. |
