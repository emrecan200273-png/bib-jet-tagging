# Running the analysis

Start with the [setup and notebooks](../README.md). Run the commands below
from the project root. Results are saved in `analysis/outputs/`.

## Data and cuts

Paths below are relative to `DataForEmre/`.

| Sample | Folder | Use |
| --- | --- | --- |
| LLP simulation | `collision/LLP/N1N2_rpv_1400_10ns` | Signal jets with `jet_is_llp_matched == 1`. |
| Unpaired cells | `non_collision/unpaired/physics_Background_NTuple_unsplit_v2` | BIB-enriched jets and cells from run `484258`. |
| Unpaired event keys | `non_collision/unpaired/physics_Background_NTuple` | Events used to match the vertex counts to `unsplit_v2`. |
| Vertex counts | `non_collision/unpaired/user.ezaya.UNPAIRED_nPV_nocells.EXT0` | Separate `BIB` tree with `nPrimaryVertices`. |

The cell-model selection is $p_T \geq 30$ GeV, $|\eta| < 1.7$ and
$0 < \text{Tile energy fraction} < 1$. Unpaired events must have zero vertices.
We train on all jet times; “delayed” means `jet_time_ATLAS > 3` ns.

Events are split 60/20/20 into training, validation and test. Jets from one
event stay together. Scaling and class weights use training data only;
score cuts use validation data. The stored BIB score is a comparison method,
not a training input.

ROOT units are MeV, mm and ns. The [cell reader](data/cell_jet_data.py) checks
jet identities and follows `jet_n_clusters` → `cluster_n_cells` → cells.
It keeps positive-energy cells with finite, non-missing features. Times are
unchanged. See the [model inputs](models/README.md#inputs).

## 1. Check files and train the baseline

The first two notebooks use small samples. The baseline verifies file hashes
and reads the selected datasets in full.

```bash
python tools/run_first_look.py 01_remote_data_inventory.ipynb
python tools/run_first_look.py 02_first_data_look.ipynb
python -m analysis.training.train_xgboost
python tools/run_first_look.py 03_xgboost_baseline.ipynb
```

Notebook 3 normally shows saved results. Set `RETRAIN = True` to train again.

## 2. Match vertex counts

Match by run, event and luminosity block, then build the zero-vertex baseline.

```bash
ppp_unpaired_dir="${PPP_DATA_ROOT:-DataForEmre}/non_collision/unpaired"
python -m analysis.data.match_unpaired_vertices \
  "$ppp_unpaired_dir/physics_Background_NTuple" \
  "$ppp_unpaired_dir/user.ezaya.UNPAIRED_nPV_nocells.EXT0" \
  --cell-tree JetContent --vertex-tree BIB --vertex-branch nPrimaryVertices \
  --key-branches runNumber eventNumber lumiBlock \
  --vertex-key-branches runNumber eventNumber lumiBlockNumber \
  --matched-only --output-dir analysis/outputs/unpaired_vertex_match

python -m analysis.training.train_xgboost_vertex
```

## 3. Train the cell models

Run these in order. Later steps reuse saved splits, scaling and reference scores.

```bash
python -m pip install 'torch>=2.5,<3'
python -m analysis.training.train_cell_models
python -m analysis.training.train_cell_models --all-cells --threads 4
python -m analysis.training.train_pairwise_cells
```

The first training run uses a jet MLP, Deep Sets and a Transformer with up to
64 cells per jet. The second uses all valid cells for Deep Sets and Transformer.
The third trains our ParT-inspired model on all valid cells.

### Set Transformer and Cluster Transformer

After ParT, run either or both:

```bash
python -m analysis.training.train_pairwise_cells --set-transformer
python -m analysis.training.train_pairwise_cells --cluster-transformer
```

The same command also accepts `--particle-net-lite` or `--deep-transformers`.
See the [model guide](models/README.md) for how they work and compare.

Each output folder contains `metrics.json`, `test_predictions.csv.gz` and
learning curves. Set and Cluster also write `kinematic_bins_delayed_cut.csv`
for the check at similar momentum and direction.

## 4. Compare with Emil's score

Run the timing-profile study first; it saves the validation cut used below.

```bash
python -m analysis.studies.study_bib_profiles
python -m analysis.studies.compare_part_emil
```

This replays the saved ParT model and compares both methods on jets with valid
scores. Results go to `analysis/outputs/part_emil_same_jets/control_484258/`.
The report includes all times, delayed jets, 3–10 ns, and shared momentum/direction bins.

For a new run, pass `--run`, `--cell-dir` (cell ROOT files) and `--vertex-dir`
(vertex ROOT files). Some older vertex files store event IDs as signed 32-bit
integers. Use `--vertex-event-int32-bits` only after checking the encoding
against the full IDs in a cell file.

## Other checks

| Script | Question |
| --- | --- |
| [study_jet_acceptance.py](studies/study_jet_acceptance.py) | What changes when we relax the jet cuts? |
| [study_unpaired_runs.py](studies/study_unpaired_runs.py) | Do jet-level results hold in other runs? |
| [study_cell_inputs.py](studies/study_cell_inputs.py) | What happens if we shuffle times between cells in each jet? |

Each script lists its options with `python -m analysis.studies.<name> --help`.

Still to confirm with Emil: the production version of `unsplit_v2`, the common
unpaired trigger selection, and a cell export for run `486706`.
