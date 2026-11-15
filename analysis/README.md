# Analysis

See the [project README](../README.md) for setup, data paths and the reference to Emil’s GitLab.

- [Notebook 1](01_remote_data_inventory.ipynb) checks file metadata and stored variables.
- [Notebook 2](02_first_data_look.ipynb) reads a small sample and makes the first plots.
- [Notebook 3](03_xgboost_baseline.ipynb) shows the first XGBoost study and its controls.
- [Model options](model_options.md) contains short Swedish notes on the reference papers and possible next models.
- `first_look.py` reads and checks the data; `first_look_plots.py` makes the figures.
- `data_paths.py` lets the data and private manifest stay outside the repository.
- `train_xgboost.py` runs the full first-model study. `xgb_data.py` checks and splits the data; `xgb_evaluation.py` evaluates the models; `xgb_matching.py` makes the separate kinematic check.

## Run the cell models

Run these commands from the project root in the Python environment described in
the project README. The vertex-count ROOT files come separately from CERNBox/EOS.
For the data layout used in this study, prepare the jet table and vertex mapping
as follows:

```bash
python -m pip install 'torch>=2.5,<3'
python analysis/train_xgboost.py

ppp_unpaired_dir="${PPP_DATA_ROOT:-DataForEmre}/non_collision/unpaired"
python analysis/match_unpaired_vertices.py \
  "$ppp_unpaired_dir/physics_Background_NTuple" \
  "$ppp_unpaired_dir/user.ezaya.UNPAIRED_nPV_nocells.EXT0" \
  --cell-tree JetContent --vertex-tree BIB --vertex-branch nPrimaryVertices \
  --key-branches runNumber eventNumber lumiBlock \
  --vertex-key-branches runNumber eventNumber lumiBlockNumber \
  --matched-only --output-dir analysis/outputs/unpaired_vertex_match

python analysis/train_xgboost_vertex.py
```

The join writes `event_mapping.csv.gz`. The training code uses the run and event
IDs to apply the vertex counts to the `unsplit_v2` jets in the baseline table.
The cell models use unpaired events with zero reconstructed vertices as a
BIB-enriched sample and keep the original event splits. The cell reader follows
the stored jet-to-cluster-to-cell counts. The model inputs include the original
stored cell times, relative cell energies and cell positions.

Run the model comparisons in this order:

```bash
python analysis/train_cell_models.py
python analysis/train_cell_models.py --all-cells --threads 4
python analysis/train_pairwise_cells.py
python analysis/train_pairwise_cells.py --particle-net-lite
python analysis/train_pairwise_cells.py --deep-transformers
```

The first command trains a jet MLP, Deep Sets and a small Transformer. The next
command compares the 64-cell pilot with runs using every valid stored cell.
The remaining commands test cell-pair attention, ParticleNet-Lite and a two-block
Transformer with and without differences derived from Emil's BIB time profile.
The comparisons save test counts and plots under `analysis/outputs/`. The
all-cell runs also report train–validation loss gaps. The threshold is chosen
on validation. The delayed test subset uses ATLAS jet time above 3 ns.

Raw data, local results and the private transfer records are ignored by Git. Locally executed notebooks keep their outputs; only notebook source is stored in Git.
