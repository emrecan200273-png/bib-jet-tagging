# Analysis

See the [project README](../README.md) for setup and the XGBoost baseline.
Run the commands below from the project folder.

## Prepare the vertex counts

The separate vertex files use the `BIB` tree. Match them to the cell files by
run, event and luminosity block. The cell studies use unpaired events with zero
reconstructed vertices and keep the original event splits.

```bash
python -m pip install 'torch>=2.5,<3'

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

## Train with cells

```bash
python analysis/train_cell_models.py
python analysis/train_cell_models.py --all-cells --threads 4
python analysis/train_pairwise_cells.py
python analysis/train_pairwise_cells.py --particle-net-lite
python analysis/train_pairwise_cells.py --deep-transformers
```

The first two commands compare the jet MLP, Deep Sets and a small Transformer
with up to 64 cells per jet and with all valid stored cells. The other commands
run ParT, ParticleNet-Lite and the deeper Transformer variants. Reports in
`analysis/outputs/` include test results and the train–validation loss gap.
The delayed region uses `jet_time_ATLAS > 3` ns.

## Check the results

| Script | What it checks |
| --- | --- |
| [study_jet_acceptance.py](study_jet_acceptance.py) | How the $p_T$, η and Tile cuts affect the samples and timing baseline |
| [study_unpaired_runs.py](study_unpaired_runs.py) | How jet-level timing and kinematic controls transfer to other runs without cell input |
| [study_bib_profiles.py](study_bib_profiles.py) | How the stored timing score compares with fits that try both beam directions |
| [study_cell_inputs.py](study_cell_inputs.py) | How ParT responds when cell times are shuffled within each jet |
| [compare_part_emil.py](compare_part_emil.py) | How frozen ParT and Emil scores compare on the same jets |

Run the cell-input check after training ParT. Run the BIB profile study before
the same-jet comparison so it can read the stored score cut. The comparison
also reports results in shared $p_T$ and |η| bins and in the 3–10 ns window.

For an independent run, pass `--run`, `--cell-dir` (a `JetContent` export) and
`--vertex-dir` (the `BIB` sidecar) to `compare_part_emil.py`. Older sidecars
can store unsigned event IDs in signed 32-bit fields. When verified on a pilot,
use `--vertex-event-int32-bits` to recover the stored bits before matching them
against the full event IDs in the cell files.

See [model options](model_options.md) for notes on the papers.
