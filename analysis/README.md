# Analysis

See the [project README](../README.md) for setup and the XGBoost baseline.
Run the commands below from the project folder.

## Input samples

Paths are relative to `DataForEmre/`.

| Sample | ROOT folder | How it is used |
| --- | --- | --- |
| LLP simulation | `collision/LLP/N1N2_rpv_1400_10ns` | Signal jets with `jet_is_llp_matched == 1`. |
| Unpaired data | `non_collision/unpaired/physics_Background_NTuple_unsplit_v2` | Jet and cell inputs for the BIB-enriched sample from run `484258`. |
| Unpaired event keys | `non_collision/unpaired/physics_Background_NTuple` | `JetContent` input to the vertex join. The resulting mapping is applied to `unsplit_v2`. |
| Vertex counts | `non_collision/unpaired/user.ezaya.UNPAIRED_nPV_nocells.EXT0` | `BIB` tree with `nPrimaryVertices` and event keys. |

The cell reader checks the event key and jet position in the original ROOT
file before following the jet-to-cluster and cluster-to-cell counts. ParT
gets the jet summaries and six features per cell: stored time, transformed
relative energy, $R$, $z$, and the offsets in $\eta$ and $\phi$ from the jet. The stored
`jet_BIB_score` is used for the comparison.

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

The first command trains a jet-level MLP and runs Deep Sets and a small
Transformer with up to 64 cells per jet. The second runs Deep Sets and the
Transformer with all valid stored cells. The other commands run ParT,
ParticleNet-Lite and the deeper Transformer variants. Reports in
`analysis/outputs/` include test results and the train–validation loss gap.
The delayed region uses `jet_time_ATLAS > 3` ns.

## Set Transformer and Cluster Transformer

- [ParT](part_cell_model.py) uses attention between pairs of cells, with
  information about each pair in the attention weights.
- [Set Transformer](set_cell_transformer.py) lets cells exchange information
  through eight learned summaries per block. This avoids comparing every
  cell pair. [Set Transformer paper](https://proceedings.mlr.press/v97/lee19d.html)
- [Cluster Transformer](cluster_cell_transformer.py) pools cells within each
  stored ATLAS topocluster, then runs attention across the clusters in the jet.
  The grouping comes from `jet_n_clusters` and `cluster_n_cells`.

All three used all valid cells and the same 4,523 test jets from unpaired run
`484258` and the LLP simulation. The counts below are for jet time above 3 ns.
Each cut targeted at least 95% LLP retention on delayed validation jets. This
comparison includes the two delayed unpaired jets without a valid stored
`jet_BIB_score`, so the denominator is 42 here and 40 in the stored-score comparison.

| Model | Unpaired tagged | Simulated LLP kept | Time to score all test jets | Validation minus training log loss |
| --- | ---: | ---: | ---: | ---: |
| ParT | 42 / 42 | 782 / 816 | 2.35 s | +0.0162 |
| Set Transformer | 42 / 42 | 779 / 816 | 0.33 s | +0.0146 |
| Cluster Transformer | 41 / 42 | 777 / 816 | 0.15 s | +0.0176 |

Times are medians over three local CPU runs with four threads, measured after
loading the inputs and warming up each model. Set Transformer tagged the same
number of delayed unpaired jets as ParT and kept three fewer LLP jets.

The loss gap uses weighted log loss in evaluation mode at the selected epoch
for each model. ParT's validation loss rose after epoch 8 as training loss fell.
That is the clearest sign of overfitting in these runs. Set Transformer's
validation loss fluctuated and ended near its minimum. Cluster Transformer had
one epoch after its minimum, too few to judge a trend. These runs reuse the
same test split from one unpaired run and one LLP signal sample.

Run the two alternatives after training ParT.

```bash
python analysis/train_pairwise_cells.py --set-transformer
python analysis/train_pairwise_cells.py --cluster-transformer
```

Results are written to `analysis/outputs/set_transformer_comparison/` and
`analysis/outputs/cluster_transformer_comparison/`. Each includes a learning
curve and `kinematic_bins_delayed_cut.csv` for shared $p_T$ and |η| bins.

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

```bash
python analysis/study_bib_profiles.py
python analysis/compare_part_emil.py
```

The same-jet table and score counts go to
`analysis/outputs/part_emil_same_jets/control_484258/`.

For an independent run, pass `--run`, `--cell-dir` (a `JetContent` export) and
`--vertex-dir` (the `BIB` sidecar) to `compare_part_emil.py`. Older sidecars
can store unsigned event IDs in signed 32-bit fields. When verified on a pilot,
use `--vertex-event-int32-bits` to recover the stored bits before matching them
against the full event IDs in the cell files.

## Open questions

- Which Athena release and export flags produced `physics_Background_NTuple_unsplit_v2`?
- Which unpaired trigger selection should we apply consistently across runs `484258` and `486706`?

See [model options](model_options.md) for notes on the papers.
