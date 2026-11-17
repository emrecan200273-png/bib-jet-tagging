# Models

Each model gives a BIB score: higher means more BIB-like, not a measured
probability. We choose the cut on validation jets and apply it to test jets.

## Inputs

XGBoost and the jet MLP use five jet summaries: ATLAS time, time spread,
time uncertainty, cluster count and Tile energy fraction.

Cell models also use time, transformed relative energy, distance from the beam
(`R`), `z`, and offsets from the jet in `eta` and `phi`. Stored times stay unchanged.
Scaling uses training data only. Models ignore padding in unused cell slots.

## What each model does

These are small models adapted to our cells, not full reproductions of the papers.

| Model | Main idea | Code |
| --- | --- | --- |
| XGBoost | Adds decision trees using the five jet summaries. | [Training](../training/train_xgboost.py) |
| Jet MLP | Passes those summaries through a small neural network. | [Jet and cell baselines](cell_models.py) |
| Deep Sets | Reads each cell with the same network, then takes the mean and maximum. | [Jet and cell baselines](cell_models.py) |
| Transformer | Lets cells exchange information through attention. | [Jet and cell baselines](cell_models.py) |
| ParT-inspired | Adds cell-pair information to the attention weights. | [ParT](part_cell_model.py) |
| ParticleNet-Lite | Learns from each cell and its nearby neighbours. | [ParticleNet](particle_net_cell_model.py) |
| Deeper Transformer | Uses two attention blocks; one variant adds BIB timing coordinates. | [Deeper variants](deep_cell_transformer.py) |
| Set Transformer | Exchanges cell information through eight learned summaries per block. | [Set](set_cell_transformer.py) |
| Cluster Transformer | Combines cells within each ATLAS topocluster, then compares clusters. | [Cluster](cluster_cell_transformer.py) |

Set and Cluster aim to use all valid cells with less computation than ParT:
one uses learned summaries, the other uses the stored clusters.

## Results and overfitting

All three use the same 4,523 test jets and all valid cells. Counts are for
time above 3 ns, with cuts targeting 95% LLP retention on delayed validation.

| Model | Unpaired tagged | LLP kept | Scoring time | Validation − training loss |
| --- | ---: | ---: | ---: | ---: |
| ParT | 42 / 42 | 782 / 816 | 2.35 s | +0.0162 |
| Set Transformer | 42 / 42 | 779 / 816 | 0.33 s | +0.0146 |
| Cluster Transformer | 41 / 42 | 777 / 816 | 0.15 s | +0.0176 |

This includes two jets missing Emil's score; the [comparison with Emil](../../README.md#latest-test-on-run-484258)
uses 40 unpaired jets. Times are medians of three warmed-up CPU runs with
four threads, excluding file reading.

Overfitting means improving on training data while doing worse on new data.
After epoch 8, ParT's training loss fell while validation loss rose. Set
fluctuated but recovered. Cluster worsened for one epoch after its best;
that is too little to judge a trend. The gaps use weighted log loss at each
best-validation checkpoint. A gap alone is not proof. We need a new test run.

## Background reading

- [Deep Sets for particle jets](https://arxiv.org/abs/1810.05165), [ParticleNet](https://arxiv.org/abs/1902.08570), [ParT](https://arxiv.org/abs/2202.03772) and [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html): the model ideas.
- [ATLAS DeParT](https://cds.cern.ch/record/2878932/): another attention model; we have not reproduced it.
- [Simone's thesis](https://kth.diva-portal.org/smash/get/diva2%3A1985126/FULLTEXT01.pdf): BIB timing fits, PDF pages 46–52 and 57–64. Its cluster results use different samples from ours.

Local `reference/` also contains Emil's `QT_TGM.pdf` (parts I–II),
`LUP_Presentation_251002.pdf`, the jet-tagging slides `ATL-PHYS-SLIDE-2025-339.pdf`
and the review `2603.12306v1.pdf`. These files are not in Git.
