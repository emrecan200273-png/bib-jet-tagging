# BIB jet tagging

Can we tag beam-induced background (BIB) in ATLAS data while keeping jets
from long-lived particles (LLPs)?

## Where to look

| Folder | What's inside |
| --- | --- |
| [notebooks](notebooks/) | File checks, first plots and the XGBoost baseline |
| [analysis/data](analysis/data/) | ROOT reading, jet selection and vertex matching |
| [analysis/models](analysis/models/) | Cell models and [notes on the papers](analysis/models/README.md) |
| [analysis/training](analysis/training/) | XGBoost and cell-model training |
| [analysis/evaluation](analysis/evaluation/) | Metrics, kinematic bins and plots |
| [analysis/studies](analysis/studies/) | Comparisons and checks on the trained models |

The [analysis guide](analysis/README.md) lists the inputs and the order to run
the code. Start with [cell_jet_data.py](analysis/data/cell_jet_data.py) to see
how the cluster counts link cells to each jet.

## Latest test on run 484258

We matched scores by jet ID and used ATLAS jet time above 3 ns. Of the 42
delayed unpaired jets, 40 had valid scores from all three methods. The cuts
were chosen on delayed LLP validation jets to keep at least 95%.

| Method | Unpaired jets tagged as BIB | Simulated LLP jets kept |
| --- | ---: | ---: |
| Set Transformer | 40 / 40 | 779 / 816 |
| ParT | 40 / 40 | 782 / 816 |
| Stored `jet_BIB_score` | 9 / 40 | 782 / 816 |

In the smaller 3–10 ns subset with similar $p_T$ and $|\eta|$, ParT kept 11 of
16 LLP jets and the stored score kept 13. The next check is a new unpaired
run. Run `486706` still needs a cell export. See the [cell-model comparison](analysis/README.md#set-transformer-and-cluster-transformer)
for timing, loss gaps and Cluster Transformer.

## Set up

Use Python 3.12 and run from the project root.

```bash
python3.12 -m venv analysis/.venv
source analysis/.venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ppp-first-look --display-name "PPP first look"
python tools/notebook_git.py --install
```

Choose **PPP first look** as the notebook kernel. The last command keeps
notebook execution outputs on your computer when you commit.

The ROOT files and the verified manifest come from CERNBox/EOS. The default
paths are `DataForEmre/` and
`analysis/provenance/DataForEmre_verified_manifest_20261029.json`.
To point to a separate copy of the same data and manifest, set the paths below.

```bash
export PPP_DATA_ROOT=/path/to/DataForEmre
export PPP_DATA_MANIFEST=/path/to/verified_manifest.json
```

[Emil's GitLab](https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public) contains
the ntuple producer. We inspected [commit a7c625c](https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public/-/commit/a7c625c364893bbc90248fbb98a5251c1a54fdeb).
The production version of our ROOT files still needs confirmation.

## Run the notebooks and baseline

```bash
python tools/run_first_look.py 01_remote_data_inventory.ipynb
python tools/run_first_look.py 02_first_data_look.ipynb
python -m analysis.training.train_xgboost
python tools/run_first_look.py 03_xgboost_baseline.ipynb
python -m unittest discover -s tests
```

Results go in `analysis/outputs/`. The first look uses up to 1,000 rows per LLP
file and 6 per unpaired file. The baseline verifies the file hashes and reads
the selected datasets in full. Set `RETRAIN = True` in notebook 3 to rebuild it.
