# BIB jet tagging

This project tests whether machine learning can tag beam-induced background
(BIB) in ATLAS jets while keeping jets from long-lived particles (LLPs).

- [Notebook 1](analysis/01_remote_data_inventory.ipynb) checks the input files.
- [Notebook 2](analysis/02_first_data_look.ipynb) plots a small sample.
- [Notebook 3](analysis/03_xgboost_baseline.ipynb) shows the XGBoost baseline.
- The [analysis guide](analysis/README.md) covers the cell models and checks.

## Setup

Use Python 3.12. From the project folder, run:

```bash
python3.12 -m venv analysis/.venv
source analysis/.venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ppp-first-look --display-name "PPP first look"
python tools/notebook_git.py --install
```

Choose **PPP first look** as the notebook kernel in VS Code. The Git filter
keeps notebook outputs on your computer and stores notebook source in Git.

## Data

The ROOT files and verified file list come from CERNBox/EOS. The default layout
uses `DataForEmre/` and
`analysis/provenance/DataForEmre_verified_manifest_20261029.json`. To use a local
copy or mounted EOS paths elsewhere, set both paths before running the analysis:

```bash
export PPP_DATA_ROOT=/path/to/DataForEmre
export PPP_DATA_MANIFEST=/path/to/verified_manifest.json
```

Use a manifest for the same version of the data. Raw data, private manifests,
plots and model outputs stay local.

[Emil's GitLab](https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public)
contains the code that writes the jet, cluster and cell variables, including
the BIB score, and small test files. We inspected
[commit a7c625c](https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public/-/commit/a7c625c364893bbc90248fbb98a5251c1a54fdeb).
The exact version used to produce the shared ROOT files still needs confirmation.

## Run

```bash
python analysis/run_first_look.py 01_remote_data_inventory.ipynb
python analysis/run_first_look.py 02_first_data_look.ipynb
python analysis/train_xgboost.py
python analysis/run_first_look.py 03_xgboost_baseline.ipynb
```

The first look reads up to 1,000 rows per LLP file and 6 per unpaired file.
XGBoost verifies the source hashes and reads the selected datasets in full.
Results go in `analysis/outputs/`. Notebook 3 uses the saved results when
available. Set `RETRAIN = True` in the notebook to run the training again.

The baseline uses $p_T$ ≥ 30 GeV, |η| < 1.7 and a Tile energy fraction between
0 and 1, excluding the endpoints. It takes truth-matched jets from the LLP
simulation and uses unpaired data as a BIB-enriched sample. All jets from an
event stay together in the 60/20/20 train/validation/test split. The score cut
targets 95% LLP retention on validation.

ROOT stores momenta in MeV, positions in mm and times in ns. The analysis uses
`jet_time_ATLAS` and the original stored cell times. See the output
`feature_schema.json` for the baseline inputs and cut.

```bash
python -m unittest discover -s tests
```
