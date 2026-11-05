# BIB jet tagging

This is my analysis project for studying whether ML can tag beam-induced
background (BIB) in ATLAS jet data while keeping jets from long-lived particles
(LLPs).

I have checked the files, made the first plots, and trained a first XGBoost
baseline. The model study includes event-separated tests and checks for sample
differences. It is an initial comparison, not a finished BIB tagger.

## Code and data

GitHub stores the analysis code and notebook source. The ROOT data, private
file lists, plots and notebook outputs stay outside Git. Executed notebooks
can still keep their results on the computer where they are run.

- [Notebook 1](analysis/01_remote_data_inventory.ipynb): check the files and variables.
- [Notebook 2](analysis/02_first_data_look.ipynb): read a small sample and make plots.
- [Notebook 3](analysis/03_xgboost_baseline.ipynb): train XGBoost and inspect its limits.
- [Analysis functions](analysis/first_look.py) and [plot functions](analysis/first_look_plots.py).

The data are supplied separately through CERNBox/EOS. Emil's GitLab contains
small test fixtures, not the full `DataForEmre` dataset. Its
[CI notes](https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public/-/blob/main/ci/README.md)
describe 16 data events and 20 MC events used for automated tests.

## Reference to Emil's code

The upstream ATLAS ntuple producer is
[Emil Zaya's AthJetConstituentsAlg-public](https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public).
It defines the stored jet, cluster and cell variables, timing methods, LLP
matching and existing BIB scores. This repository contains our analysis of
those outputs; the upstream source is referenced rather than copied.

Inspected upstream version:
[`a7c625c`](https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public/-/commit/a7c625c364893bbc90248fbb98a5251c1a54fdeb).
The production version used for our ROOT files still needs confirmation.

Meeting clarifications: energy/momentum are stored in MeV, positions in mm and
times in ns. Start from `jet_time_ATLAS` and leave the stored cell times alone.
Unpaired is the initial BIB training label, with possible contamination. The
first model uses `jet_is_llp_matched == 1` to select signal jets, following the
inspected upstream definition. The exact production version and truth-matching
definition still need confirmation with Emil.

## Set up

Use Python 3.12. From the project folder:

```bash
python3 -m venv analysis/.venv
source analysis/.venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ppp-first-look --display-name "PPP first look"
python3 tools/notebook_git.py --install
```

The last command installs a local Git filter. It removes outputs from the
version stored in Git while leaving the executed notebook on disk intact.
Run it once after cloning, before committing notebook changes.

In VS Code, open a notebook and choose the **PPP first look** kernel.
JupyterLab can be started with `jupyter lab`.

## Point the code to the data

Existing local copies keep working with this layout:

```text
DataForEmre/                         # private ROOT data, ignored by Git
analysis/provenance/
  DataForEmre_verified_manifest_20261029.json
```

On another machine, provide your authorized data and the private verified
transfer manifest separately. You can point to them without copying the
ROOT data into the project:

```bash
export PPP_DATA_ROOT=/path/to/DataForEmre
export PPP_DATA_MANIFEST=/path/to/private/verified_manifest.json
```

These paths can point to EOS folders when running in a CERN environment where
they are mounted and your account has access. A laptop needs its own data
access or local copy; cloning this repository does not provide CERN access.
The manifest's file sizes and hashes must correspond to that data version.

The notebooks open existing files through Uproot. They do not download or
update the dataset. When Emil replaces the data, use the corresponding new
file list as well.

## Run the first look

```bash
python analysis/run_first_look.py 01_remote_data_inventory.ipynb
python analysis/run_first_look.py 02_first_data_look.ipynb
```

Results are written to `analysis/outputs/`. The first look reads the first
1,000 rows of each selected LLP file and the first 6 of each selected unpaired
file. It is a small, non-random check, not a training or physics evaluation.

GitHub holds the shared version of the code. Calculations run locally or in a
CERN environment such as SWAN/lxplus, where the data can be accessed.

## First model

```bash
python analysis/train_xgboost.py
python analysis/run_first_look.py 03_xgboost_baseline.ipynb
```

The first command checks source hashes, reads the full selected LLP and
unpaired groups, trains the models and writes results under
`analysis/outputs/xgboost_baseline/`. It reads only the needed jet branches;
raw files and stored cell times are unchanged. The notebook opens the saved
results, or trains if results do not exist yet. Set `RETRAIN = True` there
to rerun the study.

Both classes use pT >= 30 GeV, |eta| < 1.7 and a Tile energy fraction strictly
between 0 and 1. This common mixed-energy region is motivated by the LAr/Tile
cell requirement in LLP matching. All jets from one event stay together in
the 60/20/20 training/validation/test split. Score cuts target 95% LLP retention
on validation, and are then held fixed for testing.

The main inputs are ATLAS jet time, cell-time spread, timing uncertainty,
cluster count and Tile energy fraction. Timing-only and kinematics-only
controls check what drives separation. A second, exploratory study balances
pT/eta bins inside the original splits and also removes timing uncertainty
as a feature check. This changes the sample being evaluated; its results are
reported separately. The delayed-time region is checked explicitly. Strong
overall separation does not establish delayed-BIB rejection, ordinary-collision
efficiency or generalization to another run or LLP model.

Saved models use XGBoost's `.ubj` format. Each main model has a
`feature_schema.json` recording its inputs, units, acceptance and validation
cut. `score_saved_model` in `analysis/train_xgboost.py` loads either the original
or balanced-check model and leaves out-of-scope jets unscored. Scores are not
calibrated physical probabilities. Models, predictions, private source records
and executed notebook outputs stay outside Git.

The detailed local report is `analysis/outputs/xgboost_baseline/run_notes.md`.
It records the sample limitations and the delayed-region result, including
small counts. Further model development needs a new independent test sample.

Run the checks with:

```bash
MPLCONFIGDIR=analysis/.cache/matplotlib python -m unittest discover -s tests -v
```
