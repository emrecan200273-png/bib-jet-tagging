# BIB jet tagging

We use machine learning to tag beam-induced background (BIB) in ATLAS while
keeping jets from long-lived particles (LLPs). BIB comes from beam interactions
with gas or material and can look like a delayed jet.

## Start here

| Notebook | What it does |
| --- | --- |
| [1. Check the files](notebooks/01_remote_data_inventory.ipynb) | Lists the data folders and reads the variables in a sample file. |
| [2. First plots](notebooks/02_first_data_look.ipynb) | Checks a small sample and compares jet distributions. |
| [3. XGBoost baseline](notebooks/03_xgboost_baseline.ipynb) | Explains the first model and its saved results. |

[Model guide](analysis/models/README.md) · [Data paths and commands](analysis/README.md)

## Latest test on run 484258

All methods use the **same jets with ATLAS time above 3 ns**. Score cuts target
95% LLP retention on validation. Two unpaired jets lack Emil's score, leaving 40.

| Method | Unpaired jets tagged as BIB | Simulated LLP jets kept |
| --- | ---: | ---: |
| Set Transformer | 40 / 40 | 779 / 816 |
| ParT-inspired model | 40 / 40 | 782 / 816 |
| Emil's stored `jet_BIB_score` | 9 / 40 | 782 / 816 |

This small test has been reused, and unpaired jets are not all confirmed BIB.
At similar momentum and direction, weighted LLP retention was 63% for ParT
and 93% for Emil (48 LLP and 24 unpaired jets). The table hides that difference.
Run `486706` still needs cell data for an independent test.

## Setup

Use Python 3.12. Run from this folder:

```bash
python3.12 -m venv analysis/.venv
source analysis/.venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ppp-first-look --display-name "PPP first look"
python tools/notebook_git.py --install
jupyter lab notebooks/
```

Choose **PPP first look** as the kernel. Notebook outputs, data, saved models
and generated plots stay local.

You need the CERNBox/EOS ROOT files in `DataForEmre/` and the verified file list
at `analysis/provenance/DataForEmre_verified_manifest_20261029.json`.
For other paths, set `PPP_DATA_ROOT` and `PPP_DATA_MANIFEST` before starting
Jupyter; see [.env.example](.env.example).

## Code layout

| Folder | Purpose |
| --- | --- |
| [data](analysis/data/) | Read ROOT files, select jets and match vertex counts. |
| [models](analysis/models/) | Define the neural networks. |
| [training](analysis/training/) | Train models and save scores. |
| [evaluation](analysis/evaluation/) | Choose score cuts and calculate metrics. |
| [studies](analysis/studies/) | Compare models and check possible biases. |
| [tests](tests/) | Check the readers, models and evaluation code. |

Run the tests with `python -m unittest discover -s tests`.

[Emil's GitLab](https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public)
contains the ROOT export code. We inspected
[commit a7c625c](https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public/-/commit/a7c625c364893bbc90248fbb98a5251c1a54fdeb);
the version that produced our files still needs confirmation.
