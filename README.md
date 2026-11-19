# BIB jet tagging

ATLAS times each hit relative to light travelling from the collision point. A slow long-lived particle (LLP) can decay on the way and leave a late jet. Background entering along the beam takes a different path and can look late too. Can the pattern of cell times tell them apart?

I use simulated jets matched to LLP decays for signal. For beam-induced background (BIB), I use *unpaired* crossings, where only one beam has a bunch. They're rich in BIB, though individual jets can have other origins.

## The notebooks

| Notebook | What I wanted to find out |
| --- | --- |
| [01 · Cells and selections](notebooks/01_cells_and_selections.ipynb) | Which cells belong to one jet, and how does Emil turn their times into a score? |
| [02 · XGBoost baseline](notebooks/02_xgboost_baseline.ipynb) | How far can I get with five numbers per jet? |
| [03 · Train and compare](notebooks/03_train_and_compare.ipynb) | Does giving the models individual cells help? How does ParT compare with Emil's score on the same jets? |
| [04 · BIB checks](notebooks/04_bib_checks.ipynb) | Why do so few delayed unpaired jets survive the cuts, and which runs have cell data? |

## What I found

I picked each cut to keep about 95% of the LLP weight on validation. Here's what happened on the same delayed test jets (ATLAS time above 3 ns).

| Method | Unpaired tagged | LLP kept | LLP kept after weighting |
| --- | ---: | ---: | ---: |
| Small ParT-inspired model | 41/44 | 792/803 | 84.7% |
| Emil's timing score | 5/44 | 784/803 | 97.1% |

I weight the jets so the two samples have similar momentum and direction. ParT tagged more unpaired jets. The LLP jets it lost carried enough weight to pull retention down to 84.7%.

On the test ROC curve I can compare the methods at the same retention. At about 95%, ParT tagged **40/44** and Emil's score tagged **13/44**. All 44 unpaired jets came from run 484258. I'd like to repeat this on a second run.

## Run it in VS Code

Put the CERN ROOT files in `DataForEmre/` with the same folder names as on CERNBox. From the project folder, run this once with Python 3.12.

```bash
python3.12 -m venv analysis/.venv
source analysis/.venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ppp-first-look --display-name "PPP first look"
```

Open a notebook, select **PPP first look** and run the cells from the top. The training notebooks save each run in a new folder. [Data folders and saved results](analysis/README.md)

Emil's code shows how the ROOT files were made. Simone's thesis explains the timing fit for clusters that I look at in notebook 01. [Emil's GitLab](https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public) · [Simone Gallareto's thesis, chapter 7](https://kth.diva-portal.org/smash/get/diva2%3A1985126/FULLTEXT01.pdf)
