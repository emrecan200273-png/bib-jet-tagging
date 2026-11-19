# BIB jet tagging

I use machine learning to tag beam-induced background (BIB) in ATLAS while
keeping jets from long-lived particles (LLPs). I use unpaired bunch crossings
as the background sample and simulated LLP jets as the signal.

| Notebook | What I look at |
| --- | --- |
| [01 · Cells and selections](notebooks/01_cells_and_selections.ipynb) | Follow one jet from its clusters to its cells and check its timing fit. |
| [02 · XGBoost baseline](notebooks/02_xgboost_baseline.ipynb) | Work through a tree split and train with jet variables. |
| [03 · Train and compare](notebooks/03_train_and_compare.ipynb) | Train four models and compare ParT with Emil's timing score. |
| [04 · BIB checks](notebooks/04_bib_checks.ipynb) | Check which runs have cells and how the cuts change the sample. |

In the weighted ROC scan, ParT tags **40/44** delayed unpaired test jets
and Emil's score tags **13/44** at about 95% weighted LLP retention.
Notebook 03 also shows the results with cuts fixed on validation.

## Run locally

Use Python 3.12. From the project folder:

```bash
python3.12 -m venv analysis/.venv
source analysis/.venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ppp-first-look --display-name "PPP first look"
jupyter lab notebooks/
```

Choose **PPP first look** in VS Code or Jupyter. Put the ROOT files in
`DataForEmre/` with their original folder structure. The notebooks read
those files and write tables, plots and weights under `analysis/outputs/`.

The ROOT export code and timing study are described in
[Emil's GitLab](https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public)
and [Simone Gallareto's thesis, sections 7–9](https://kth.diva-portal.org/smash/get/diva2%3A1985126/FULLTEXT01.pdf).
