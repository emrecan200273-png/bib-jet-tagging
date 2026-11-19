# BIB jet tagging

I'm trying to tell two kinds of late jets apart. A long-lived particle (LLP) can travel some distance before it decays. Beam-induced background (BIB) can also leave a late jet, even though the particle that made it was moving close to the speed of light.

## Why a jet can look late

ATLAS subtracts the flight time expected for light from the collision point to each cell. Suppose an LLP travels a distance $L$ at speed $\beta c$, then its decay products continue along the same line at speed $c$. The delay comes from the first part of the trip,

$$
\Delta t = \frac{L}{\beta c}-\frac{L}{c}
         = \frac{L}{c}\left(\frac{1}{\beta}-1\right).
$$

For $L=1$ m and $\beta=0.5$, that's about **3.3 ns**. A BIB particle travels along the beam instead. Its cell times change with position along the beam, and assigning its hits to a different 25 ns bunch crossing can shift the whole pattern. That's why I look inside the jet even when I already know its time.

Simone Gallareto compared a flat time with the beam timing pattern for the cells in each **cluster**. Emil's score uses the cells of a whole **jet**. I use his score as a physics reference, then ask what the models can learn from those same jets. [Simone's thesis, chapter 7](https://kth.diva-portal.org/smash/get/diva2%3A1985126/FULLTEXT01.pdf#page=45) · [Emil's ROOT export code](https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public)

I use simulated jets matched to LLP decays for signal. For BIB, I use *unpaired* crossings, where only one beam has a bunch. They're rich in BIB, though individual jets can have other origins.

## The notebooks

Open any notebook on GitHub to see the saved figures and the notes underneath. I'd start with the example jet in 01, then look at the last plot in 03 to compare ParT with Emil's score.

| Notebook | What I wanted to find out |
| --- | --- |
| [01 · Cells and selections](notebooks/01_cells_and_selections.ipynb) | Which cells belong to one jet, and how does Emil turn their times into a score? |
| [02 · XGBoost baseline](notebooks/02_xgboost_baseline.ipynb) | How does a tree choose a cut? How far can I get with five numbers per jet? |
| [03 · Train and compare](notebooks/03_train_and_compare.ipynb) | What changes from a jet MLP to Deep Sets, a Set Transformer and a small ParT-inspired model? I compare ParT with Emil on the same jets. |
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
