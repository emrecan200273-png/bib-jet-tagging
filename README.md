# BIB jet tagging

This project compares methods for identifying beam-induced background (BIB) in ATLAS while retaining delayed jets from long-lived particle (LLP) decays. Both can produce a late jet. The arrival times and positions of individual calorimeter cells provide additional information about its origin.

## Timing in the calorimeter

ATLAS subtracts the time light would take to travel from the collision point to each cell. For a simple example, consider an LLP that travels a distance $L$ at speed $\beta c$ before decaying. If its decay products continue along the same line at speed $c$, the delay relative to a prompt particle is

$$
\Delta t = \frac{L}{\beta c}-\frac{L}{c}
         = \frac{L}{c}\left(\frac{1}{\beta}-1\right).
$$

For $L=1$ m and $\beta=0.5$, the delay is about **3.3 ns**. A BIB particle travelling along the beam produces a timing pattern that varies with position along the beam axis. Assigning its hits to an adjacent bunch crossing shifts their recorded times by 25 ns. The pattern across cells can therefore help distinguish BIB from an LLP decay even when the jet times are similar.

Simone Gallareto compared two timing fits to the cells of each **cluster**. One assumes a common arrival time after the flight-time correction. The other uses the time expected for a particle travelling along the beam. Emil's score applies this comparison to the cells of a **jet** and serves as the reference for the machine-learning models. [Simone's thesis, chapter 7](https://kth.diva-portal.org/smash/get/diva2%3A1985126/FULLTEXT01.pdf#page=45) · [Emil's ROOT export code](https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public)

## Notebooks

The figures and captions are visible directly on GitHub.

| Notebook | Contents |
| --- | --- |
| [01 · Cells and selections](notebooks/01_cells_and_selections.ipynb) | Cell positions, projected decay products and both timing fits for one simulated LLP jet. |
| [02 · XGBoost baseline](notebooks/02_xgboost_baseline.ipynb) | A worked tree split and a classifier trained on five jet variables. |
| [03 · Model comparison](notebooks/03_train_and_compare.ipynb) | Jet MLP, Deep Sets, Set Transformer and a small ParT-inspired model. The final figure compares ParT with Emil's score on the same delayed jets. |
| [04 · BIB checks](notebooks/04_bib_checks.ipynb) | Availability of cell data by run and the effect of the momentum and calorimeter cuts on the delayed sample. |

## Results on delayed jets

The signal sample contains simulated jets matched to an LLP decay. The background sample comes from *unpaired* crossings, where only one beam has a bunch. These crossings are enriched in BIB, but individual jets can have other origins.

Each method has a score threshold chosen to retain about 95% of the LLP weight on validation. The table shows what happens when those thresholds are applied to the same test jets with ATLAS jet time above 3 ns.

| Method | Unpaired jets tagged | LLP jets retained | Weighted LLP retention |
| --- | ---: | ---: | ---: |
| Small ParT-inspired model | 41/44 | 792/803 | 84.7% |
| Emil's timing score | 5/44 | 784/803 | 97.1% |

The weights balance the two samples within bins of transverse momentum and direction. The 11 LLP jets rejected by ParT carry 15.3% of the total LLP weight, which explains the difference between the raw count and weighted retention.

The test ROC curve also allows a comparison at the same weighted LLP retention on **test**. At about 95%, ParT tags **40/44** delayed unpaired jets and Emil's score tags **13/44**. All 44 come from run 484258. Testing on a second run requires another unpaired sample with cell data.

## Running the notebooks in VS Code

Place the CERN ROOT files under `DataForEmre/` using the folder names from CERNBox. From the project folder, create the Python 3.12 environment and register its notebook kernel.

```bash
python3.12 -m venv analysis/.venv
source analysis/.venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ppp-first-look --display-name "PPP first look"
```

Open a notebook, select **PPP first look** and run the cells in order. Each training run writes to a separate folder. [Data folders and saved results](analysis/README.md)
