# Analysis

See the [project README](../README.md) for setup, data paths and the reference to Emil’s GitLab.

- [Notebook 1](01_remote_data_inventory.ipynb) checks file metadata and stored variables.
- [Notebook 2](02_first_data_look.ipynb) reads a small sample and makes the first plots.
- `first_look.py` reads and checks the data; `first_look_plots.py` makes the figures.
- `data_paths.py` lets the data and private manifest stay outside the repository.

Raw data, local results and the private transfer records are ignored by Git. Locally executed notebooks keep their outputs; only notebook source is stored in Git.
