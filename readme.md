# FYS5429 Project Code

This repository contains the code used for my FYS5429 project. The project studies parametric matrix models, compression methods, neural network baselines, and time evolution for Lipkin and pairing model systems.

 The Python files contain shared model, data, training, compression, benchmark, and plotting support code. The notebooks are used for running experiments and producing the figures and tables used in the report.

## How To Use The Repository

Run the results notebook before running the plotting notebook, i.e. 

1. Run `lipkin_results.ipynb`
2. Run `pairing_results.ipynb`
3. Run `time_evolution_results.ipynb`
4. Run `plots.ipynb`

The first three notebooks compute and save numerical results. The plotting notebook loads those saved results and generates the report figures and tables.

Below is a summary of what each code files does. 

## Main Notebooks

### `lipkin_results.ipynb`

Runs the Lipkin model experiments and saves the numerical results used later by the plotting notebook.

### `pairing_results.ipynb`

Runs the static pairing model experiments and saves the corresponding numerical results.

### `time_evolution_results.ipynb`

Runs the pairing quench time evolution experiments, including exact, compressed exact, PMM, and sequence baseline results.

### `plots.ipynb`

Loads the saved result files and creates the final figures and tables for the report.

## Python Files

### `common.py`

Contains shared constants, random seed setup, scaling utilities, and common error metric functions.

### `data.py`

Contains data generation helpers for the Lipkin and pairing model experiments.

### `models.py`

Contains the exact model definitions and Hamiltonian construction used in the project.

### `pmm.py`

Contains the parametric matrix model implementation, training helpers, prediction helpers, and compressed PMM utilities.

### `compression.py`

Contains routines for building compression bases and applying reduced model calculations.

### `benchmarks.py`

Contains runtime measurement helpers, speedup and break even calculations.

### `baselines.py`

Contains neural network and recurrent baseline models, including training and grid search helpers.

### `time_evolution.py`

Contains exact and PMM based time evolution functions for survival probabilities and observables.



