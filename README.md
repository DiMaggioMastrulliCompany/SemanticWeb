# SemanticWeb

## Installation

Install `uv` by following the instructions at [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

uv is a fast Python package installer and resolver, written in Rust, designed to replace pip and virtualenv for managing dependencies and virtual environments.


## Running EDA

To run the Exploratory Data Analysis scripts:

### GraphWiz Dataset
```bash
uv run .\EDA\graphwiz.py
```
Plots will be saved to: `EDA/eda_plots_graphwiz/`

### NLGraph Dataset
```bash
uv run .\EDA\nlgraph.py
```
Plots will be saved to: `EDA/eda_plots_nlgraph/`
