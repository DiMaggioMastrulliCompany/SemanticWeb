# Exploratory Data Analysis (EDA)

This directory contains scripts to generate visualizations and insights for the GraphWiz and NLGraph datasets.

## Contents

- `graphwiz.py`: Analysis script for the GraphWiz/GraphInstruct dataset.
- `nlgraph.py`: Analysis script for the NLGraph dataset.
- `eda_plots_graphwiz/`: Generated plots for GraphWiz.
- `eda_plots_nlgraph/`: Generated plots for NLGraph.

## Running the Analysis

Ensure you have installed the project dependencies using `uv`.

### GraphWiz Analysis
```bash
uv run .\EDA\graphwiz.py
```
This will generate distribution plots for node counts, edge counts, and task types in `eda_plots_graphwiz/`.

### NLGraph Analysis
```bash
uv run .\EDA\nlgraph.py
```
This will generate similar statistics and plots for the NLGraph dataset in `eda_plots_nlgraph/`.
