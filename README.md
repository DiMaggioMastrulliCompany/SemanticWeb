# SemanticWeb: Agentic Graph Reasoning

A research project involving Multi-Agent Large Language Model systems for solving computational graph reasoning problems. This project implements and compares various agent architectures.

## Repository Structure

The codebase is organized into distinct modules, each with its own responsibilities:

- **[`project/`](./project/)**: **The Core System**. Contains the LangGraph implementation of the Multi-Agent solvers (Backtracking & Consensus), along with the evaluation suite.
- **[`Preprocessing/`](./Preprocessing/)**: Scripts for loading, cleaning, and sanitizing the GraphWiz and NLGraph datasets.
- **[`EDA/`](./EDA/)**: Exploratory Data Analysis. Scripts and visualizations to understand the dataset distributions (task types, graph sizes...).
- **[`Tools introduction/`](./Tools%20introduction/)**: Interactive notebooks introducing the foundational technologies.

## Installation

This project uses `uv` for fast and reliable dependency management.

1. **Install uv**: Follow instructions at [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
2. **Sync Dependencies**:
   ```bash
   uv sync
   ```

## Getting Started

1. **Configure the Environment**:
   This project uses **OpenRouter** to access models like *Mistral* and *Llama 3.1* easily. You need to set your API key.
   
   Create a `.env` file in the `project/` directory:
   ```bash
   echo "OPENAI_API_KEY=sk-or-..." > project/.env
   ```
   *(Note: The `OPENAI_API_KEY` variable name is used for compatibility with LangChain/LangGraph's default `ChatOpenAI` client, even though we are using OpenRouter endpoints.)*

2. **Run the Demo**:
   To immediately see the system in action, you can run the main project demo:

   ```bash
   uv run project/langgraph_multi_agent.py
   ```

For detailed instructions on running evaluations, data analysis, or tutorials, please refer to the **README** files in each subdirectory:
- [**Project & Evaluation Instructions**](./project/README.md)
- [**Preprocessing Details**](./Preprocessing/README.md)
- [**EDA Instructions**](./EDA/README.md)
- [**Tools & Notebooks**](./Tools%20introduction/README.md)

## Reproducibility

We have taken strict measures to ensure the results of this project are reproducible:
- **Dependency Locking**: `uv.lock` ensures that all Python packages are installed at the exact same versions used during development.
- **Data Splitting**: The `Preprocessing` module uses fixed random seeds to ensure the Train/Test splits are consistent for every user.
- **Evaluation Protocols**: The `project/test_all_tasks.py` script runs on a deterministic subset of the test data, logging all inputs and outputs to text files for auditability.
