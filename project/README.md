# Graph Reasoning Multi-Agent System

This directory contains the core implementation of the Agentic Graph Reasoning System. It leverages `LangGraph` to orchestrate multiple LLM agents (Proposers, Verifiers, Arbiters) to solve complex graph tasks.

## Features

The system implements two distinct multi-agent architectures to approach graph reasoning tasks:

### 1. Backtracking Architecture (Single-Threaded Search)
A first approach where a single **Proposer** generates steps, a **Verifier** checks strict logical validity, and a **Manager** maintains the search stack. If a path fails or is invalid, the Manager instructs the Proposer to **backtrack** and try a different route, leading to an exploration of the solution space.

### 2. Consensus Architecture (Multi-Agent Debate)
A second approach leveraging diversity. Three distinct **Proposers** (configured with BFS, DFS, and Heuristic strategies) generate competing next steps. An **Arbiter** evaluates these proposals based on progress and consistency, selecting the best one to proceed.

## Directory Structure

- `langgraph_multi_agent.py`: **Main Logic**. Defines the `GraphState`, single Agents (`proposer`, `verifier`, `arbiter`, `manager`), and the two main architectures (`build_backtracking_solver`, `build_consensus_solver`).
- `test_all_tasks.py`: **Evaluation Suite**. Runs the solvers against the GraphWiz test set, producing detailed logs and performance plots.
- `results-*/`: Directories containing output artifacts from evaluation runs.
- `preprocess.py` & `nlgraph_loader.py`: Local helpers for loading data within the project scope.

## Installation

Ensure you are in the project root and have dependencies installed:
```bash
uv sync
```

### Configuration

Create a `.env` file in this directory (`project/.env`) containing your OpenRouter API key:
```env
OPENAI_API_KEY=sk-or-your-key-here
```
Usage of OpenRouter allows seamless switching between **Mistral** and **Llama** models without code changes.

## Usage

### 1. Run a Single Demo
To see the system solve a single Hamiltonian path problem with the Backtracking solver:
```bash
uv run project/langgraph_multi_agent.py
```

### 2. Run the Evaluation Suite
To assess the performance of the system on a larger set of tasks, use `test_all_tasks.py`. This script ensures reproducibility by testing a fixed subset of the test split.

**Arguments:**
- `--solver`: `backtracking` (Single Agent) or `consensus` (Multi-Agent Debate).
- `--model`: `mistral` or `llama`.

**Example: Run Consensus Solver with Llama 3.1**
```bash
uv run project/test_all_tasks.py --solver consensus --model llama
```

**Example: Run Backtracking Solver with Mistral**
```bash
uv run project/test_all_tasks.py --solver backtracking --model mistral
```

The script will:
1. Load the test dataset.
2. Run the selected solver on the first 200 examples.
3. Log all interactions to `test_results.txt`.
4. Generate performance plots:
   - `results_task_percentages.png`: Correctness per task type.
   - `results_task_distribution.png`: Distribution of tasks in the sample.

## Reproducibility

We ensure reproducibility through several layers:
1. **Environment**: Managed via `uv.lock` to guarantee exact package versions.
2. **Data**: The `Preprocessing` module uses a fixed random seed when splitting the GraphWiz dataset, ensuring the "Test" set is identical across runs.
3. **Evaluation**: `test_all_tasks.py` deterministically iterates through the dataset.
