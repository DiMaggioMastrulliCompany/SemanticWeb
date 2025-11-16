# Preprocessing

This directory contains the preprocessing pipeline for preparing graph reasoning datasets used in the multi-agent debate system.

## Overview

The preprocessing module handles two main datasets:
- **GraphWiz/GraphInstruct**
- **NLGraph**

The goal is to clean and structure these datasets for training and testing.

## Dataset Statistics

After preprocessing, the datasets contain:

| Dataset | Train Examples | Test Examples |
|---------|---------------|---------------|
| GraphWiz | 14,432 | 3,609 |
| NLGraph | 4,821 | 961 |

## Preprocessing Steps

### GraphWiz Dataset

The GraphWiz preprocessing performs the following operations:

1. **Loading**: Loads the `GraphWiz/GraphInstruct` dataset from HuggingFace
2. **Answer Extraction**: Extracts answers using regex pattern matching (looking for content after "###" marker)
3. **Filtering**: Removes examples without clear answers (e.g., incomplete reasoning traces)
4. **Train/Test Split**: Applies an 80/20 split with shuffle (using fixed seed for reproducibility)

**Why filtering?** Some rows contain only reasoning steps without a final answer, making them unsuitable for supervised learning.
For example, there are entries like:

```
- Dequeue 31, current = 31
- 31 is not 2, so check its neighbors: 1, 3, 16, 33
- Enqueue 1, 3, 16, 33 and add them to visited, queue = [1, 3, 16, 33], visited = {31, 1, 3, 16, 33}

...

- Dequeue 32, current = 32
- 32 is not 2, so check its neighbors: 0, 4, 21
- 0, 4, 21 are already in visited, so skip them
- Dequeue 19, current = 19
```

The answer ends here abruptly without a definitive solution.

### NLGraph Dataset

The NLGraph preprocessing is minimal:
- After inspection, we determined that the dataset already contains clean answers
- It was pre-split by the creator into train/test sets using about an 80/20 ratio
- So it is returned without much additional processing

## Usage

Run the preprocessing script:

```bash
uv run .\Preprocessing\preprocess.py
```

This will load both datasets and display statistics about the number of examples in each split.
