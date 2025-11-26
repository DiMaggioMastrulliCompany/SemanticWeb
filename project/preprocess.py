import random
import re

from datasets import load_dataset
from nlgraph_loader import load_nlgraph

random.seed(42)  # For reproducibility


def extract_answer_graphwiz(query: str):
    # Extract the answer following the "###" marker
    match = re.search(r"###\s*(.*)", query)
    if match:
        return match.group(1)
    return None


def preprocess_graphwiz(split_ratio=0.8):
    # There are a few rows without answers, just reasoning, so we filter them out
    # E.g. - Initialize queue = [31] and visited = {31}
    # - Dequeue 31, current = 31
    # - 31 is not 2, so check its neighbors: 1, 3, 16, 33
    # - Enqueue 1, 3, 16, 33 and add them to visited, queue = [1, 3, 16, 33], visited = {31, 1, 3, 16, 33}
    #
    # ...
    #
    # - Dequeue 32, current = 32
    # - 32 is not 2, so check its neighbors: 0, 4, 21
    # - 0, 4, 21 are already in visited, so skip them
    # - Dequeue 19, current = 19

    # As we can see, there isn't a clear answer

    graphwiz_dataset = load_dataset("GraphWiz/GraphInstruct")

    examples = []
    for example in graphwiz_dataset["train"]:
        ds_answer = example["answer"]  # pyright: ignore[reportArgumentType, reportCallIssue]
        answer = extract_answer_graphwiz(ds_answer)
        if answer:
            examples.append(example)

    # Splitting
    random.shuffle(examples)
    split_index = int(len(examples) * split_ratio)
    examples = {"train": examples[:split_index], "test": examples[split_index:]}

    return examples


def preprocess_nlgraph():
    dataset = load_nlgraph()
    # This dataset has already clean answers
    # It is also already split into train and test from the loader
    # So we just return it as is with no further processing
    return dataset


if __name__ == "__main__":
    graphinstruct = preprocess_graphwiz()
    nlgraph = preprocess_nlgraph()
    print(f"GraphWiz train examples: {len(graphinstruct['train'])}")
    print(f"GraphWiz test examples: {len(graphinstruct['test'])}")
    print(f"NLGraph train examples: {len(nlgraph['train'])}")
    print(f"NLGraph test examples: {len(nlgraph['test'])}")
