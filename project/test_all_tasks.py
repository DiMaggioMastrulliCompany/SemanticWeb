
import sys
import os
from datasets import load_dataset
from langgraph_multi_agent import build_backtracking_solver, GraphState, verify_answer
import time

from preprocess import preprocess_graphwiz

def main():
    results_file = "test_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Loading dataset...\n")
        ds = preprocess_graphwiz()["test"]

        task_types = [
            'connectivity',
            'cycle',
            'flow',
            'bipartite',
            'hamilton',
            'triangle',
            'shortest',
            'topology',
            'substructure'
        ]

        # Collect examples
        examples_by_task = {t: [] for t in task_types}
        target_count = 5

        f.write(f"Collecting {target_count} examples for each task type...\n")

        # Iterate through dataset to find enough examples
        # This might be slow if the dataset is huge, but GraphInstruct is manageable
        count_found = 0
        for ex in ds:
            t = ex.get("task")
            if t in task_types and len(examples_by_task[t]) < target_count:
                examples_by_task[t].append(ex)

            if all(len(examples_by_task[t]) >= target_count for t in task_types):
                break

        # Build the solver with a higher recursion limit
        app = build_backtracking_solver(recursion_limit=100)

        # Summary stats
        stats = {t: {"correct": 0, "total": 0, "errors": 0} for t in task_types}

        for task in task_types:
            f.write(f"\n\n{'='*60}\n")
            f.write(f"TESTING TASK TYPE: {task}\n")
            f.write(f"{'='*60}\n")

            examples = examples_by_task[task]
            if not examples:
                f.write(f"No examples found for task: {task}\n")
                continue

            for i, example in enumerate(examples):
                f.write(f"\n--- Example {i+1}/{len(examples)} ---\n")
                # f.write(f"Query: {example['query'][:100]}...\n")
                f.write(f"Expected Answer: {example['answer']}\n")

                initial_state: GraphState = {
                    "query": str(example["query"]),
                    "task_type": str(example["task"]),
                    "partial_solution": "",
                    "forbidden_moves": "",
                    "current_proposal": "",
                    "verifier_feedback": "",
                    "attempt_count": 0,
                    "final_output": "",
                    "status": "SEARCHING",
                }

                try:
                    final_state = None
                    # Run the graph
                    # We use a simple loop. The recursion limit in LangGraph handles infinite loops.
                    for step in app.stream(initial_state):
                        for node_name, node_state in step.items():
                            final_state = node_state

                    if final_state and "final_output" in final_state:
                        prediction = final_state['final_output']
                        ground_truth = example['answer']

                        is_correct = verify_answer(prediction, ground_truth, task)

                        f.write(f"Prediction: {prediction}\n")
                        f.write(f"Verification: {'CORRECT' if is_correct else 'INCORRECT'}\n")

                        stats[task]["total"] += 1
                        if is_correct:
                            stats[task]["correct"] += 1
                    else:
                        f.write("Prediction: No final output captured.\n")
                        f.write("Verification: ERROR\n")
                        stats[task]["total"] += 1
                        stats[task]["errors"] += 1

                except Exception as e:
                    f.write(f"Error running solver: {e}\n")
                    stats[task]["total"] += 1
                    stats[task]["errors"] += 1

                f.flush()

        # Write Summary
        f.write(f"\n\n{'='*60}\n")
        f.write("FINAL SUMMARY STATISTICS\n")
        f.write(f"{'='*60}\n")
        for task, data in stats.items():
            acc = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
            f.write(f"{task}: {data['correct']}/{data['total']} ({acc:.1f}%) - Errors: {data['errors']}\n")

if __name__ == "__main__":
    main()
