
from langgraph_multi_agent import build_backtracking_solver, GraphState, verify_answer

from preprocess import preprocess_graphwiz
import matplotlib.pyplot as plt
from collections import defaultdict

# Per-task recursion limits tuned from results_task_percentages_with_errors.png
# Note: LangGraph's `recursion_limit` counts node transitions. One
# proposer->verifier->manager loop is ~3 transitions. Larger values
# allow deeper searches for combinatorial tasks like Hamilton/substructure.
TASK_STEP_LIMITS = {
    # Deep, combinatorial search
    "hamilton": 300,
    "substructure": 240,
    # Moderate depth graph reasoning
    "cycle": 180,
    "flow": 200,
    "shortest": 180,
    "topology": 140,
    # Generally simpler; cap to reduce error loops observed
    "connectivity": 120,
    "bipartite": 90,
    "triangle": 90,
}

DEFAULT_RECURSION_LIMIT = 150

def main():
    results_file = "test_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Loading dataset...\n")
        ds = preprocess_graphwiz()["test"]

        # We'll test only the first N examples from the test set
        max_examples = 200

        f.write(f"Testing first {max_examples} examples from test set...\n")

        # Build the solver once
        app = build_backtracking_solver(recursion_limit=40)

        # Track stats per task. We keep counts and later convert to percentages.
        stats = defaultdict(lambda: {"correct": 0, "incorrect": 0, "errors": 0, "total": 0})

        processed = 0
        for idx, example in enumerate(ds):
            if processed >= max_examples:
                break

            task = example.get("task", "unknown")

            f.write(f"\n--- Example {processed+1}/{max_examples} (idx={idx}) ---\n")
            f.write(f"Task: {task}\n")
            f.write(f"Query: {example.get('query')}\n")
            f.write(f"Expected Answer: {example.get('answer')}\n")

            initial_state: GraphState = {
                "query": str(example.get("query", "")),
                "task_type": str(task),
                "partial_solution": "",
                "forbidden_moves": "",
                "current_proposal": "",
                "verifier_feedback": "",
                "attempt_count": 0,
                "final_output": "",
                "status": "SEARCHING",
            }

            try:
                # Select per-task step limit (fallback to default)
                step_limit = TASK_STEP_LIMITS.get(task, DEFAULT_RECURSION_LIMIT)
                f.write(f"RecursionLimit: {step_limit}\n")
                final_state = None
                for step in app.stream(initial_state, config={"recursion_limit": step_limit}):
                    for node_name, node_state in step.items():
                        final_state = node_state

                stats[task]["total"] += 1

                if final_state and "final_output" in final_state:
                    prediction = final_state['final_output']
                    ground_truth = example.get('answer')

                    is_correct = verify_answer(prediction, ground_truth, task)

                    f.write(f"Prediction: {prediction}\n")
                    f.write(f"Verification: {'CORRECT' if is_correct else 'INCORRECT'}\n")

                    if is_correct:
                        stats[task]["correct"] += 1
                    else:
                        stats[task]["incorrect"] += 1
                else:
                    f.write("Prediction: No final output captured.\n")
                    f.write("Verification: ERROR\n")
                    stats[task]["errors"] += 1

            except Exception as e:
                f.write(f"Error running solver: {e}\n")
                stats[task]["total"] += 1
                stats[task]["errors"] += 1

            f.flush()
            processed += 1

        # After running examples, compute percentage-based summaries and graphs
        f.write(f"\n\n{'='*60}\n")
        f.write("FINAL SUMMARY STATISTICS (PERCENTAGES)\n")
        f.write(f"{'='*60}\n")

        # Prepare data for plotting
        tasks = sorted(stats.keys())
        correct_pct = []
        incorrect_pct = []
        task_sample_pct = []

        total_processed = sum(stats[t]['total'] for t in tasks)
        total_processed = total_processed if total_processed > 0 else processed

        for t in tasks:
            tdata = stats[t]
            tot = tdata['total']
            if tot == 0:
                correct = 0.0
                incorrect_merged = 0.0
            else:
                correct = 100.0 * tdata['correct'] / tot
                # Merge errors into incorrect for reporting and plotting
                incorrect_merged = 100.0 * (tdata['incorrect'] + tdata['errors']) / tot

            correct_pct.append(correct)
            incorrect_pct.append(incorrect_merged)

            # fraction of the sampled set that this task represents (percentage)
            task_sample_pct.append(100.0 * (tdata['total'] / total_processed) if total_processed > 0 else 0.0)

            f.write(f"{t}: Correct={correct:.1f}%, Incorrect={incorrect_merged:.1f}%, SamplePct={task_sample_pct[-1]:.1f}%\n")

        # Save a stacked bar chart with percentages per task (errors merged into incorrect)
        x = range(len(tasks))
        fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 0.6), 6))
        ax.bar(x, correct_pct, label='Correct', color='#4CAF50')
        # incorrect_pct already includes errors
        ax.bar(x, incorrect_pct, bottom=correct_pct, label='Incorrect (incl. errors)', color='#FF9800')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Per-Task Result Percentages (Correct / Incorrect — errors merged)')
        ax.legend()
        plt.tight_layout()
        fig_path1 = 'results_task_percentages.png'
        fig.savefig(fig_path1)
        f.write(f"Saved per-task percentages plot to {fig_path1}\n")

        # Save a bar chart showing percentage of sampled examples per task
        fig2, ax2 = plt.subplots(figsize=(max(8, len(tasks) * 0.6), 5))
        ax2.bar(x, task_sample_pct, color='#2196F3')
        ax2.set_xticks(x)
        ax2.set_xticklabels(tasks, rotation=45, ha='right')
        ax2.set_ylabel('Percentage of Sample (%)')
        ax2.set_title('Distribution of Sampled Test Examples by Task (Percentage)')
        plt.tight_layout()
        fig_path2 = 'results_task_distribution.png'
        fig2.savefig(fig_path2)
        f.write(f"Saved task distribution plot to {fig_path2}\n")

        # Report processed fraction as percentage of requested sample (results are percentages only)
        pct_processed = 100.0 * processed / max_examples if max_examples > 0 else 0.0
        f.write(f"Processed sample: {pct_processed:.1f}% of requested\n")

        # Write final summary (percentages only, errors merged into incorrect)
        f.write(f"\n\n{'='*60}\n")
        f.write("FINAL SUMMARY STATISTICS (PERCENTAGES)\n")
        f.write(f"{'='*60}\n")
        for t in tasks:
            tdata = stats[t]
            tot = tdata['total']
            if tot == 0:
                correct = 0.0
                incorrect_merged = 0.0
            else:
                correct = 100.0 * tdata['correct'] / tot
                incorrect_merged = 100.0 * (tdata['incorrect'] + tdata['errors']) / tot
            f.write(f"{t}: Correct={correct:.1f}%, Incorrect={incorrect_merged:.1f}%\n")

if __name__ == "__main__":
    main()
