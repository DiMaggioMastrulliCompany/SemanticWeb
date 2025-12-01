import argparse
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt


def parse_results(log_path: Path):
    """Parse a test log and return counts per task."""
    stats = defaultdict(lambda: {"correct": 0, "incorrect": 0, "errors": 0, "total": 0})
    current_task = None

    with log_path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            if line.startswith("Task:"):
                current_task = line.split("Task:", 1)[1].strip()

            elif line.startswith("Verification:"):
                if not current_task:
                    continue
                status = line.split("Verification:", 1)[1].strip().upper()
                task_stats = stats[current_task]
                task_stats["total"] += 1

                if status.startswith("CORRECT"):
                    task_stats["correct"] += 1
                elif status.startswith("INCORRECT"):
                    task_stats["incorrect"] += 1
                else:
                    task_stats["errors"] += 1
                current_task = None

            elif line.startswith("Error running solver"):
                if not current_task:
                    continue
                task_stats = stats[current_task]
                task_stats["total"] += 1
                task_stats["errors"] += 1
                current_task = None

    return stats


def compute_percentages(stats):
    tasks = sorted(stats.keys())
    total_processed = sum(stats[t]["total"] for t in tasks)

    data = {
        "tasks": tasks,
        "correct_pct": [],
        "incorrect_pct": [],
        "errors_pct": [],
        "sample_pct": [],
    }

    for t in tasks:
        totals = stats[t]["total"]
        if totals > 0:
            data["correct_pct"].append(100.0 * stats[t]["correct"] / totals)
            data["incorrect_pct"].append(100.0 * stats[t]["incorrect"] / totals)
            data["errors_pct"].append(100.0 * stats[t]["errors"] / totals)
        else:
            data["correct_pct"].append(0.0)
            data["incorrect_pct"].append(0.0)
            data["errors_pct"].append(0.0)

        data["sample_pct"].append(100.0 * totals / total_processed if total_processed else 0.0)

    return data


def plot_percentages(data, output_path: Path):
    tasks = data["tasks"]
    x = range(len(tasks))

    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 0.6), 6))
    ax.bar(x, data["correct_pct"], label="Correct", color="#4CAF50")
    ax.bar(x, data["incorrect_pct"], bottom=data["correct_pct"], label="Incorrect", color="#FF9800")
    bottom_errors = [c + i for c, i in zip(data["correct_pct"], data["incorrect_pct"])]
    ax.bar(x, data["errors_pct"], bottom=bottom_errors, label="Errors", color="#F44336")

    ax.set_xticks(list(x))
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Per-Task Result Percentages (Correct / Incorrect / Errors)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path)


def plot_distribution(data, output_path: Path):
    tasks = data["tasks"]
    x = range(len(tasks))

    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 0.6), 5))
    ax.bar(x, data["sample_pct"], color="#2196F3")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_ylabel("Percentage of Sample (%)")
    ax.set_title("Distribution of Sampled Test Examples by Task (Percentage)")
    plt.tight_layout()
    fig.savefig(output_path)


def main():
    parser = argparse.ArgumentParser(description="Plot per-task results with errors separated.")
    parser.add_argument(
        "--results-file",
        default="test_results.txt",
        help="Path to the test results log produced by test_all_tasks.py",
    )
    parser.add_argument(
        "--percentages-output",
        default="results_task_percentages_with_errors.png",
        help="Output path for the stacked percentage bar chart",
    )
    parser.add_argument(
        "--distribution-output",
        default="results_task_distribution_with_errors.png",
        help="Output path for the task distribution bar chart",
    )
    args = parser.parse_args()

    log_path = Path(args.results_file)
    stats = parse_results(log_path)
    if not stats:
        raise SystemExit("No tasks found in the results file.")

    percentages = compute_percentages(stats)

    out_dir = log_path.parent
    pct_path = Path(args.percentages_output)
    if not pct_path.is_absolute():
        pct_path = out_dir / pct_path

    dist_path = Path(args.distribution_output)
    if not dist_path.is_absolute():
        dist_path = out_dir / dist_path

    plot_percentages(percentages, pct_path)
    plot_distribution(percentages, dist_path)

    print("Saved plots to:")
    print(f"- {pct_path}")
    print(f"- {dist_path}")


if __name__ == "__main__":
    main()
