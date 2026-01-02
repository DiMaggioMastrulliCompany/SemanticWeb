import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nlgraph_loader import load_nlgraph


BINARY_TOKEN_RE = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)
NUMERIC_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")

def extract_binary_label(text: str) -> str | None:
    """
    Return the last binary token found in the text as a normalized label:
    'yes', 'no', 'true', or 'false'. If none found, return None.
    """
    if not isinstance(text, str) or not text:
        return None
    matches = list(BINARY_TOKEN_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).lower()


def extract_numeric_value(text: str) -> float | None:
    """
    Return the last numeric value found in the text as float, else None.
    Handles integers and floats with optional sign.
    """
    if not isinstance(text, str) or not text:
        return None
    matches = list(NUMERIC_RE.finditer(text))
    if not matches:
        return None
    try:
        return float(matches[-1].group(0))
    except Exception:
        return None


def classify_answer_type(df: pd.DataFrame) -> pd.Series:
    """
    Classify answers into Binary / Numeric / Other for NLGraph.

    - Binary if any standalone yes/no/true/false token appears (case-insensitive),
      even if wrapped in extra text (e.g., 'The answer is yes.').
    - Numeric if a number appears (anywhere).
    - If both appear, prefer Binary (useful for tasks like connectivity).
    - Otherwise Other.
    """
    ans = df["answer"].fillna("").astype(str)

    binary_label = ans.apply(extract_binary_label)
    has_binary = binary_label.notna()

    numeric_value = ans.apply(extract_numeric_value)
    has_numeric = numeric_value.notna()

    # Prefer Binary when both present
    return np.where(
        has_binary, "Binary",
        np.where(has_numeric, "Numeric", "Other")
    )


def safe_len_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    return s.str.len()


def run_eda():
    """
    Performs Exploratory Data Analysis (EDA) on the NLGraph dataset.
    Expects columns: query, answer, task (as produced by load_nlgraph()).
    """
    # --- 1. Load Dataset ---
    dataset = load_nlgraph()

    print("\n--- Dataset Structure ---")
    print(dataset)

    # --- 2. Initial Inspection ---
    if "train" not in dataset:
        print("Error: 'train' split not found in dataset.")
        return

    print("\n--- 'train' Split Features ---")
    print(dataset["train"].features)

    print("\n--- First Example Record (train[0]) ---")
    example = dataset["train"][0]
    for key, value in example.items():
        value_str = str(value)
        if len(value_str) > 200:
            value_str = value_str[:200] + "..."
        print(f"  {key}: {value_str}")

    # --- 3. Convert to Pandas for Analysis ---
    print("\nConverting 'train' split to pandas DataFrame...")
    df = dataset["train"].to_pandas()
    # Ensure expected columns exist
    expected_cols = {"query", "answer", "task"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in dataset: {missing}")

    print("\n--- DataFrame Head ---")
    print(df.head())

    print("\n--- DataFrame Info ---")
    df.info()

    # --- 4. Feature Engineering: Text Lengths & Answer Types ---
    print("\nCalculating text lengths for query and answer...")
    df["query"] = df["query"].fillna("").astype(str)
    df["answer"] = df["answer"].fillna("").astype(str)

    df["query_length"] = safe_len_series(df["query"])
    df["answer_length"] = safe_len_series(df["answer"])

    print("Classifying answer types (Binary / Numeric / Other)...")
    df["answer_type"] = classify_answer_type(df)

    # Add helpful derived columns for deeper checks
    print("Extracting normalized binary labels and numeric values...")
    df["answer_binary_norm"] = df["answer"].apply(extract_binary_label)  # 'yes','no','true','false' or None
    df["numeric_value"] = df["answer"].apply(extract_numeric_value)      # float or None

    # --- 5. Summary Statistics ---
    print("\n--- Summary Statistics (Text Lengths) ---")
    print(df[["query_length", "answer_length"]].describe())

    # --- 6. Categorical Analysis ---
    print("\n--- 'task' Column Distribution ---")
    print(df["task"].value_counts(normalize=True))

    print("\n--- 'Answer Type' Distribution ---")
    print(df["answer_type"].value_counts(normalize=True))

    # Optional: show per-task binary share to verify connectivity looks binary
    print("\n--- Binary Share by Task (sanity check) ---")
    print(
        df.assign(is_binary=(df["answer_type"] == "Binary"))
          .groupby("task")["is_binary"].mean()
          .sort_values(ascending=False)
    )

    # --- 7. Visualization ---
    print("\nGenerating and saving visualizations...")

    output_dir = "EDA/eda_plots_nlgraph"
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Plot 1: Distribution of Query Length
    plt.figure(figsize=(12, 6))
    sns.histplot(df["query_length"], bins=50, kde=True)
    plt.title("Distribution of Query Lengths")
    plt.xlabel("Length (characters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plot_path1 = os.path.join(output_dir, "query_length_dist.png")
    plt.savefig(plot_path1)
    print(f"Saved: {plot_path1}")

    # Plot 2: Distribution of Answer Length (clipped for readability)
    answer_clip_val = df["answer_length"].quantile(0.99)
    df["answer_length_clipped"] = df["answer_length"].clip(upper=answer_clip_val)

    plt.figure(figsize=(12, 6))
    sns.histplot(df["answer_length_clipped"], bins=50, kde=True)
    plt.title(
        f"Distribution of Answer Lengths (Clipped at 99th percentile: {answer_clip_val:.0f})"
    )
    plt.xlabel("Length (characters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, "answer_length_dist_clipped.png")
    plt.savefig(plot_path2)
    print(f"Saved: {plot_path2}")

    # Plot 3: Task distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, y="task", order=df["task"].value_counts().index)
    plt.title("Count of Records by Task")
    plt.xlabel("Count")
    plt.ylabel("Task")
    plt.tight_layout()
    plot_path3 = os.path.join(output_dir, "task_distribution.png")
    plt.savefig(plot_path3)
    print(f"Saved: {plot_path3}")

    # Plot 4: Query Length vs. Task (Boxplot)
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, y="task", x="query_length")
    plt.title("Query Length Distribution by Task")
    plt.xlabel("Query Length (characters)")
    plt.ylabel("Task")
    # Use log scale for x-axis if distribution is heavily skewed
    plt.xscale('log')
    from matplotlib.ticker import LogLocator, NullFormatter
    ax = plt.gca()
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plot_path4 = os.path.join(output_dir, "query_length_by_task.png")
    plt.savefig(plot_path4)
    print(f"Saved: {plot_path4}")

    # Plot 5: Answer Length vs. Task (Boxplot)
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, y="task", x="answer_length")
    plt.title("Answer Length Distribution by Task")
    plt.xlabel("Answer Length (characters)")
    plt.ylabel("Task")
    plt.xscale('log')
    ax = plt.gca()
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plot_path5 = os.path.join(output_dir, "answer_length_by_task.png")
    plt.savefig(plot_path5)
    print(f"Saved: {plot_path5}")

    # Plot 6: Query Length vs. Answer Length (Scatterplot)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="query_length",
        y="answer_length",
        alpha=0.1,
        s=10,
    )
    plt.title("Query Length vs. Answer Length")
    plt.xlabel("Query Length (characters)")
    plt.ylabel("Answer Length (characters)")
    # Using log scale can help visualize dense clusters
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plot_path6 = os.path.join(output_dir, "query_vs_answer_length_scatter.png")
    plt.savefig(plot_path6)
    print(f"Saved: {plot_path6}")

    # Plot 7: Answer Type Distribution by Task
    plt.figure(figsize=(12, 8))
    sns.countplot(
        data=df, y="task", hue="answer_type", order=df["task"].value_counts().index
    )
    plt.title("Answer Type Distribution by Task")
    plt.xlabel("Count")
    plt.ylabel("Task")
    plt.legend(title="Answer Type")
    plt.tight_layout()
    plot_path7 = os.path.join(output_dir, "answer_type_by_task.png")
    plt.savefig(plot_path7)
    print(f"Saved: {plot_path7}")

    print("\nEDA complete. Check the 'EDA/eda_plots_nlgraph' directory for visualizations.")


if __name__ == "__main__":
    run_eda()
