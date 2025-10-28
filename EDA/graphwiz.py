import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
import os
import re # Import regex for content analysis
import numpy as np # Import numpy for np.select

def run_eda():
    """
    Performs Exploratory Data Analysis (EDA) on the GraphWiz/GraphInstruct dataset.
    """

    # --- 1. Load Dataset ---
    print("Loading dataset 'GraphWiz/GraphInstruct'...")
    try:
        dataset = load_dataset("GraphWiz/GraphInstruct")
    except Exception as e:
        print(f"Error loading dataset. Do you have an internet connection? \n{e}")
        return

    print("\n--- Dataset Structure ---")
    print(dataset)

    # --- 2. Initial Inspection ---
    # Check the features of the 'train' split
    if 'train' not in dataset:
        print("Error: 'train' split not found in dataset.")
        return

    print("\n--- 'train' Split Features ---")
    print(dataset['train'].features)

    # Print the first example
    print("\n--- First Example Record (train[0]) ---")
    example = dataset['train'][0]
    for key, value in example.items():
        # Truncate long values for readability
        value_str = str(value)
        if len(value_str) > 200:
            value_str = value_str[:200] + "..."
        print(f"  {key}: {value_str}")

    # --- 3. Convert to Pandas for Analysis ---
    print("\nConverting 'train' split to pandas DataFrame...")
    df = dataset['train'].to_pandas()

    print("\n--- DataFrame Head ---")
    print(df.head())

    print("\n--- DataFrame Info ---")
    df.info()

    # --- 4. Feature Engineering: Text Lengths ---
    print("\nCalculating text lengths for query and answer...")
    df['query_length'] = df['query'].str.len()
    df['answer_length'] = df['answer'].str.len()

    print("Performing content-specific analysis...")
    # Specific Analysis 1: Categorize answer types (Binary, Numeric, Other.)
    # We use regex to find answers that end with specific patterns

    # Define conditions for each answer type
    binary_cond = df['answer'].str.contains(r'###\s*(Yes|No)\.?$', regex=True, na=False)
    # This pattern looks for integers (e.g., ### 10) or floats (e.g., ### 10.5)
    numeric_cond = df['answer'].str.contains(r'###\s*(\d+(\.\d+)?)\.?$', regex=True, na=False)

    # Create the conditions and choices lists for np.select
    conditions = [
        binary_cond,
        numeric_cond
    ]
    choices = ['Binary', 'Numeric']

    # Use np.select to create the new column
    # default='Other' will be assigned to all rows that don't meet other conditions
    df['answer_type'] = np.select(conditions, choices, default='Other')

    # --- 5. Summary Statistics ---
    print("\n--- Summary Statistics (Text Lengths) ---")
    print(df[['query_length', 'answer_length']].describe())

    # --- 6. Categorical Analysis ---
    if 'task' in df.columns:
        print("\n--- 'task' Column Distribution ---")
        print(df['task'].value_counts(normalize=True))

    print("\n--- 'Answer Type' Distribution ---")
    print(df['answer_type'].value_counts(normalize=True))

    # --- 7. Visualization ---
    print("\nGenerating and saving visualizations...")

    # Create an 'eda_plots' directory if it doesn't exist
    output_dir = "eda_plots"
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Plot 1: Distribution of Query Length
    plt.figure(figsize=(12, 6))
    sns.histplot(df['query_length'], bins=50, kde=True)
    plt.title('Distribution of Query Lengths')
    plt.xlabel('Length (characters)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plot_path1 = os.path.join(output_dir, 'query_length_dist.png')
    plt.savefig(plot_path1)
    print(f"Saved: {plot_path1}")

    # Plot 2: Distribution of Answer Length (clipped for readability)
    # Clipping at the 99th percentile to handle extreme outliers
    answer_clip_val = df['answer_length'].quantile(0.99)
    df['answer_length_clipped'] = df['answer_length'].clip(upper=answer_clip_val)

    plt.figure(figsize=(12, 6))
    sns.histplot(df['answer_length_clipped'], bins=50, kde=True)
    plt.title(f'Distribution of Answer Lengths (Clipped at 99th percentile: {answer_clip_val:.0f})')
    plt.xlabel('Length (characters)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, 'answer_length_dist_clipped.png')
    plt.savefig(plot_path2)
    print(f"Saved: {plot_path2}")

    # Plot 3: Task distribution
    if 'task' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, y='task', order=df['task'].value_counts().index)
        plt.title('Count of Records by Task')
        plt.xlabel('Count')
        plt.ylabel('Task')
        plt.tight_layout()
        plot_path3 = os.path.join(output_dir, 'task_distribution.png')
    plt.savefig(plot_path3)
    print(f"Saved: {plot_path3}")

    # === Specific EDA Plots Start ===

    # Plot 4: Query Length vs. Task (Boxplot)
    if 'task' in df.columns:
        plt.figure(figsize=(12, 8))
        # Using query_length (unclipped)
        sns.boxplot(data=df, y='task', x='query_length')
        plt.title('Query Length Distribution by Task')
        plt.xlabel('Query Length (characters)')
        plt.ylabel('Task')
        # Use log scale for x-axis if distribution is heavily skewed
        plt.xscale('log')
        plt.tight_layout()
        plot_path4 = os.path.join(output_dir, 'query_length_by_task.png')
        plt.savefig(plot_path4)
        print(f"Saved: {plot_path4}")

    # Plot 5: Answer Length vs. Task (Boxplot)
    if 'task' in df.columns:
        plt.figure(figsize=(12, 8))
        # Using answer_length (unclipped)
        sns.boxplot(data=df, y='task', x='answer_length')
        plt.title('Answer Length Distribution by Task')
        plt.xlabel('Answer Length (characters)')
        plt.ylabel('Task')
        # Use log scale for x-axis if distribution is heavily skewed
        plt.xscale('log')
        plt.tight_layout()
        plot_path5 = os.path.join(output_dir, 'answer_length_by_task.png')
        plt.savefig(plot_path5)
        print(f"Saved: {plot_path5}")

    # Plot 6: Query Length vs. Answer Length (Scatterplot)
    plt.figure(figsize=(10, 8))
    # Using alpha=0.1 to handle overplotting
    sns.scatterplot(data=df, x='query_length', y='answer_length', alpha=0.1, s=10)
    plt.title('Query Length vs. Answer Length')
    plt.xlabel('Query Length (characters)')
    plt.ylabel('Answer Length (characters)')
    # Using log scale can help visualize dense clusters
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plot_path6 = os.path.join(output_dir, 'query_vs_answer_length_scatter.png')
    plt.savefig(plot_path6)
    print(f"Saved: {plot_path6}")

    # Plot 7: Answer Type Distribution by Task
    if 'task' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.countplot(data=df, y='task', hue='answer_type', order=df['task'].value_counts().index)
        plt.title('Answer Type Distribution by Task')
        plt.xlabel('Count')
        plt.ylabel('Task')
        plt.legend(title='Answer Type')
        plt.tight_layout()
        plot_path7 = os.path.join(output_dir, 'answer_type_by_task.png')
        plt.savefig(plot_path7)
        print(f"Saved: {plot_path7}")

    # === Specific EDA Plots End ===

    print("\nEDA complete. Check the 'eda_plots' directory for visualizations.")

if __name__ == "__main__":
    # Install necessary packages if you don't have them
    # pip install datasets pandas seaborn matplotlib
    run_eda()
