import re
from datasets import load_dataset


def extract_answer_graphwiz(query: str):
    match = re.search(r"###\s*(.*)", query)
    if match:
        return match.group(1)
    return None


def main():
    ds = load_dataset("GraphWiz/GraphInstruct")
    train = ds["train"]

    # Filter examples that have an extractable answer (same logic as preprocess.py)
    examples = []
    for ex in train:
        ds_answer = ex.get("answer")
        if ds_answer and extract_answer_graphwiz(ds_answer):
            examples.append(ex)

    if not examples:
        print("No train examples with an extractable '###' answer found.")
        return

    # Try to detect a task-type key to group by
    possible_task_keys = [
        "task",
        "type",
        "task_type",
        "instruction",
        "instruction_type",
        "category",
        "prompt",
        "query",
    ]

    task_key = None
    sample = examples[0]
    for k in possible_task_keys:
        if k in sample:
            task_key = k
            break

    groups = {}
    if task_key:
        for ex in examples:
            key = ex.get(task_key, "UNKNOWN")
            groups.setdefault(key, []).append(ex)
    else:
        # Fallback: put everything under a single group
        groups["all"] = examples

    # Print up to 5 examples per group with a short, helpful summary
    for key, items in groups.items():
        print(f"\n=== Task type: {key} ({len(items)} examples) ===")
        for i, ex in enumerate(items[:5]):
            # Determine a prompt-like field to show
            prompt_field = None
            for pf in ["input", "prompt", "question", "instruction", "query", "text"]:
                if pf in ex:
                    prompt_field = pf
                    break

            ds_answer = ex.get("answer", "")
            extracted = extract_answer_graphwiz(ds_answer) if ds_answer else None

            print(f"\n-- Example {i+1} --")
            if prompt_field:
                print(f"Prompt ({prompt_field}): {ex[prompt_field]}")
            else:
                # Show a short preview of the example fields if no prompt-like field is present
                shown_keys = list(ex.keys())[:8]
                preview = {k: ex[k] for k in shown_keys}
                print("Preview keys:", ", ".join(shown_keys))
                print(preview)

            print(f"Extracted Answer: {extracted}")


if __name__ == "__main__":
    main()
