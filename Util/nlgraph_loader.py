import os
import json
import glob
import zipfile
import tempfile
import shutil
from typing import Dict, Iterable, List, Tuple, Any, Optional
from pathlib import Path

import requests
from datasets import Dataset, DatasetDict, Features, Value

# Known task subdirectories under the NLGraph/ root
_TASK_DIRS = [
    "connectivity",
    "cycle",
    "flow",
    "hamilton",
    "matching",
    "shortest_path",
    "topology",
]

_SPLIT_ALIASES = {
    "train": ["train"],
    "validation": ["validation", "valid", "val", "dev"],
    "test": ["test", "eval"],
}

_FILE_EXTS = [".jsonl", ".json"]

_FEATURES = Features(
    {
        "query": Value("string"),
        "answer": Value("string"),
        "task": Value("string"),
    }
)


def load_nlgraph(
    data_dir: Optional[str] = None,
    repo_owner: str = "Arthur-Heng",
    repo_name: str = "NLGraph",
    branch: str = "main",
    subdir: str = "NLGraph",
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Load NLGraph as a unified Hugging Face DatasetDict with columns: query, answer, task.

    If data_dir is provided, it must point to the NLGraph subdirectory that contains
    the task folders. If not provided, this function auto-downloads the GitHub repo zip
    and extracts the given subdir.

    Returns:
        DatasetDict with keys among {"train", "validation", "test"} (or only "train"
        if no explicit splits found). Each row is one example.
    """
    root = None
    if data_dir:
        root = _validate_dir(data_dir)
    else:
        root = _download_and_extract_repo(repo_owner, repo_name, branch, subdir, cache_dir)

    # Collect files per split
    split_to_files: Dict[str, List[Tuple[str, str]]] = {}
    for split_key in ["train", "validation", "test"]:
        files = _collect_split_files(root, split_key)
        if files:
            split_to_files[split_key] = files

    # Fallback: no explicit splits found; read anything as 'train'
    if not split_to_files:
        fallback_files = _collect_any_files(root)
        if not fallback_files:
            raise FileNotFoundError(
                f"No JSON/JSONL files found under known task directories in: {root}"
            )
        split_to_files["train"] = fallback_files

    # Build DatasetDict
    result = {}
    for split_name, files in split_to_files.items():
        examples = []
        for file_path, task_type in files:
            for rec in _iter_records(file_path):
                norm = _normalize_to_schema(rec, task_type)
                if norm is not None:
                    examples.append(norm)
        result[split_name] = Dataset.from_list(examples, features=_FEATURES)

    return DatasetDict(result)


def _validate_dir(path: str) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"data_dir does not exist or is not a directory: {p}")
    return str(p)


def _download_and_extract_repo(
    repo_owner: str,
    repo_name: str,
    branch: str,
    subdir: str,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Download GitHub repository zip and return the path to the requested subdir.
    Uses a simple on-disk cache to avoid re-downloading.
    """
    url = f"https://github.com/{repo_owner}/{repo_name}/archive/refs/heads/{branch}.zip"
    cache_base = Path(cache_dir or Path.home() / ".cache" / "nlgraph_loader").expanduser()
    cache_base.mkdir(parents=True, exist_ok=True)

    zip_path = cache_base / f"{repo_owner}_{repo_name}_{branch}.zip"
    extract_root = cache_base / f"{repo_owner}_{repo_name}_{branch}"
    target_subdir = extract_root / subdir

    if target_subdir.is_dir():
        return str(target_subdir)

    # Download if needed
    if not zip_path.exists():
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    # Extract to a temp then move into cache
    tmp_dir = Path(tempfile.mkdtemp(prefix="nlgraph_extract_"))
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        # Resolve top-level directory inside the zip
        top_dirs = [p for p in tmp_dir.iterdir() if p.is_dir()]
        if len(top_dirs) == 1:
            extracted_top = top_dirs[0]
        else:
            fallback = tmp_dir / f"{repo_name}-{branch}"
            extracted_top = fallback if fallback.is_dir() else (top_dirs[0] if top_dirs else tmp_dir)

        # Move to extract_root (overwrite if exists)
        if extract_root.exists():
            shutil.rmtree(extract_root)
        shutil.move(str(extracted_top), str(extract_root))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Find the subdir
    candidate = extract_root / subdir
    if candidate.is_dir():
        return str(candidate)

    # Last resort: search for any directory named subdir
    for p in extract_root.rglob("*"):
        if p.is_dir() and p.name == subdir:
            return str(p)

    raise FileNotFoundError(
        f"Could not locate subdir '{subdir}' within extracted repository at: {extract_root}"
    )


def _collect_split_files(root: str, split_key: str) -> List[Tuple[str, str]]:
    """Return list of (filepath, task_type) for a desired split_key across all tasks."""
    files: List[Tuple[str, str]] = []
    aliases = _SPLIT_ALIASES[split_key]
    for task in _TASK_DIRS:
        task_dir = os.path.join(root, task)
        if not os.path.isdir(task_dir):
            continue

        patterns = []
        for alias in aliases:
            for ext in _FILE_EXTS:
                patterns.append(os.path.join(task_dir, f"{alias}{ext}"))          # task/train.jsonl
                patterns.append(os.path.join(task_dir, alias, f"*{ext}"))         # task/train/*.jsonl

        matched = []
        for pat in patterns:
            matched.extend(glob.glob(pat))

        for path in sorted(set(matched)):
            if os.path.isfile(path):
                files.append((path, task))
    return files


def _collect_any_files(root: str) -> List[Tuple[str, str]]:
    """Fallback: collect all JSON/JSONL files under each known task directory."""
    files: List[Tuple[str, str]] = []
    for task in _TASK_DIRS:
        task_dir = os.path.join(root, task)
        if not os.path.isdir(task_dir):
            continue
        for ext in _FILE_EXTS:
            for path in glob.glob(os.path.join(task_dir, f"**/*{ext}"), recursive=True):
                if os.path.isfile(path):
                    files.append((path, task))
    return files


def _iter_records(file_path: str) -> Iterable[Dict[str, Any]]:
    """
    Iterate over records contained in a JSONL/JSON file.

    - JSONL: each line is one record.
    - JSON list: each element is one record.
    - JSON dict:
        - If most keys are numeric (e.g., "0","1",...), treat values as separate records.
        - Else, try to find a list under keys like data/examples/instances/records.
        - Else, yield the dict as a single record.
    """
    _, ext = os.path.splitext(file_path)
    if ext.lower() == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    if ext.lower() != ".json":
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for obj in data:
            yield obj
        return

    if isinstance(data, dict):
        # Case 1: numeric-keyed mapping of records
        keys = list(data.keys())
        numeric_keys = [k for k in keys if isinstance(k, int) or (isinstance(k, str) and k.isdigit())]
        if numeric_keys and len(numeric_keys) / max(1, len(keys)) >= 0.5:
            for k in sorted(numeric_keys, key=lambda x: int(x) if isinstance(x, str) else x):
                yield data[k]
            return

        # Case 2: dict with list under common container keys
        for container in ["data", "examples", "instances", "records"]:
            val = data.get(container)
            if isinstance(val, list):
                for obj in val:
                    yield obj
                return

        # Case 3: single record dict
        yield data
        return


def _to_text(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _normalize_to_schema(rec: Dict[str, Any], task_type: str) -> Optional[Dict[str, str]]:
    """
    Map a raw record to the target schema:
      - query (string)
      - answer (string)
      - task (string)

    If both query and answer are missing, return None to drop the example.
    """
    # Common field fallbacks observed in many NL-style datasets
    query = rec.get("question") or rec.get("query") or rec.get("prompt") or rec.get("instruction")
    answer = rec.get("answer") or rec.get("label") or rec.get("target") or rec.get("output")

    # Ensure strings
    query_text = _to_text(query)
    answer_text = _to_text(answer)

    if query_text is None or answer_text is None:
        return None

    return {
        "query": query_text,
        "answer": answer_text,
        "task": task_type,
    }


if __name__ == "__main__":
    dataset = load_nlgraph()
    print(dataset)
    print("\n--- First Example Record (train[0]) ---")
    example = dataset['train'][0]
    for key, value in example.items():
        # Truncate long values for readability
        value_str = str(value)
        if len(value_str) > 200:
            value_str = value_str[:200] + "..."
        print(f"  {key}: {value_str}")
