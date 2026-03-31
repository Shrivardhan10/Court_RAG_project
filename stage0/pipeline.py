"""
pipeline.py - Stage 0: Orchestrator for Legal Case Preprocessing
Reads raw .txt case files, cleans, splits, and saves structured JSON output.
"""

import os
import json
from typing import Dict, List

try:
    from stage0.cleaner import clean_text
    from stage0.splitter import split_sentences
except ModuleNotFoundError:
    from cleaner import clean_text  # type: ignore
    from splitter import split_sentences  # type: ignore


def process_case(file_path: str) -> List[str]:
    """
    Read a single case file, clean it, and split into sentences.

    Args:
        file_path: Path to the raw .txt case file.

    Returns:
        List of cleaned sentence strings.

    Raises:
        FileNotFoundError: If file_path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Case file not found: {file_path}")

    # UTF-8 with latin-1 fallback for legacy legal documents
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            raw = f.read()

    cleaned = clean_text(raw)
    sentences = split_sentences(cleaned)
    return sentences


def process_text(text: str) -> List[str]:
    """
    Clean and split raw user text in-memory.

    Args:
        text: Raw case narrative.

    Returns:
        List of cleaned sentence strings.
    """
    cleaned = clean_text(text or "")
    return split_sentences(cleaned)


def save_stage0_output(
    case_id: str,
    sentences: List[str],
    output_dir: str = "data/stage0",
) -> None:
    """
    Save processed sentences as a structured JSON file.

    Args:
        case_id:    Identifier for the case (e.g. "C14").
        sentences:  List of sentence strings to save.
        output_dir: Directory to write JSON files into (created if absent).
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{case_id}.json")
    payload = {"case_id": case_id, "sentences": sentences}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def process_all_cases(
    folder_path: str,
    output_dir: str = "data/stage0",
) -> Dict[str, List[str]]:
    """
    Process every .txt file in folder_path and save JSON outputs.

    Args:
        folder_path: Directory containing raw .txt case files.
        output_dir:  Directory to write JSON outputs (default: "data/stage0").

    Returns:
        Dict mapping case_id → list of sentences for every successfully
        processed file.
    """
    txt_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(".txt")
    )
    total = len(txt_files)
    results: Dict[str, List[str]] = {}

    print(f"[Stage 0] Starting processing of {total} files from '{folder_path}'")

    for idx, filename in enumerate(txt_files, start=1):
        case_id = os.path.splitext(filename)[0]
        file_path = os.path.join(folder_path, filename)

        try:
            sentences = process_case(file_path)
            save_stage0_output(case_id, sentences, output_dir=output_dir)
            results[case_id] = sentences
        except Exception as exc:
            print(f"[Stage 0] ERROR processing {filename}: {exc}")
            continue

        if idx % 100 == 0 or idx == total:
            print(f"[Stage 0] Progress: {idx}/{total} files processed")

    print(f"[Stage 0] Done. {len(results)}/{total} files saved to '{output_dir}'")
    return results


if __name__ == "__main__":
    process_all_cases("data/cases")
