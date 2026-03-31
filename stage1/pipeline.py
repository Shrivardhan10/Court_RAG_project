"""
pipeline.py - Stage 1: Hybrid GPU pipeline. Always uses rules + ML model.
Model loads once at import. Batch inference only. No rules-only mode.
"""

import os
import json
from typing import Dict, List

from stage1.classifier import classify_batch
from stage1.model import load_model, is_model_available
from stage0.pipeline import process_text

_MODEL_BOOTSTRAPPED = False


def _ensure_model_loaded() -> None:
    global _MODEL_BOOTSTRAPPED
    if not _MODEL_BOOTSTRAPPED:
        load_model()
        _MODEL_BOOTSTRAPPED = True


def process_case_stage1(file_path: str) -> List[Dict[str, str]]:
    """
    Load Stage 0 JSON and classify all sentences using hybrid pipeline.

    Args:
        file_path: Path to Stage 0 JSON (e.g. data/stage0/C1.json).

    Returns:
        List of {"text": ..., "label": ...} dicts.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Stage 0 file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    sentences = [s for s in data.get("sentences", []) if s and s.strip()]
    if not sentences:
        return []

    _ensure_model_loaded()
    labels = classify_batch(sentences)
    return [{"text": s, "label": l} for s, l in zip(sentences, labels)]


def process_sentences_stage1(sentences: List[str]) -> List[Dict[str, str]]:
    """
    Classify an in-memory list of sentences with Stage 1.

    Args:
        sentences: List of cleaned sentence strings.

    Returns:
        List of {"text": ..., "label": ...} dicts.
    """
    valid_sentences = [s for s in (sentences or []) if s and s.strip()]
    if not valid_sentences:
        return []

    _ensure_model_loaded()
    labels = classify_batch(valid_sentences)
    return [{"text": s, "label": l} for s, l in zip(valid_sentences, labels)]


def process_text_stage1(text: str) -> List[Dict[str, str]]:
    """
    Run Stage 0 + Stage 1 directly on raw text.

    Args:
        text: Raw case narrative.

    Returns:
        List of {"text": ..., "label": ...} dicts.
    """
    sentences = process_text(text)
    return process_sentences_stage1(sentences)


def save_stage1_output(
    case_id: str,
    labeled_sentences: List[Dict[str, str]],
    output_dir: str = "data/stage1",
) -> None:
    """Save labeled sentences as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{case_id}.json")
    payload  = {"case_id": case_id, "labeled_sentences": labeled_sentences}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def process_all_cases_stage1(
    folder_path: str = "data/stage0",
    output_dir:  str = "data/stage1",
) -> Dict[str, List[Dict[str, str]]]:
    """
    Process all Stage 0 JSON files with hybrid GPU pipeline.

    Args:
        folder_path: Directory with Stage 0 JSON files.
        output_dir:  Directory to write Stage 1 JSON files.

    Returns:
        Dict mapping case_id → labeled_sentences.
    """
    json_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(".json")
    )
    total   = len(json_files)
    results: Dict[str, List[Dict[str, str]]] = {}

    print(f"[Stage 1] Processing {total} files | model: {is_model_available()}")

    for idx, filename in enumerate(json_files, start=1):
        case_id   = os.path.splitext(filename)[0]
        file_path = os.path.join(folder_path, filename)

        try:
            labeled = process_case_stage1(file_path)
            save_stage1_output(case_id, labeled, output_dir=output_dir)
            results[case_id] = labeled
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            continue

        if idx % 50 == 0 or idx == total:
            print(f"[Stage 1] {idx}/{total} completed")

    print(f"[Stage 1] Done. {len(results)}/{total} saved to '{output_dir}'")
    return results


if __name__ == "__main__":
    process_all_cases_stage1()
