from __future__ import annotations

import argparse
from pathlib import Path

from retrieval import precedent_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
DEBUG_INDEX_PATH = VECTOR_DB_DIR / "cases_processed_debug.index"
DEBUG_METADATA_PATH = VECTOR_DB_DIR / "cases_processed_debug.pkl"


def _print_results(title: str, results: list[dict]) -> None:
    print(f"\n========== {title} ==========")
    if not results:
        print("No results found.")
        return

    for rank, item in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Case ID: {item.get('case_id', '')}")
        print(f"Chunk ID: {item.get('chunk_id', '')}")
        print(f"Section: {item.get('section', '')}")
        print(f"Side Hint: {item.get('side_hint', '')}")
        print(f"Score: {item.get('score', 0.0):.4f}")
        preview = (item.get("text_preview", "") or "").replace("\n", " ").strip()
        print(f"Preview: {preview[:240]}")


def run_debug_retrieval(case_text: str, top_k: int = 5, rebuild: bool = False) -> None:
    if rebuild or (not DEBUG_INDEX_PATH.exists()) or (not DEBUG_METADATA_PATH.exists()):
        print("[debug] Building dedicated processed-cases index...")
        precedent_retriever.build_and_save_precedent_index(
            index_path=DEBUG_INDEX_PATH,
            metadata_path=DEBUG_METADATA_PATH,
        )
    else:
        print("[debug] Reusing existing processed-cases debug index (no rebuild).")

    print("[debug] Loading dedicated processed-cases index...")
    init_info = precedent_retriever.initialize_precedent_retriever(
        index_path=DEBUG_INDEX_PATH,
        metadata_path=DEBUG_METADATA_PATH,
    )
    print(f"[debug] Records indexed: {init_info['case_count']}")
    print(f"[debug] Index path: {init_info['index_path']}")
    print(f"[debug] Metadata path: {init_info['metadata_path']}")

    prosecution_results = precedent_retriever.retrieve_precedents(
        query=case_text,
        top_k=top_k,
        side_hint="prosecution",
        allowed_sections=[
            "facts",
            "issues",
            "prosecution_arguments",
            "court_analysis",
            "decision",
            "other",
        ],
    )

    defense_results = precedent_retriever.retrieve_precedents(
        query=case_text,
        top_k=top_k,
        side_hint="defense",
        allowed_sections=[
            "facts",
            "issues",
            "defense_arguments",
            "court_analysis",
            "decision",
            "other",
        ],
    )

    judge_results = precedent_retriever.retrieve_precedents(
        query=case_text,
        top_k=top_k,
        side_hint="judge",
    )

    _print_results("PROSECUTION RETRIEVAL (PROCESSED CHUNKS)", prosecution_results)
    _print_results("DEFENSE RETRIEVAL (PROCESSED CHUNKS)", defense_results)
    _print_results("JUDGE RETRIEVAL (PROCESSED CHUNKS)", judge_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug retrieval over processed chunk index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-embedding and rebuild of the processed debug index.",
    )
    args = parser.parse_args()

    print("Enter case facts/query text (single line):")
    user_input = input().strip()
    if not user_input:
        raise ValueError("Input cannot be empty.")
    run_debug_retrieval(user_input, rebuild=args.rebuild)
