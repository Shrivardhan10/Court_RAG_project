"""Build FAISS index for structured precedent cases."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
	sys.path.insert(0, str(BACKEND_DIR))

from retrieval.precedent_retriever import build_and_save_precedent_index


if __name__ == "__main__":
	build_and_save_precedent_index()
