"""Precedent retriever using SentenceTransformers + FAISS (CPU only).

Dependency install command:
	pip install sentence-transformers faiss-cpu numpy pickle
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILENAME = "cases.index"
METADATA_FILENAME = "cases.pkl"
EMBEDDING_BATCH_SIZE = 128


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = PROJECT_ROOT / "data" / "cases"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
INDEX_PATH = VECTOR_DB_DIR / INDEX_FILENAME
METADATA_PATH = VECTOR_DB_DIR / METADATA_FILENAME


_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_cases: Optional[List[Dict[str, str]]] = None


def _get_model() -> SentenceTransformer:
	global _model
	if _model is None:
		print(f"[precedent_retriever] Loading embedding model: {MODEL_NAME} (CPU only)...")
		_model = SentenceTransformer(MODEL_NAME, device="cpu")
		print("[precedent_retriever] Model loaded.")
	return _model


def _load_case_documents(cases_dir: Path = CASES_DIR) -> List[Dict[str, str]]:
	if not cases_dir.exists():
		raise FileNotFoundError(f"Cases directory not found: {cases_dir}")

	files = sorted(cases_dir.glob("*.txt"))
	if not files:
		raise FileNotFoundError(f"No .txt case files found in: {cases_dir}")

	print(f"[precedent_retriever] Found {len(files)} case files. Loading text...")
	records: List[Dict[str, str]] = []
	for idx, file_path in enumerate(files, start=1):
		case_text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
		records.append(
			{
				"case_id": file_path.name,
				"text": case_text,
			}
		)
		if idx % 500 == 0 or idx == len(files):
			print(f"[precedent_retriever] Processed {idx}/{len(files)} case files...")

	print("[precedent_retriever] Completed loading case documents.")
	return records


def _embed_texts(texts: Sequence[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
	model = _get_model()
	print(f"[precedent_retriever] Generating embeddings for {len(texts)} cases...")
	embeddings = model.encode(
		list(texts),
		batch_size=batch_size,
		show_progress_bar=True,
		convert_to_numpy=True,
		normalize_embeddings=True,
	)
	embeddings = np.asarray(embeddings, dtype=np.float32)
	print(f"[precedent_retriever] Embeddings generated with shape: {embeddings.shape}")
	return embeddings


def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
	if embeddings.ndim != 2:
		raise ValueError("Embeddings must be a 2D array of shape (n_samples, embedding_dim)")

	embedding_dim = embeddings.shape[1]
	print(f"[precedent_retriever] Building FAISS index (dim={embedding_dim})...")
	index = faiss.IndexFlatIP(embedding_dim)
	index.add(embeddings)
	print(f"[precedent_retriever] FAISS index built with {index.ntotal} vectors.")
	return index


def build_and_save_precedent_index(
	cases_dir: Path = CASES_DIR,
	index_path: Path = INDEX_PATH,
	metadata_path: Path = METADATA_PATH,
) -> None:
	print("[precedent_retriever] Starting precedent indexing pipeline...")
	records = _load_case_documents(cases_dir)
	texts = [record["text"] for record in records]
	embeddings = _embed_texts(texts)
	index = _build_faiss_index(embeddings)

	index_path.parent.mkdir(parents=True, exist_ok=True)
	print(f"[precedent_retriever] Saving FAISS index to: {index_path}")
	faiss.write_index(index, str(index_path))

	print(f"[precedent_retriever] Saving case metadata to: {metadata_path}")
	with metadata_path.open("wb") as fp:
		pickle.dump(records, fp)

	print("[precedent_retriever] Precedent indexing pipeline completed successfully.")


def _load_saved_index_and_metadata(
	index_path: Path = INDEX_PATH,
	metadata_path: Path = METADATA_PATH,
) -> tuple[faiss.Index, List[Dict[str, str]]]:
	if not index_path.exists() or not metadata_path.exists():
		raise FileNotFoundError(
			"Saved precedent index artifacts not found. "
			"Run build_and_save_precedent_index() first."
		)

	print(f"[precedent_retriever] Loading FAISS index from: {index_path}")
	index = faiss.read_index(str(index_path))

	print(f"[precedent_retriever] Loading case metadata from: {metadata_path}")
	with metadata_path.open("rb") as fp:
		records = pickle.load(fp)

	if len(records) != index.ntotal:
		raise ValueError(f"Index/case count mismatch: index={index.ntotal}, cases={len(records)}")

	return index, records


def initialize_precedent_retriever(
	cases_dir: Path = CASES_DIR,
	index_path: Path = INDEX_PATH,
	metadata_path: Path = METADATA_PATH,
) -> Dict[str, object]:
	global _index, _cases

	created = False
	if not index_path.exists() or not metadata_path.exists():
		print("[precedent_retriever] Index artifacts missing. Rebuilding case embeddings...")
		build_and_save_precedent_index(
			cases_dir=cases_dir,
			index_path=index_path,
			metadata_path=metadata_path,
		)
		created = True

	_index, _cases = _load_saved_index_and_metadata(index_path=index_path, metadata_path=metadata_path)

	return {
		"created": created,
		"case_count": len(_cases),
		"index_path": str(index_path),
		"metadata_path": str(metadata_path),
	}


def _ensure_loaded() -> tuple[faiss.Index, List[Dict[str, str]]]:
	global _index, _cases
	if _index is None or _cases is None:
		_index, _cases = _load_saved_index_and_metadata()
	return _index, _cases


def retrieve_precedents(query: str, top_k: int = 3) -> List[Dict[str, object]]:
	if not query or not query.strip():
		raise ValueError("Query must be a non-empty string.")
	if top_k <= 0:
		raise ValueError("top_k must be greater than 0.")

	index, records = _ensure_loaded()
	model = _get_model()

	print(f"[precedent_retriever] Retrieving top {top_k} precedents for query...")
	query_embedding = model.encode(
		[query.strip()],
		convert_to_numpy=True,
		normalize_embeddings=True,
	)
	query_embedding = np.asarray(query_embedding, dtype=np.float32)

	search_k = min(top_k, index.ntotal)
	scores, indices = index.search(query_embedding, search_k)

	results: List[Dict[str, object]] = []
	for score, idx in zip(scores[0], indices[0]):
		if idx < 0:
			continue

		record = records[idx]
		full_text = (record.get("text", "") or "").strip()
		results.append(
			{
				"case_id": record.get("case_id", ""),
				"score": float(score),
				"text_preview": full_text[:300],
				"text": full_text,
			}
		)

	print(f"[precedent_retriever] Retrieved {len(results)} precedents.")
	return results


if __name__ == "__main__":
	build_and_save_precedent_index()
