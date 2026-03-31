"""IPC statute retriever with FAISS indexing from ipc.json."""

from __future__ import annotations

import json
import pickle
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .bm25_index import BM25Index, tokenize
from .hybrid_retriever import rerank_statute_candidates


MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILENAME = "ipc.index"
METADATA_FILENAME = "ipc.pkl"
BM25_FILENAME = "ipc_bm25.pkl"
EMBEDDING_BATCH_SIZE = 128


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IPC_JSON_PATH = PROJECT_ROOT / "ipc.json"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
INDEX_PATH = VECTOR_DB_DIR / INDEX_FILENAME
METADATA_PATH = VECTOR_DB_DIR / METADATA_FILENAME
BM25_PATH = VECTOR_DB_DIR / BM25_FILENAME


_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_statutes: Optional[List[Dict[str, object]]] = None
_bm25: Optional[BM25Index] = None
_model_load_attempted: bool = False

ALLOW_REMOTE_MODEL_DOWNLOAD = os.getenv("ALLOW_REMOTE_MODEL_DOWNLOAD", "0").strip() == "1"

_STOPWORDS = {
	"the", "and", "or", "of", "to", "for", "a", "an", "is", "are", "in", "on", "with", "by", "be", "as", "that", "this", "it", "which", "who", "shall", "any", "under", "when", "if", "not",
}

def _get_model() -> Optional[SentenceTransformer]:
	global _model, _model_load_attempted
	if _model is None and not _model_load_attempted:
		_model_load_attempted = True
		print(f"[statute_retriever] Loading embedding model: {MODEL_NAME} (CPU only)...")
		try:
			_model = SentenceTransformer(
				MODEL_NAME,
				device="cpu",
				local_files_only=not ALLOW_REMOTE_MODEL_DOWNLOAD,
			)
			print("[statute_retriever] Model loaded.")
		except Exception as exc:
			print(f"[statute_retriever] Embedding model unavailable, using BM25-only fallback: {exc}")
			_model = None
	return _model


def _extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
	words = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
	freq: Dict[str, int] = {}
	for word in words:
		if word in _STOPWORDS:
			continue
		freq[word] = freq.get(word, 0) + 1

	ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
	return [word for word, _ in ordered[:max_keywords]]


def _build_embedding_text(section_number: str, title: str, description: str) -> str:
	keywords = _extract_keywords(description)
	keyword_text = " ".join(keywords)
	return f"IPC Section {section_number} {title}. Keywords: {keyword_text}".strip()


def _load_ipc_records(ipc_json_path: Path = IPC_JSON_PATH) -> List[Dict[str, object]]:
	if not ipc_json_path.exists():
		raise FileNotFoundError(f"IPC JSON not found: {ipc_json_path}")

	payload = json.loads(ipc_json_path.read_text(encoding="utf-8", errors="ignore"))
	if not isinstance(payload, list):
		raise ValueError("ipc.json must contain a list of section objects.")

	records: List[Dict[str, object]] = []
	for row in payload:
		if not isinstance(row, dict):
			continue

		section_number = str(row.get("Section", "") or row.get("section", "")).strip()
		if not section_number:
			continue

		title = str(row.get("section_title", "") or "").strip()
		description = str(row.get("section_desc", "") or "").strip()
		chapter = str(row.get("chapter", "") or "").strip()
		chapter_title = str(row.get("chapter_title", "") or "").strip()

		record = {
			"section": section_number,
			"title": title,
			"description": description,
			"chapter": chapter,
			"chapter_title": chapter_title,
			"embedding_text": _build_embedding_text(section_number, title, description),
		}
		records.append(record)

	if not records:
		raise ValueError("No valid IPC records parsed from ipc.json")

	return records


def _embed_texts(texts: Sequence[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
	model = _get_model()
	if model is None:
		raise RuntimeError("Embedding model unavailable for indexing.")
	embeddings = model.encode(
		list(texts),
		batch_size=batch_size,
		show_progress_bar=True,
		convert_to_numpy=True,
		normalize_embeddings=True,
	)
	return np.asarray(embeddings, dtype=np.float32)


def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
	if embeddings.ndim != 2:
		raise ValueError("Embeddings must be a 2D array.")
	index = faiss.IndexFlatIP(embeddings.shape[1])
	index.add(embeddings)
	return index


def build_and_save_statute_index(
	ipc_json_path: Path = IPC_JSON_PATH,
	index_path: Path = INDEX_PATH,
	metadata_path: Path = METADATA_PATH,
	bm25_path: Path = BM25_PATH,
) -> None:
	print("[statute_retriever] Starting IPC indexing pipeline...")
	records = _load_ipc_records(ipc_json_path)
	embeddings = _embed_texts([str(x.get("embedding_text", "")) for x in records])
	index = _build_faiss_index(embeddings)
	bm25_docs = [
		tokenize(
			" ".join(
				[
					str(row.get("embedding_text", "")),
					str(row.get("title", "")),
					str(row.get("description", "")),
					str(row.get("section", "")),
				]
			)
		)
		for row in records
	]
	bm25_index = BM25Index.from_documents(bm25_docs)

	index_path.parent.mkdir(parents=True, exist_ok=True)
	faiss.write_index(index, str(index_path))
	with metadata_path.open("wb") as fp:
		pickle.dump(records, fp)
	with bm25_path.open("wb") as fp:
		pickle.dump(bm25_index, fp)
	print("[statute_retriever] IPC index saved successfully.")


def _load_saved_index_and_texts(
	index_path: Path = INDEX_PATH,
	metadata_path: Path = METADATA_PATH,
	bm25_path: Path = BM25_PATH,
) -> tuple[faiss.Index, List[Dict[str, object]], BM25Index]:
	if not index_path.exists() or not metadata_path.exists() or not bm25_path.exists():
		raise FileNotFoundError("IPC index artifacts not found. Run build_and_save_statute_index() first.")

	index = faiss.read_index(str(index_path))
	with metadata_path.open("rb") as fp:
		records = pickle.load(fp)
	with bm25_path.open("rb") as fp:
		bm25_index = pickle.load(fp)

	if len(records) != index.ntotal:
		raise ValueError(f"Index/statute mismatch: index={index.ntotal}, statutes={len(records)}")
	if not isinstance(bm25_index, BM25Index):
		raise TypeError("Invalid BM25 artifact for IPC index.")
	return index, records, bm25_index


def _ensure_loaded() -> tuple[faiss.Index, List[Dict[str, object]], BM25Index]:
	global _index, _statutes, _bm25
	if _index is None or _statutes is None or _bm25 is None:
		_index, _statutes, _bm25 = _load_saved_index_and_texts()
	return _index, _statutes, _bm25


def initialize_statute_retriever(
	ipc_json_path: Path = IPC_JSON_PATH,
	index_path: Path = INDEX_PATH,
	metadata_path: Path = METADATA_PATH,
	bm25_path: Path = BM25_PATH,
) -> Dict[str, object]:
	global _index, _statutes, _bm25

	created = False
	if not index_path.exists() or not metadata_path.exists() or not bm25_path.exists():
		build_and_save_statute_index(
			ipc_json_path=ipc_json_path,
			index_path=index_path,
			metadata_path=metadata_path,
			bm25_path=bm25_path,
		)
		created = True

	_index, _statutes, _bm25 = _load_saved_index_and_texts(index_path=index_path, metadata_path=metadata_path, bm25_path=bm25_path)
	return {
		"created": created,
		"statute_count": len(_statutes),
		"index_path": str(index_path),
		"metadata_path": str(metadata_path),
		"bm25_path": str(bm25_path),
	}


def retrieve_statutes(
	query: str,
	top_k: int = 3,
	chapter: Optional[str] = None,
	section_filter: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
	if not query or not query.strip():
		raise ValueError("Query must be a non-empty string.")
	if top_k <= 0:
		raise ValueError("top_k must be > 0.")
	result_count = max(3, top_k)

	index, statutes, bm25_index = _ensure_loaded()
	model = _get_model()
	semantic_candidates: List[tuple[int, float]] = []
	if model is not None:
		query_embedding = model.encode([query.strip()], convert_to_numpy=True, normalize_embeddings=True)
		query_embedding = np.asarray(query_embedding, dtype=np.float32)

		search_k = min(max(result_count * 10, 20), index.ntotal)
		scores, indices = index.search(query_embedding, search_k)
		semantic_candidates = [(int(idx), float(score)) for score, idx in zip(scores[0], indices[0]) if idx >= 0][:20]
	bm25_candidates = bm25_index.top_n(tokenize(query.strip()), n=min(20, len(statutes)))

	reranked = rerank_statute_candidates(
		query=query.strip(),
		records=statutes,
		semantic_candidates=semantic_candidates,
		bm25_candidates=bm25_candidates,
		top_k=result_count,
	)

	allowed_sections = {str(x).strip() for x in (section_filter or []) if str(x).strip()}
	filtered = [
		row for row in reranked
		if (chapter is None or str(row.get("chapter", "")) == str(chapter))
		and (not allowed_sections or str(row.get("section", "")) in allowed_sections)
	]

	if filtered:
		return filtered[:result_count]

	if chapter is None and not allowed_sections:
		return reranked[:result_count]

	return []


if __name__ == "__main__":
	build_and_save_statute_index()
