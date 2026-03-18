"""Statute retriever using SentenceTransformers + FAISS (CPU only).

Dependency install command:
	pip install sentence-transformers faiss-cpu numpy pickle
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILENAME = "statutes.index"
TEXTS_FILENAME = "statutes.pkl"
EMBEDDING_BATCH_SIZE = 128


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATUTES_DIR = PROJECT_ROOT / "data" / "statutes"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
INDEX_PATH = VECTOR_DB_DIR / INDEX_FILENAME
TEXTS_PATH = VECTOR_DB_DIR / TEXTS_FILENAME


_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_statutes: Optional[List[Dict[str, str]]] = None


@dataclass(frozen=True)
class StatuteRecord:
	filename: str
	title: str
	description: str

	@property
	def combined_text(self) -> str:
		return f"{self.title.strip()}\n{self.description.strip()}".strip()


def _get_model() -> SentenceTransformer:
	global _model
	if _model is None:
		print(f"[statute_retriever] Loading embedding model: {MODEL_NAME} (CPU only)...")
		_model = SentenceTransformer(MODEL_NAME, device="cpu")
		print("[statute_retriever] Model loaded.")
	return _model


def _parse_statute_text(raw_text: str) -> tuple[str, str]:
	lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
	if not lines:
		return "", ""

	title = ""
	description_lines: List[str] = []
	mode = None

	for line in lines:
		lower = line.lower()
		if lower.startswith("title:"):
			title = line.split(":", 1)[1].strip()
			mode = "description"
			continue
		if lower.startswith("description:"):
			content = line.split(":", 1)[1].strip()
			if content:
				description_lines.append(content)
			mode = "description"
			continue

		if mode == "description":
			description_lines.append(line)
		elif not title:
			title = line
		else:
			description_lines.append(line)

	if not title:
		title = lines[0]
	if not description_lines and len(lines) > 1:
		description_lines = lines[1:]

	return title.strip(), "\n".join(description_lines).strip()


def _load_statute_records(statutes_dir: Path = STATUTES_DIR) -> List[StatuteRecord]:
	if not statutes_dir.exists():
		raise FileNotFoundError(f"Statutes directory not found: {statutes_dir}")

	files = sorted(statutes_dir.glob("*.txt"))
	if not files:
		raise FileNotFoundError(f"No .txt statute files found in: {statutes_dir}")

	print(f"[statute_retriever] Found {len(files)} statute files. Loading text...")
	records: List[StatuteRecord] = []

	for idx, file_path in enumerate(files, start=1):
		raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
		title, description = _parse_statute_text(raw_text)
		records.append(
			StatuteRecord(
				filename=file_path.name,
				title=title,
				description=description,
			)
		)

		if idx % 500 == 0 or idx == len(files):
			print(f"[statute_retriever] Loaded {idx}/{len(files)} statutes...")

	print("[statute_retriever] Completed loading statute texts.")
	return records


def _embed_texts(texts: Sequence[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
	model = _get_model()
	print(f"[statute_retriever] Generating embeddings for {len(texts)} statutes...")
	embeddings = model.encode(
		list(texts),
		batch_size=batch_size,
		show_progress_bar=True,
		convert_to_numpy=True,
		normalize_embeddings=True,
	)
	embeddings = np.asarray(embeddings, dtype=np.float32)
	print(f"[statute_retriever] Embeddings generated with shape: {embeddings.shape}")
	return embeddings


def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
	if embeddings.ndim != 2:
		raise ValueError("Embeddings must be a 2D array of shape (n_samples, embedding_dim)")

	embedding_dim = embeddings.shape[1]
	print(f"[statute_retriever] Building FAISS index (dim={embedding_dim})...")
	index = faiss.IndexFlatIP(embedding_dim)
	index.add(embeddings)
	print(f"[statute_retriever] FAISS index built with {index.ntotal} vectors.")
	return index


def build_and_save_statute_index(
	statutes_dir: Path = STATUTES_DIR,
	index_path: Path = INDEX_PATH,
	texts_path: Path = TEXTS_PATH,
) -> None:
	print("[statute_retriever] Starting statute indexing pipeline...")
	records = _load_statute_records(statutes_dir)
	combined_texts = [record.combined_text for record in records]
	embeddings = _embed_texts(combined_texts)
	index = _build_faiss_index(embeddings)

	index_path.parent.mkdir(parents=True, exist_ok=True)
	print(f"[statute_retriever] Saving FAISS index to: {index_path}")
	faiss.write_index(index, str(index_path))

	serialized_records = [
		{
			"filename": record.filename,
			"title": record.title,
			"description": record.description,
			"text": record.combined_text,
		}
		for record in records
	]
	print(f"[statute_retriever] Saving statute texts to: {texts_path}")
	with texts_path.open("wb") as fp:
		pickle.dump(serialized_records, fp)

	print("[statute_retriever] Indexing pipeline completed successfully.")


def _load_saved_index_and_texts(
	index_path: Path = INDEX_PATH,
	texts_path: Path = TEXTS_PATH,
) -> tuple[faiss.Index, List[Dict[str, str]]]:
	if not index_path.exists() or not texts_path.exists():
		raise FileNotFoundError(
			"Saved statute index artifacts not found. "
			"Run build_and_save_statute_index() first."
		)

	print(f"[statute_retriever] Loading FAISS index from: {index_path}")
	index = faiss.read_index(str(index_path))

	print(f"[statute_retriever] Loading statutes from: {texts_path}")
	with texts_path.open("rb") as fp:
		statutes = pickle.load(fp)

	if len(statutes) != index.ntotal:
		raise ValueError(
			f"Index/statute count mismatch: index={index.ntotal}, statutes={len(statutes)}"
		)

	return index, statutes


def _ensure_loaded() -> tuple[faiss.Index, List[Dict[str, str]]]:
	global _index, _statutes
	if _index is None or _statutes is None:
		_index, _statutes = _load_saved_index_and_texts()
	return _index, _statutes


def initialize_statute_retriever(
	statutes_dir: Path = STATUTES_DIR,
	index_path: Path = INDEX_PATH,
	texts_path: Path = TEXTS_PATH,
) -> Dict[str, object]:
	global _index, _statutes

	created = False
	if not index_path.exists() or not texts_path.exists():
		print("[statute_retriever] Index artifacts missing. Rebuilding statute embeddings...")
		build_and_save_statute_index(
			statutes_dir=statutes_dir,
			index_path=index_path,
			texts_path=texts_path,
		)
		created = True

	_index, _statutes = _load_saved_index_and_texts(index_path=index_path, texts_path=texts_path)

	return {
		"created": created,
		"statute_count": len(_statutes),
		"index_path": str(index_path),
		"texts_path": str(texts_path),
	}


def retrieve_statutes(query: str, top_k: int = 3) -> List[Dict[str, object]]:
	if not query or not query.strip():
		raise ValueError("Query must be a non-empty string.")
	if top_k <= 0:
		raise ValueError("top_k must be greater than 0.")

	index, statutes = _ensure_loaded()
	model = _get_model()

	print(f"[statute_retriever] Retrieving top {top_k} statutes for query...")
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
		statute = statutes[idx]
		results.append(
			{
				"rank": len(results) + 1,
				"score": float(score),
				"filename": statute.get("filename", ""),
				"title": statute.get("title", ""),
				"description": statute.get("description", ""),
				"text": statute.get("text", ""),
			}
		)

	print(f"[statute_retriever] Retrieved {len(results)} statutes.")
	return results


if __name__ == "__main__":
	build_and_save_statute_index()
