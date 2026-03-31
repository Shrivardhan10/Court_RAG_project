"""Precedent retriever with structured-case FAISS indexing + IPC-aware hybrid filtering."""

from __future__ import annotations

import json
import pickle
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .bm25_index import BM25Index, tokenize
from .hybrid_retriever import rerank_precedent_candidates


MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILENAME = "precedents.index"
METADATA_FILENAME = "precedents.pkl"
BM25_FILENAME = "precedents_bm25.pkl"
EMBEDDING_BATCH_SIZE = 128


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_CASES_DIR = PROJECT_ROOT / "processed_cases"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
INDEX_PATH = VECTOR_DB_DIR / INDEX_FILENAME
METADATA_PATH = VECTOR_DB_DIR / METADATA_FILENAME
BM25_PATH = VECTOR_DB_DIR / BM25_FILENAME


_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_cases: Optional[List[Dict[str, object]]] = None
_bm25: Optional[BM25Index] = None
_model_load_attempted: bool = False

ALLOW_REMOTE_MODEL_DOWNLOAD = os.getenv("ALLOW_REMOTE_MODEL_DOWNLOAD", "0").strip() == "1"


def _get_model() -> Optional[SentenceTransformer]:
	global _model, _model_load_attempted
	if _model is None and not _model_load_attempted:
		_model_load_attempted = True
		print(f"[precedent_retriever] Loading embedding model: {MODEL_NAME} (CPU only)...")
		try:
			_model = SentenceTransformer(
				MODEL_NAME,
				device="cpu",
				local_files_only=not ALLOW_REMOTE_MODEL_DOWNLOAD,
			)
			print("[precedent_retriever] Model loaded.")
		except Exception as exc:
			print(f"[precedent_retriever] Embedding model unavailable, using BM25-only fallback: {exc}")
			_model = None
	return _model


def _extract_ipc_sections(text: str) -> List[str]:
	if not text:
		return []

	found: List[str] = []
	for grp in re.findall(r"\b(\d{1,3}[A-Z]?(?:\s*/\s*\d{1,3}[A-Z]?)+)\s*IPC\b", text, flags=re.IGNORECASE):
		for sec in re.split(r"\s*/\s*", grp):
			sec = sec.strip().upper()
			if sec and sec not in found:
				found.append(sec)

	pattern = r"\b(?:Section|Sections|u/s|under sections?)\s+([0-9]{1,3}[A-Z]?(?:\s*(?:,|and|/|r/w|read with)\s*[0-9]{1,3}[A-Z]?){0,8})\s*(?:IPC|I\.P\.C\.|of IPC)?\b"
	for match in re.finditer(pattern, text, flags=re.IGNORECASE):
		chunk = match.group(1)
		for sec in re.findall(r"\b\d{1,3}[A-Z]?\b", chunk):
			sec = sec.upper()
			if sec not in found:
				found.append(sec)

	def _key(val: str) -> tuple[int, str]:
		m = re.match(r"(\d+)([A-Z]?)", val)
		if not m:
			return (9999, val)
		return (int(m.group(1)), m.group(2))

	return sorted(found, key=_key)


def _build_embedding_text(metadata: Dict[str, object]) -> str:
	facts = str(metadata.get("facts", "") or "").strip()
	actions = metadata.get("key_actions", []) or []
	if not isinstance(actions, list):
		actions = []
	ipc_sections = metadata.get("ipc_sections", []) or []
	if not isinstance(ipc_sections, list):
		ipc_sections = []

	actions_text = ", ".join(str(x) for x in actions if str(x).strip()) or "none"
	sections_text = ", ".join(str(x) for x in ipc_sections if str(x).strip()) or "none"
	return (
		f"Case: {metadata.get('case_name', 'Unknown Case')} | "
		f"Facts: {facts[:700]} | "
		f"Actions: {actions_text} | "
		f"Outcome: {metadata.get('outcome', 'unknown')} | "
		f"IPC Sections: {sections_text} | "
		f"Judgment: {metadata.get('judgment', 'unknown')}"
	)


def _normalize_structured_case(case_id: str, payload: dict) -> Dict[str, object]:
	metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
	if not isinstance(metadata, dict):
		metadata = {}

	case_name = str(metadata.get("case_name", "") or case_id)
	facts = str(metadata.get("facts", "") or "")
	ipc_sections = metadata.get("ipc_sections", []) or []
	if not isinstance(ipc_sections, list):
		ipc_sections = []
	ipc_sections = [str(x).upper().strip() for x in ipc_sections if str(x).strip()]

	entry: Dict[str, object] = {
		"case_id": case_id,
		"case_name": case_name,
		"facts": facts,
		"ipc_sections": ipc_sections,
		"judgment": str(metadata.get("judgment", "unknown") or "unknown"),
		"weapon": str(metadata.get("weapon", "") or ""),
		"intent": str(metadata.get("intent", "unknown") or "unknown"),
		"outcome": str(metadata.get("outcome", "") or ""),
		"key_actions": metadata.get("key_actions", []) if isinstance(metadata.get("key_actions", []), list) else [],
		"chunk_type": str(metadata.get("chunk_type", "facts") or "facts"),
	}

	custom_embedding_text = str(payload.get("embedding_text", "") or "").strip()
	entry["embedding_text"] = custom_embedding_text if custom_embedding_text else _build_embedding_text(entry)
	if not entry["ipc_sections"]:
		entry["ipc_sections"] = _extract_ipc_sections(entry.get("embedding_text", ""))
	return entry


def _normalize_multi_chunks(case_id: str, payload: dict) -> List[Dict[str, object]]:
	case_name = str(payload.get("case_name", "") or case_id)
	chunks = payload.get("chunks", []) if isinstance(payload.get("chunks", []), list) else []
	rows: List[Dict[str, object]] = []

	for idx, chunk in enumerate(chunks, start=1):
		if not isinstance(chunk, dict):
			continue
		metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata", {}), dict) else {}
		chunk_text = str(chunk.get("text", "") or "").strip()
		if not chunk_text:
			continue

		chunk_case_name = str(metadata.get("case_name", "") or case_name)
		ipc_sections = metadata.get("ipc_sections", []) if isinstance(metadata.get("ipc_sections", []), list) else []
		ipc_sections = [str(x).upper().strip() for x in ipc_sections if str(x).strip()]

		embedding_text = str(chunk.get("embedding_text", "") or "").strip()
		if not embedding_text:
			embedding_text = (
				f"Case: {chunk_case_name} | Chunk Type: {metadata.get('chunk_type', chunk.get('chunk_type', 'facts'))} | "
				f"Text: {chunk_text[:1200]} | IPC: {', '.join(ipc_sections) if ipc_sections else 'none'}"
			)

		rows.append(
			{
				"case_id": case_id,
				"case_name": chunk_case_name,
				"chunk_id": f"{case_id}_CH{idx:03d}",
				"chunk_type": str(metadata.get("chunk_type", chunk.get("chunk_type", "facts")) or "facts"),
				"facts": chunk_text,
				"ipc_sections": ipc_sections,
				"judgment": str(metadata.get("judgment", "unknown") or "unknown"),
				"weapon": str(metadata.get("weapon", "") or ""),
				"intent": str(metadata.get("intent", "unknown") or "unknown"),
				"outcome": str(metadata.get("outcome", "") or ""),
				"key_actions": metadata.get("actions", []) if isinstance(metadata.get("actions", []), list) else [],
				"embedding_text": embedding_text,
			}
		)

	return rows


def _normalize_legacy_chunks(case_id: str, chunks: list) -> Dict[str, object]:
	texts = [str(chunk.get("text", "") or "") for chunk in chunks if isinstance(chunk, dict)]
	joined_text = " ".join(t for t in texts if t).strip()
	preview = " ".join(joined_text.split()[:180])

	judgment = "unknown"
	lower = joined_text.lower()
	if "acquitted" in lower:
		judgment = "acquitted"
	elif "convicted" in lower or "found guilty" in lower:
		judgment = "convicted"
	elif "appeal is dismissed" in lower:
		judgment = "dismissed"
	elif "appeal is allowed" in lower:
		judgment = "allowed"

	entry: Dict[str, object] = {
		"case_id": case_id,
		"case_name": case_id,
		"chunk_type": "facts",
		"facts": preview,
		"ipc_sections": _extract_ipc_sections(joined_text),
		"judgment": judgment,
		"weapon": "",
		"intent": "unknown",
		"outcome": "",
		"key_actions": [],
	}
	entry["embedding_text"] = _build_embedding_text(entry)
	return entry


def _load_processed_case_metadata(processed_cases_dir: Path = PROCESSED_CASES_DIR) -> List[Dict[str, object]]:
	if not processed_cases_dir.exists():
		raise FileNotFoundError(f"Processed cases directory not found: {processed_cases_dir}")

	files = sorted(processed_cases_dir.glob("*.json"))
	if not files:
		raise FileNotFoundError(f"No processed case JSON files found in: {processed_cases_dir}")

	records: List[Dict[str, object]] = []
	print(f"[precedent_retriever] Loading {len(files)} processed case files...")
	for idx, file_path in enumerate(files, start=1):
		case_id = file_path.stem
		try:
			payload = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
		except json.JSONDecodeError:
			continue

		if isinstance(payload, dict) and "chunks" in payload:
			records.extend(_normalize_multi_chunks(case_id, payload))
		elif isinstance(payload, dict) and "metadata" in payload:
			records.append(_normalize_structured_case(case_id, payload))
		elif isinstance(payload, list):
			records.append(_normalize_legacy_chunks(case_id, payload))

		if idx % 500 == 0 or idx == len(files):
			print(f"[precedent_retriever] Parsed {idx}/{len(files)} case files...")

	if not records:
		raise ValueError("No valid structured/legacy precedent records found in processed_cases.")
	return records


def _embed_texts(texts: Sequence[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
	model = _get_model()
	if model is None:
		raise RuntimeError("Embedding model unavailable for indexing.")
	print(f"[precedent_retriever] Generating embeddings for {len(texts)} cases...")
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
		raise ValueError("Embeddings must be a 2D array of shape (n_samples, embedding_dim)")

	index = faiss.IndexFlatIP(embeddings.shape[1])
	index.add(embeddings)
	return index


def build_and_save_precedent_index(
	processed_cases_dir: Path = PROCESSED_CASES_DIR,
	index_path: Path = INDEX_PATH,
	metadata_path: Path = METADATA_PATH,
	bm25_path: Path = BM25_PATH,
) -> None:
	print("[precedent_retriever] Starting precedent indexing pipeline...")
	records = _load_processed_case_metadata(processed_cases_dir)
	embeddings = _embed_texts([str(r.get("embedding_text", "")) for r in records])
	index = _build_faiss_index(embeddings)
	bm25_docs = [
		tokenize(
			" ".join(
				[
					str(record.get("embedding_text", "")),
					str(record.get("facts", "")),
					" ".join(str(x) for x in (record.get("ipc_sections", []) or [])),
					" ".join(str(x) for x in (record.get("key_actions", []) or [])),
					str(record.get("intent", "")),
				]
			)
		)
		for record in records
	]
	bm25_index = BM25Index.from_documents(bm25_docs)

	index_path.parent.mkdir(parents=True, exist_ok=True)
	faiss.write_index(index, str(index_path))
	with metadata_path.open("wb") as fp:
		pickle.dump(records, fp)
	with bm25_path.open("wb") as fp:
		pickle.dump(bm25_index, fp)
	print("[precedent_retriever] Precedent index saved successfully.")


def _load_saved_index_and_metadata(
	index_path: Path = INDEX_PATH,
	metadata_path: Path = METADATA_PATH,
	bm25_path: Path = BM25_PATH,
) -> tuple[faiss.Index, List[Dict[str, object]], BM25Index]:
	if not index_path.exists() or not metadata_path.exists() or not bm25_path.exists():
		raise FileNotFoundError("Precedent index artifacts missing. Run build_and_save_precedent_index().")

	index = faiss.read_index(str(index_path))
	with metadata_path.open("rb") as fp:
		records = pickle.load(fp)
	with bm25_path.open("rb") as fp:
		bm25_index = pickle.load(fp)

	if len(records) != index.ntotal:
		raise ValueError(f"Index/metadata mismatch: index={index.ntotal}, records={len(records)}")
	if not isinstance(bm25_index, BM25Index):
		raise TypeError("Invalid BM25 artifact for precedents index.")
	return index, records, bm25_index


def initialize_precedent_retriever(
	processed_cases_dir: Path = PROCESSED_CASES_DIR,
	index_path: Path = INDEX_PATH,
	metadata_path: Path = METADATA_PATH,
	bm25_path: Path = BM25_PATH,
) -> Dict[str, object]:
	global _index, _cases, _bm25

	created = False
	if not index_path.exists() or not metadata_path.exists() or not bm25_path.exists():
		build_and_save_precedent_index(
			processed_cases_dir=processed_cases_dir,
			index_path=index_path,
			metadata_path=metadata_path,
			bm25_path=bm25_path,
		)
		created = True

	_index, _cases, _bm25 = _load_saved_index_and_metadata(index_path=index_path, metadata_path=metadata_path, bm25_path=bm25_path)
	return {
		"created": created,
		"case_count": len(_cases),
		"index_path": str(index_path),
		"metadata_path": str(metadata_path),
		"bm25_path": str(bm25_path),
	}


def _ensure_loaded() -> tuple[faiss.Index, List[Dict[str, object]], BM25Index]:
	global _index, _cases, _bm25
	if _index is None or _cases is None or _bm25 is None:
		_index, _cases, _bm25 = _load_saved_index_and_metadata()
	return _index, _cases, _bm25


def retrieve_precedents(
	query: str,
	top_k: int = 3,
	side_hint: Optional[str] = None,
	allowed_sections: Optional[Sequence[str]] = None,
	ipc_sections: Optional[Sequence[str]] = None,
	require_ipc_match: bool = False,
	use_cross_encoder: bool = False,
) -> List[Dict[str, object]]:
	del side_hint, allowed_sections

	if not query or not query.strip():
		raise ValueError("Query must be a non-empty string.")
	if top_k <= 0:
		raise ValueError("top_k must be > 0.")
	result_count = max(3, top_k)

	index, records, bm25_index = _ensure_loaded()
	query_text = query.strip()
	query_ipc = [str(x).upper().strip() for x in (ipc_sections or _extract_ipc_sections(query_text)) if str(x).strip()]

	model = _get_model()
	semantic_candidates: List[tuple[int, float]] = []
	if model is not None:
		query_embedding = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
		query_embedding = np.asarray(query_embedding, dtype=np.float32)

		search_k = min(max(result_count * 10, 20), index.ntotal)
		scores, indices = index.search(query_embedding, search_k)
		semantic_candidates = [(int(idx), float(score)) for score, idx in zip(scores[0], indices[0]) if idx >= 0][:20]
	bm25_candidates = bm25_index.top_n(tokenize(query_text), n=min(20, len(records)))

	results = rerank_precedent_candidates(
		query=query_text,
		records=records,
		semantic_candidates=semantic_candidates,
		bm25_candidates=bm25_candidates,
		top_k=result_count,
		use_cross_encoder=use_cross_encoder,
	)

	if require_ipc_match and query_ipc:
		qset = set(query_ipc)
		filtered = [
			row for row in results
			if qset.intersection({str(x).upper().strip() for x in (row.get("ipc_sections", []) or []) if str(x).strip()})
		]
		if filtered:
			results = filtered

	return results[:result_count]


if __name__ == "__main__":
	build_and_save_precedent_index()
