from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bm25_index import normalize_scores


_HOMICIDE_SECTIONS = {"299", "300", "302", "304", "304A"}
_FRAUD_SECTIONS = {"405", "406", "409", "415", "417", "418", "420", "467", "468", "471"}
_INJURY_SECTIONS = {"319", "320", "321", "323", "324", "325", "326", "307"}


def _extract_query_sections(query: str) -> set[str]:
	return set(re.findall(r"\b\d{1,3}[A-Z]?\b", query or ""))


def _chunk_type_boost(chunk_type: str) -> float:
	ct = (chunk_type or "").lower()
	if ct == "argument":
		return 0.15
	if ct == "reasoning":
		return 0.10
	if ct == "judgment":
		return 0.05
	return 0.0


def _ipc_overlap_score(query_ipc: set[str], record_ipc: Sequence[str]) -> float:
	if not query_ipc:
		return 0.0
	record_set = {str(x).upper().strip() for x in record_ipc if str(x).strip()}
	overlap = query_ipc.intersection(record_set)
	if not overlap:
		return 0.0
	return min(1.0, len(overlap) / max(1, len(query_ipc)))


def rerank_precedent_candidates(
	query: str,
	records: Sequence[Dict[str, object]],
	semantic_candidates: Sequence[Tuple[int, float]],
	bm25_candidates: Sequence[Tuple[int, float]],
	top_k: int,
	use_cross_encoder: bool = False,
) -> List[Dict[str, object]]:
	del use_cross_encoder
	query_ipc = {x.upper() for x in _extract_query_sections(query)}

	semantic_map: Dict[int, float] = {idx: float(score) for idx, score in semantic_candidates}
	bm25_map: Dict[int, float] = {idx: float(score) for idx, score in bm25_candidates}
	candidate_ids = sorted(set(semantic_map.keys()).union(bm25_map.keys()))
	if not candidate_ids:
		return []

	semantic_norm_values = normalize_scores([semantic_map.get(i, 0.0) for i in candidate_ids])
	bm25_norm_values = normalize_scores([bm25_map.get(i, 0.0) for i in candidate_ids])
	semantic_norm = {idx: float(score) for idx, score in zip(candidate_ids, semantic_norm_values)}
	bm25_norm = {idx: float(score) for idx, score in zip(candidate_ids, bm25_norm_values)}

	scored_rows: List[Dict[str, object]] = []
	for idx in candidate_ids:
		record = records[idx]
		record_ipc = [str(x).upper().strip() for x in (record.get("ipc_sections", []) or []) if str(x).strip()]
		overlap_score = _ipc_overlap_score(query_ipc, record_ipc)
		chunk_type = str(record.get("chunk_type", "facts"))

		final_score = (
			0.6 * semantic_norm.get(idx, 0.0)
			+ 0.3 * bm25_norm.get(idx, 0.0)
			+ 0.1 * overlap_score
			+ _chunk_type_boost(chunk_type)
		)

		scored_rows.append(
			{
				"record_index": idx,
				"final_score": float(final_score),
				"semantic_score": float(semantic_norm.get(idx, 0.0)),
				"bm25_score": float(bm25_norm.get(idx, 0.0)),
				"ipc_overlap_score": float(overlap_score),
			}
		)

	scored_rows.sort(key=lambda x: x["final_score"], reverse=True)

	deduped: List[Dict[str, object]] = []
	seen = set()
	for row in scored_rows:
		record = records[row["record_index"]]
		key = f"{record.get('case_name', '')}|{record.get('chunk_type', 'facts')}"
		if key in seen:
			continue
		seen.add(key)
		payload = {
			"case_name": str(record.get("case_name", "")),
			"chunk_type": str(record.get("chunk_type", "facts")),
			"facts": str(record.get("facts", "")),
			"judgment": str(record.get("judgment", "unknown")),
			"ipc_sections": [str(x).upper().strip() for x in (record.get("ipc_sections", []) or []) if str(x).strip()],
			"actions": record.get("key_actions", []) if isinstance(record.get("key_actions", []), list) else [],
			"weapon": str(record.get("weapon", "")),
			"intent": str(record.get("intent", "unknown")),
			"outcome": str(record.get("outcome", "")),
			"score": float(row["final_score"]),
			"semantic_score": float(row["semantic_score"]),
			"bm25_score": float(row["bm25_score"]),
			"ipc_overlap_score": float(row["ipc_overlap_score"]),
		}
		deduped.append(payload)
		if len(deduped) >= max(3, top_k):
			break

	return deduped


def _domain_bonus(query: str, section: str, title: str, description: str) -> float:
	query_low = query.lower()
	target = f"{title} {description}".lower()
	bonus = 0.0

	if section in _extract_query_sections(query):
		bonus += 0.5

	if "murder" in query_low or "death" in query_low or "stab" in query_low or "knife" in query_low:
		if section in _HOMICIDE_SECTIONS:
			bonus += 0.35
	if "fraud" in query_low or "cheat" in query_low or "forgery" in query_low:
		if section in _FRAUD_SECTIONS:
			bonus += 0.35
	if "hurt" in query_low or "injury" in query_low or "assault" in query_low:
		if section in _INJURY_SECTIONS:
			bonus += 0.25

	query_terms = set(re.findall(r"[a-zA-Z]{3,}", query_low))
	term_overlap = sum(1 for term in query_terms if term in target)
	bonus += min(0.2, term_overlap * 0.01)

	return bonus


def rerank_statute_candidates(
	query: str,
	records: Sequence[Dict[str, object]],
	semantic_candidates: Sequence[Tuple[int, float]],
	bm25_candidates: Sequence[Tuple[int, float]],
	top_k: int,
) -> List[Dict[str, object]]:
	semantic_map: Dict[int, float] = {idx: float(score) for idx, score in semantic_candidates}
	bm25_map: Dict[int, float] = {idx: float(score) for idx, score in bm25_candidates}
	candidate_ids = sorted(set(semantic_map.keys()).union(bm25_map.keys()))
	if not candidate_ids:
		return []

	semantic_norm_values = normalize_scores([semantic_map.get(i, 0.0) for i in candidate_ids])
	bm25_norm_values = normalize_scores([bm25_map.get(i, 0.0) for i in candidate_ids])
	semantic_norm = {idx: float(score) for idx, score in zip(candidate_ids, semantic_norm_values)}
	bm25_norm = {idx: float(score) for idx, score in zip(candidate_ids, bm25_norm_values)}

	scored_rows: List[Dict[str, object]] = []
	for idx in candidate_ids:
		record = records[idx]
		section = str(record.get("section", "")).strip()
		title = str(record.get("title", ""))
		description = str(record.get("description", ""))
		domain_bonus = _domain_bonus(query, section, title, description)
		final_score = (
			0.5 * semantic_norm.get(idx, 0.0)
			+ 0.3 * bm25_norm.get(idx, 0.0)
			+ 0.2 * min(1.0, domain_bonus)
		)
		if section in _extract_query_sections(query):
			final_score += 0.5

		scored_rows.append(
			{
				"record_index": idx,
				"final_score": float(final_score),
				"semantic_score": float(semantic_norm.get(idx, 0.0)),
				"bm25_score": float(bm25_norm.get(idx, 0.0)),
				"domain_bonus": float(domain_bonus),
			}
		)

	scored_rows.sort(key=lambda x: x["final_score"], reverse=True)
	results: List[Dict[str, object]] = []
	seen_sections = set()
	for row in scored_rows:
		record = records[row["record_index"]]
		section = str(record.get("section", "")).strip()
		if not section or section in seen_sections:
			continue
		seen_sections.add(section)
		results.append(
			{
				"section": section,
				"title": str(record.get("title", "")),
				"description": str(record.get("description", "")),
				"chapter": str(record.get("chapter", "")),
				"chapter_title": str(record.get("chapter_title", "")),
				"score": float(row["final_score"]),
				"semantic_score": float(row["semantic_score"]),
				"bm25_score": float(row["bm25_score"]),
				"domain_bonus": float(row["domain_bonus"]),
			}
		)
		if len(results) >= max(3, top_k):
			break

	return results
