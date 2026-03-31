from __future__ import annotations

from typing import Dict, List

from .extractor import (
	build_metadata_template,
	sentence_role_score,
	sentence_tokenize,
)

TARGET_MIN_TOKENS = 120
TARGET_MAX_TOKENS = 450
TARGET_TARGET_TOKENS = 340
ROLE_ORDER = ["facts", "issue", "argument", "reasoning", "judgment"]


def _token_count(text: str) -> int:
	return len((text or "").split())


def _build_chunk_text(sentences: List[str], fallback_sentences: List[str]) -> str:
	if not sentences:
		sentences = fallback_sentences[:8]

	selected: List[str] = []
	tokens = 0
	for sentence in sentences:
		sentence_tokens = _token_count(sentence)
		if selected and tokens + sentence_tokens > TARGET_MAX_TOKENS:
			break
		selected.append(sentence)
		tokens += sentence_tokens
		if tokens >= TARGET_TARGET_TOKENS:
			break

	if tokens < TARGET_MIN_TOKENS:
		for sentence in fallback_sentences:
			if sentence in selected:
				continue
			sentence_tokens = _token_count(sentence)
			if selected and tokens + sentence_tokens > TARGET_MAX_TOKENS:
				break
			selected.append(sentence)
			tokens += sentence_tokens
			if tokens >= TARGET_MIN_TOKENS:
				break

	return " ".join(selected).strip()


def _embedding_text(chunk_type: str, text: str, metadata: Dict[str, object]) -> str:
	ipc = ", ".join(metadata.get("ipc_sections", [])) if metadata.get("ipc_sections") else "none"
	actions = ", ".join(metadata.get("actions", [])) if metadata.get("actions") else "none"
	weapon = metadata.get("weapon", "unknown")
	outcome = metadata.get("outcome", "unknown")
	intent = metadata.get("intent", "unknown")
	judgment = metadata.get("judgment", "unknown")
	case_name = metadata.get("case_name", "Unknown Case")

	if chunk_type == "facts":
		return (
			f"FACT_CHUNK | Case: {case_name} | Facts: {text[:1000]} | "
			f"Actions: {actions} | Weapon: {weapon} | Outcome: {outcome} | IPC: {ipc}"
		)
	if chunk_type == "argument":
		return (
			f"ARGUMENT_CHUNK | Case: {case_name} | Arguments: {text[:1000]} | "
			f"Intent: {intent} | IPC: {ipc} | Judgment context: {judgment}"
		)
	if chunk_type == "reasoning":
		return (
			f"REASONING_CHUNK | Case: {case_name} | Judicial reasoning: {text[:1000]} | "
			f"Intent: {intent} | Outcome: {outcome} | IPC: {ipc}"
		)
	if chunk_type == "judgment":
		return (
			f"JUDGMENT_CHUNK | Case: {case_name} | Final judgment: {text[:1000]} | "
			f"Judgment: {judgment} | IPC: {ipc}"
		)
	if chunk_type == "issue":
		return (
			f"ISSUE_CHUNK | Case: {case_name} | Legal issues: {text[:1000]} | "
			f"IPC: {ipc}"
		)

	return f"{chunk_type.upper()}_CHUNK | Case: {case_name} | {text[:1000]}"


def build_multi_granular_chunks(case_name: str, full_text: str) -> List[Dict[str, object]]:
	sentences = sentence_tokenize(full_text)
	if not sentences:
		return []

	buckets: Dict[str, List[str]] = {role: [] for role in ROLE_ORDER}
	total = len(sentences)
	for idx, sentence in enumerate(sentences):
		scores = sentence_role_score(sentence, idx, total)
		best_role = max(scores, key=lambda role: scores[role])
		if scores[best_role] <= 0.05:
			best_role = "facts"
		buckets[best_role].append(sentence)

	chunks: List[Dict[str, object]] = []
	base_meta = build_metadata_template(case_name, full_text)

	for role in ROLE_ORDER:
		text = _build_chunk_text(buckets.get(role, []), sentences)
		if not text or len(text.split()) < 40:
			continue
		metadata = dict(base_meta)
		metadata["chunk_type"] = role
		chunk = {
			"chunk_type": role,
			"text": text,
			"embedding_text": _embedding_text(role, text, metadata),
			"metadata": metadata,
		}
		chunks.append(chunk)

	if base_meta.get("ipc_sections"):
		ipc_meta = dict(base_meta)
		ipc_meta["chunk_type"] = "ipc_summary"
		ipc_text = (
			f"Applicable IPC Sections: {', '.join(ipc_meta['ipc_sections'])}. "
			f"Actions: {', '.join(ipc_meta['actions']) if ipc_meta['actions'] else 'none'}. "
			f"Weapon: {ipc_meta['weapon']}. Outcome: {ipc_meta['outcome']}. Judgment: {ipc_meta['judgment']}."
		)
		if len(ipc_text.split()) >= 20:
			chunks.append(
				{
					"chunk_type": "ipc_summary",
					"text": ipc_text,
					"embedding_text": _embedding_text("judgment", ipc_text, ipc_meta),
					"metadata": ipc_meta,
				}
			)

	if len(chunks) < 4:
		# enforce minimum chunk count by splitting full narrative windows
		window = []
		window_tokens = 0
		for sentence in sentences:
			st = len(sentence.split())
			if window and window_tokens + st > TARGET_MAX_TOKENS:
				chunk_text = " ".join(window).strip()
				meta = dict(base_meta)
				meta["chunk_type"] = "facts"
				chunks.append({
					"chunk_type": "facts",
					"text": chunk_text,
					"embedding_text": _embedding_text("facts", chunk_text, meta),
					"metadata": meta,
				})
				window = [sentence]
				window_tokens = st
			else:
				window.append(sentence)
				window_tokens += st
		if window:
			chunk_text = " ".join(window).strip()
			meta = dict(base_meta)
			meta["chunk_type"] = "facts"
			chunks.append(
				{
					"chunk_type": "facts",
					"text": chunk_text,
					"embedding_text": _embedding_text("facts", chunk_text, meta),
					"metadata": meta,
				}
			)

	# dedupe and keep informative chunks only
	final_chunks: List[Dict[str, object]] = []
	seen_text = set()
	for chunk in chunks:
		text = str(chunk.get("text", "")).strip()
		if len(text.split()) < 35:
			continue
		key = " ".join(text.lower().split()[:30])
		if key in seen_text:
			continue
		seen_text.add(key)
		final_chunks.append(chunk)

	return final_chunks[:6]
