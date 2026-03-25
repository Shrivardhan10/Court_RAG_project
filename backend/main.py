import re

from agents.defense import run_defense_agent
from agents.judge import run_judge_agent
from agents.prosecution import run_prosecution_agent
from retrieval import precedent_retriever, statute_retriever
from utils.llm import generate_response
from utils.prompts import get_fact_extraction_prompt


INPUT_CASE_TEXT = (
    "The patient died during a routine surgical procedure. "
    "It is alleged that the doctor failed to follow standard protocols and administered incorrect medication. "
    "The defence claims it was an unforeseen complication."
)


def _parse_fact_fields(facts: str) -> dict[str, str]:
	fields = {"act": "", "intent": "", "weapon": "", "outcome": ""}
	for line in (facts or "").splitlines():
		if ":" not in line:
			continue
		key, value = line.split(":", 1)
		k = key.strip().lower()
		if k in fields:
			fields[k] = value.strip()
	return fields


def _is_low_quality_facts(facts: str) -> bool:
	text = (facts or "").strip()
	if not text:
		return True
	fields = _parse_fact_fields(text)
	missing = sum(1 for value in fields.values() if not value)
	if missing >= 2:
		return True
	if text.lower().count("act:") > 2:
		return True
	if len(text) < 40:
		return True
	return False


def _extract_facts_with_fallback(case_text: str) -> str:
	fact_prompt = get_fact_extraction_prompt(case_text)
	facts = generate_response(fact_prompt)
	if not _is_low_quality_facts(facts):
		return facts

	text = (case_text or "").strip()
	lower = text.lower()

	intent = "Intent unclear from available text."
	if any(token in lower for token in ["intentional", "deliberate", "premeditated"]):
		intent = "Indications of intentional conduct are present."
	elif any(token in lower for token in ["negligent", "accident", "reckless"]):
		intent = "Possible negligence or absence of specific intent."

	weapon = "No specific weapon identified."
	for candidate in ["iron rod", "knife", "gun", "pistol", "stick", "stone"]:
		if candidate in lower:
			weapon = candidate
			break

	outcome = "Outcome not clearly stated."
	if any(token in lower for token in ["died", "death", "dead", "fatal"]):
		outcome = "Victim death is alleged."
	elif any(token in lower for token in ["injury", "injured", "hurt"]):
		outcome = "Physical injury is alleged."

	act = text[:220].strip()
	return (
		f"act: {act}\n"
		f"intent: {intent}\n"
		f"weapon: {weapon}\n"
		f"outcome: {outcome}"
	)


def _infer_issue_tags(facts: str) -> list[str]:
	text = (facts or "").lower()
	tags: list[str] = []

	if "intent" in text or "intentional" in text:
		tags.append("intent")
	if "weapon" in text or "rod" in text or "knife" in text or "gun" in text:
		tags.append("weapon-use")
	if "died" in text or "death" in text or "fatal" in text:
		tags.append("causation-of-death")
	if "witness" in text:
		tags.append("witness-credibility")
	if not tags:
		tags.append("general-criminal-liability")

	return tags


def _extract_case_name(preview: str, case_id: str) -> str:
	text = (preview or "").replace("\n", " ").strip()
	if not text:
		return case_id.replace(".txt", "")

	# Keep only the caption segment before court/procedural metadata.
	for marker in [" Supreme Court", " High Court", " Appeal", " W.P.", " Cr.A."]:
		if marker in text:
			text = text.split(marker, 1)[0].strip()
			break

	if len(text) < 8:
		return case_id.replace(".txt", "")
	return text[:120]


def _format_statutes_for_prompt(statutes: list[dict]) -> str:
	if not statutes:
		return "No statutes retrieved."
	parts = []
	for statute in statutes[:3]:
		statute_id = statute.get("filename", "")
		title = statute.get("title", "")
		description = (statute.get("description", "") or "").replace("\n", " ").strip()
		parts.append(f"- {statute_id} | {title}: {description[:180]}")
	return "\n".join(parts)


def _format_precedents_for_prompt(precedents: list[dict], facts: str) -> tuple[str, dict[str, str]]:
	if not precedents:
		return "No precedents retrieved.", {}

	issue_tags = ", ".join(_infer_issue_tags(facts))
	parts = []
	citation_map: dict[str, str] = {}
	for rank, precedent in enumerate(precedents[:3], start=1):
		case_id = precedent.get("case_id", "")
		score = float(precedent.get("score", 0.0))
		case_name = _extract_case_name(precedent.get("text_preview", ""), case_id)
		citation = f"P{rank}"
		citation_map[case_id] = citation
		parts.append(
			f"- [{citation}] {case_name} (similarity={score:.4f}) | Potential relevance: {issue_tags}"
		)

	return "\n".join(parts), citation_map


def _sanitize_agent_output(text: str) -> str:
	if not text:
		return text
	clean = text
	clean = re.sub(r"C\d+\.txt", "precedent", clean)
	clean = re.sub(r"\b\.txt\b", "", clean)
	clean = re.sub(r"\(similarity=[^)]+\)", "", clean)
	clean = re.sub(r"\|\s*Potential relevance:[^\n]*", "", clean)
	clean = clean.replace("rank=", "similarity-rank=")
	# Remove repeated procedural noise.
	for marker in ["Supreme Court of India", "Appeal (Crl.)", "The Judgment was delivered by"]:
		clean = clean.replace(marker, "")
	return re.sub(r"\s+", " ", clean).strip()


def _looks_non_argumentative(text: str) -> bool:
	content = (text or "").strip().lower()
	if not content:
		return True
	if len(content) < 80:
		return True
	if any(token in content for token in ["rank=", ".txt", "score="]):
		return True
	if "potential relevance:" in content:
		return True
	if content.count("[p") >= 2 and content.count("\n") <= 1:
		return True
	if content.count("is the name of a weapon") >= 2:
		return True
	return False


def _extract_precedent_labels(precedents: str) -> list[str]:
	labels = re.findall(r"\[(P\d+)\]", precedents or "")
	# Preserve order, remove duplicates.
	seen: set[str] = set()
	unique: list[str] = []
	for label in labels:
		if label in seen:
			continue
		seen.add(label)
		unique.append(label)
	return unique


def _extract_statute_labels(statutes: str) -> list[str]:
	labels: list[str] = []
	for line in (statutes or "").splitlines():
		match = re.search(r"-\s*([^|:]+)", line)
		if not match:
			continue
		raw = match.group(1).strip()
		labels.append(raw.replace(".txt", ""))
	return labels[:3]


def _has_section_markers(text: str, markers: list[str]) -> bool:
	content = (text or "").lower()
	return all(marker.lower() in content for marker in markers)


def _is_valid_prosecution_output(text: str) -> bool:
	if _looks_non_argumentative(text):
		return False
	return _has_section_markers(
		text,
		["charges", "key evidence", "statutory basis", "precedent support", "conclusion"],
	)


def _is_valid_defense_output(text: str) -> bool:
	if _looks_non_argumentative(text):
		return False
	return _has_section_markers(
		text,
		[
			"weaknesses in prosecution case",
			"statutory counter-interpretation",
			"precedent distinguishing",
			"mitigating factors",
			"relief sought",
		],
	)


def _is_valid_judge_output(text: str) -> bool:
	if _looks_non_argumentative(text):
		return False
	return _has_section_markers(
		text,
		[
			"findings",
			"reasoning",
			"accepted/rejected citations",
			"final verdict",
			"suggested sentence/relief",
		],
	)


def _build_prosecution_fallback(facts: str, statutes: str, precedents: str) -> str:
	fields = _parse_fact_fields(facts)
	statute_labels = _extract_statute_labels(statutes)
	precedent_labels = _extract_precedent_labels(precedents)
	statute_ref = ", ".join(statute_labels) if statute_labels else "retrieved statutes"
	precedent_ref = ", ".join(f"[{label}]" for label in precedent_labels) if precedent_labels else "retrieved precedents"
	return (
		"1) Charges\n"
		"The prosecution alleges serious violent liability based on the factual record and seeks conviction on the principal charge.\n\n"
		"2) Key Evidence from Facts\n"
		f"- Act: {fields.get('act') or 'Assaultive conduct is alleged.'}\n"
		f"- Intent: {fields.get('intent') or 'Intent is inferred from repeated targeted acts.'}\n"
		f"- Weapon: {fields.get('weapon') or 'A harmful instrument appears to be used.'}\n"
		f"- Outcome: {fields.get('outcome') or 'Serious consequence is alleged.'}\n\n"
		"3) Statutory Basis\n"
		f"The prosecution relies on {statute_ref} to establish unlawful assault, culpable intent, and punishment.\n\n"
		"4) Precedent Support\n"
		f"The prosecution cites {precedent_ref} for principles on intent inference, weapon use, and causation from the proved facts.\n\n"
		"5) Conclusion\n"
		"Given the alleged intentional assault, weapon use, and grave outcome, conviction on the primary charge is sought."
	)


def _build_defense_fallback(facts: str, statutes: str, precedents: str, prosecution_output: str) -> str:
	fields = _parse_fact_fields(facts)
	statute_labels = _extract_statute_labels(statutes)
	precedent_labels = _extract_precedent_labels(precedents)
	statute_ref = ", ".join(statute_labels) if statute_labels else "retrieved statutes"
	precedent_ref = ", ".join(f"[{label}]" for label in precedent_labels) if precedent_labels else "retrieved precedents"
	return (
		"1) Weaknesses in Prosecution Case\n"
		"The prosecution has not conclusively established the highest threshold of intent and causation beyond doubt.\n\n"
		"2) Statutory Counter-Interpretation\n"
		f"The same statutory material ({statute_ref}) may support a lesser charge depending on mens rea and causation findings.\n\n"
		"3) Precedent Distinguishing\n"
		f"Cited precedents ({precedent_ref}) should be treated cautiously because factual alignment on intention, context, and causation is not exact.\n\n"
		"4) Mitigating Factors\n"
		f"Possible mitigation includes dispute context and ambiguity regarding intention ({fields.get('intent') or 'intent disputed'}).\n\n"
		"5) Relief Sought\n"
		"Acquittal on the most serious charge or, alternatively, conviction on a lesser offence with proportional sentencing."
	)


def _build_judge_fallback(
	facts: str,
	statutes: str,
	precedents: str,
	prosecution_output: str,
	defense_output: str,
) -> str:
	statute_labels = _extract_statute_labels(statutes)
	precedent_labels = _extract_precedent_labels(precedents)
	statute_ref = ", ".join(statute_labels) if statute_labels else "retrieved statutes"
	precedent_ref = ", ".join(f"[{label}]" for label in precedent_labels) if precedent_labels else "retrieved precedents"
	return (
		"Findings:\n"
		"- The factual matrix supports that a violent act occurred and caused serious harm.\n"
		"- The exact degree of intent remains the central legal issue.\n\n"
		"Reasoning:\n"
		"- The prosecution has stronger footing on occurrence and consequence.\n"
		"- The defense raises a plausible challenge on degree of mens rea and precise charge fit.\n"
		f"- Statutory context considered: {statute_ref}.\n"
		f"- Precedent context considered: {precedent_ref}.\n\n"
		"Accepted/Rejected Citations:\n"
		"- Accepted where factual similarity is substantial; reduced weight where context differs materially.\n"
		f"- Citations reviewed: {precedent_ref}.\n\n"
		"Final Verdict:\n"
		"- Liability is established, with final section selection depending on proven intent threshold.\n\n"
		"Suggested Sentence/Relief:\n"
		"- Proportionate sentence after considering weapon use, harm gravity, and mitigating context."
	)


def _print_statutes(statutes: list[dict]) -> None:
	print("\n========== STATUTES ==========")
	if not statutes:
		print("No statutes retrieved.")
		return

	for statute in statutes:
		description = (statute.get("description", "") or "").replace("\n", " ").strip()
		print(f"\nRank: {statute.get('rank')}")
		print(f"Statute ID: {statute.get('filename', '')}")
		print(f"Title: {statute.get('title', '')}")
		print(f"Description: {description[:200]}")
		print(f"Score: {statute.get('score', 0.0):.4f}")


def _print_precedents(precedents: list[dict]) -> None:
	print("\n========== PRECEDENTS ==========")
	if not precedents:
		print("No precedents retrieved.")
		return

	for rank, precedent in enumerate(precedents, start=1):
		preview = (precedent.get("text_preview", "") or "").replace("\n", " ").strip()
		print(f"\nRank: {rank}")
		print(f"Case ID: {precedent.get('case_id', '')}")
		print(f"Score: {precedent.get('score', 0.0):.4f}")
		print(f"Preview: {preview}")


def run_legal_pipeline(case_text: str) -> None:
	if not case_text or not case_text.strip():
		raise ValueError("Input case text must be a non-empty string.")

	print("[main] Starting legal RAG + agents pipeline (CPU only)...")

	statute_init = statute_retriever.initialize_statute_retriever()
	if statute_init["created"]:
		print("[main] Statute index missing. Built new statute index.")
	else:
		print("[main] Loaded existing statute index.")
	print(f"[main] Statutes loaded: {statute_init['statute_count']}")

	precedent_init = precedent_retriever.initialize_precedent_retriever()
	if precedent_init["created"]:
		print("[main] Precedent index missing. Built new precedent index.")
	else:
		print("[main] Loaded existing precedent index.")
	print(f"[main] Cases loaded: {precedent_init['case_count']}")

	print("[main] Extracting facts using LLM...")
	facts = _extract_facts_with_fallback(case_text)

	print("[main] Retrieving statutes based on extracted facts...")
	statutes = statute_retriever.retrieve_statutes(facts, top_k=3)

	print("[main] Retrieving precedents based on extracted facts...")
	precedents = precedent_retriever.retrieve_precedents(facts, top_k=3)

	statutes_for_prompt = _format_statutes_for_prompt(statutes)
	precedents_for_prompt, _ = _format_precedents_for_prompt(precedents, facts)

	print("[main] Running prosecution agent...")
	prosecution_output = run_prosecution_agent(
		facts=facts,
		statutes=statutes_for_prompt,
		precedents=precedents_for_prompt,
	)
	prosecution_output = _sanitize_agent_output(prosecution_output)
	if not _is_valid_prosecution_output(prosecution_output):
		prosecution_output = _build_prosecution_fallback(
			facts=facts,
			statutes=statutes_for_prompt,
			precedents=precedents_for_prompt,
		)

	print("[main] Running defense agent...")
	defense_output = run_defense_agent(
		facts=facts,
		statutes=statutes_for_prompt,
		precedents=precedents_for_prompt,
		prosecution_output=prosecution_output,
	)
	defense_output = _sanitize_agent_output(defense_output)
	if not _is_valid_defense_output(defense_output):
		defense_output = _build_defense_fallback(
			facts=facts,
			statutes=statutes_for_prompt,
			precedents=precedents_for_prompt,
			prosecution_output=prosecution_output,
		)

	print("[main] Running judge agent...")
	judge_output = run_judge_agent(
		facts=facts,
		prosecution_output=prosecution_output,
		defense_output=defense_output,
		statutes=statutes_for_prompt,
		precedents=precedents_for_prompt,
	)
	judge_output = _sanitize_agent_output(judge_output)
	if not _is_valid_judge_output(judge_output):
		judge_output = _build_judge_fallback(
			facts=facts,
			statutes=statutes_for_prompt,
			precedents=precedents_for_prompt,
			prosecution_output=prosecution_output,
			defense_output=defense_output,
		)

	print("\n========== FACTS ==========")
	print(facts)

	_print_statutes(statutes)
	_print_precedents(precedents)

	print("\n========== PROSECUTION ARGUMENT ==========")
	print(prosecution_output)

	print("\n========== DEFENSE ARGUMENT ==========")
	print(defense_output)

	print("\n========== FINAL VERDICT ==========")
	print(judge_output)


if __name__ == "__main__":
	run_legal_pipeline(INPUT_CASE_TEXT)
