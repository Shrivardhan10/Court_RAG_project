from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

from retrieval.precedent_retriever import initialize_precedent_retriever, retrieve_precedents
from retrieval.statute_retriever import initialize_statute_retriever, retrieve_statutes
from utils.llm import generate_response


MAX_REWRITE_ATTEMPTS = 2

_NUMBER_WORDS = {
	"one": 1,
	"two": 2,
	"three": 3,
	"four": 4,
	"five": 5,
	"six": 6,
	"seven": 7,
	"eight": 8,
	"nine": 9,
	"ten": 10,
	"eleven": 11,
	"twelve": 12,
	"thirteen": 13,
	"fourteen": 14,
	"fifteen": 15,
	"sixteen": 16,
	"seventeen": 17,
	"eighteen": 18,
	"nineteen": 19,
	"twenty": 20,
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

try:
	from stage0.pipeline import process_text as process_text_stage0
	from stage1.pipeline import process_sentences_stage1
	_STAGE_PREPROCESSING_AVAILABLE = True
except Exception as _stage_import_error:
	_STAGE_PREPROCESSING_AVAILABLE = False
 


def _extract_ipc_sections(text: str) -> List[str]:
	found: List[str] = []
	if not text:
		return found

	section_first_pattern = r"\b(?:Section|Sections|u/s|under sections?)\s+([0-9]{1,3}[A-Z]?(?:\s*(?:,|and|/|r/w|read with)\s*[0-9]{1,3}[A-Z]?){0,8})\s*(?:IPC|I\.P\.C\.|of IPC)?\b"
	for match in re.finditer(section_first_pattern, text, flags=re.IGNORECASE):
		for sec in re.findall(r"\b\d{1,3}[A-Z]?\b", match.group(1)):
			sec = sec.upper()
			if sec not in found:
				found.append(sec)

	ipc_first_pattern = r"\b(?:IPC|I\.P\.C\.)\s*([0-9]{1,3}[A-Z]?(?:\s*(?:,|and|/)\s*[0-9]{1,3}[A-Z]?){0,8})\b"
	for match in re.finditer(ipc_first_pattern, text, flags=re.IGNORECASE):
		for sec in re.findall(r"\b\d{1,3}[A-Z]?\b", match.group(1)):
			sec = sec.upper()
			if sec not in found:
				found.append(sec)
	return found


def _clean_text(text: object, max_len: int = 260) -> str:
	cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
	return cleaned[:max_len].rstrip()


def _extract_keywords(text: str, max_keywords: int = 12) -> List[str]:
	words = re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
	stopwords = {
		"with", "from", "that", "this", "there", "their", "have", "were", "been", "when", "where", "which", "under", "into", "upon", "during", "after", "before", "about", "against", "victim", "accused", "person", "court", "section", "sections", "indian", "penal", "code", "state",
	}
	filtered = [word for word in words if word not in stopwords]
	if not filtered:
		return []
	counts = Counter(filtered)
	return [word for word, _ in counts.most_common(max_keywords)]


def _build_structured_input(user_case: str) -> Dict[str, object]:
	query = (user_case or "").strip()
	if not query:
		return {
			"facts": [],
			"arguments": [],
			"reasoning": [],
			"all_sentences": [],
			"keywords": [],
			"retrieval_query": "",
		}

	if _STAGE_PREPROCESSING_AVAILABLE:
		stage0_sentences = [s.strip() for s in process_text_stage0(query) if s and s.strip()]
		labeled = process_sentences_stage1(stage0_sentences)
	else:
		stage0_sentences = [query]
		labeled = [{"text": query, "label": "FACT"}]

	facts = [_clean_text(x.get("text", ""), max_len=220) for x in labeled if str(x.get("label", "")).upper() == "FACT"]
	arguments = [_clean_text(x.get("text", ""), max_len=220) for x in labeled if str(x.get("label", "")).upper() == "ARGUMENT"]
	reasoning = [_clean_text(x.get("text", ""), max_len=220) for x in labeled if str(x.get("label", "")).upper() == "REASONING"]

	if not facts and stage0_sentences:
		facts = [_clean_text(stage0_sentences[0], max_len=220)]

	compact_facts = facts[:4]
	compact_arguments = arguments[:3]
	compact_reasoning = reasoning[:3]
	keywords = _extract_keywords(" ".join(stage0_sentences) or query)
	ipc_candidates = _extract_ipc_sections(" ".join(stage0_sentences) or query)

	retrieval_parts = []
	if compact_facts:
		retrieval_parts.append("Facts: " + " ".join(compact_facts))
	if compact_arguments:
		retrieval_parts.append("Arguments: " + " ".join(compact_arguments))
	if compact_reasoning:
		retrieval_parts.append("Reasoning: " + " ".join(compact_reasoning))
	if keywords:
		retrieval_parts.append("Keywords: " + ", ".join(keywords[:8]))
	if ipc_candidates:
		retrieval_parts.append("IPC hints: " + ", ".join(ipc_candidates[:8]))

	return {
		"facts": compact_facts,
		"arguments": compact_arguments,
		"reasoning": compact_reasoning,
		"all_sentences": stage0_sentences,
		"keywords": keywords,
		"ipc_candidates": ipc_candidates,
		"retrieval_query": " | ".join(retrieval_parts) if retrieval_parts else query,
	}


def _clean_precedents(precedents: List[Dict[str, object]], limit: int = 4) -> List[Dict[str, str]]:
	cleaned: List[Dict[str, str]] = []
	seen = set()
	for item in precedents:
		case_name = _clean_text(item.get("case_name", "Unknown Case"), max_len=120)
		if not case_name or case_name.lower() in seen:
			continue
		seen.add(case_name.lower())

		judgment = _clean_text(item.get("judgment", "unknown"), max_len=40).lower()
		ipc_sections = [str(sec).upper().strip() for sec in (item.get("ipc_sections", []) or []) if str(sec).strip()]
		charges = ", ".join(f"IPC {sec}" for sec in ipc_sections[:4]) if ipc_sections else "charges not clearly specified"
		cleaned.append(
			{
				"case_name": case_name,
				"judgment": judgment,
				"charges": charges,
			}
		)
		if len(cleaned) >= limit:
			break
	return cleaned


def _clean_statutes(statutes: List[Dict[str, object]], limit: int = 5) -> List[Dict[str, str]]:
	cleaned: List[Dict[str, str]] = []
	seen = set()
	for item in statutes:
		section = str(item.get("section", "")).strip().upper()
		if not section or section in seen:
			continue
		seen.add(section)
		title = _clean_text(item.get("title", ""), max_len=130)
		description = _clean_text(item.get("description", ""), max_len=220)
		cleaned.append(
			{
				"section": section,
				"title": title,
				"description": description,
			}
		)
		if len(cleaned) >= limit:
			break
	return cleaned


def _format_precedents(precedents: List[Dict[str, str]]) -> str:
	if not precedents:
		return "No reliable precedents available from retrieval."
	items = [f"{item['case_name']} ({item['judgment']}; charges: {item.get('charges', 'unknown')})" for item in precedents]
	return "; ".join(items)


def _format_statutes(statutes: List[Dict[str, str]]) -> str:
	if not statutes:
		return "No reliable IPC sections available from retrieval."
	items = [f"IPC {item['section']} - {item['title']} ({item.get('description', '')})" for item in statutes]
	return "; ".join(items)


def _token_to_int(token: str) -> int | None:
	cleaned = (token or "").strip().lower().replace("-", " ")
	if not cleaned:
		return None
	if cleaned.isdigit():
		return int(cleaned)
	if cleaned in _NUMBER_WORDS:
		return _NUMBER_WORDS[cleaned]
	if " " in cleaned:
		parts = [p for p in cleaned.split() if p]
		if all(part in _NUMBER_WORDS for part in parts):
			return sum(_NUMBER_WORDS[part] for part in parts)
	return None


def _extract_year_values(text: str) -> List[int]:
	if not text:
		return []
	years: List[int] = []
	for token in re.findall(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b\s+years?", text.lower()):
		value = _token_to_int(token)
		if value is not None:
			years.append(value)
	return years


def _punishment_profile(statute: Dict[str, str]) -> Dict[str, object]:
	section = str(statute.get("section", "")).strip().upper()
	title = str(statute.get("title", "") or "")
	description = str(statute.get("description", "") or "")
	joined = f"{title} {description}".lower()

	years = _extract_year_values(joined)
	min_years = None
	min_match = re.search(
		r"not less than\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+years?",
		joined,
	)
	if min_match:
		min_years = _token_to_int(min_match.group(1))

	return {
		"section": section,
		"title": title,
		"description": description,
		"allows_death": bool(re.search(r"\bdeath\b", joined)),
		"allows_life": "imprisonment for life" in joined or "life imprisonment" in joined,
		"max_years": max(years) if years else None,
		"min_years": min_years,
		"allows_fine": "fine" in joined,
	}


def _punishment_severity(profile: Dict[str, object]) -> int:
	if profile.get("allows_death"):
		return 100
	if profile.get("allows_life"):
		return 80
	max_years = profile.get("max_years")
	if isinstance(max_years, int):
		return max_years
	return 5


def _rank_profiles(statutes: List[Dict[str, str]]) -> List[Dict[str, object]]:
	profiles = [_punishment_profile(item) for item in statutes]
	return sorted(profiles, key=_punishment_severity, reverse=True)


def _format_inr(amount: int) -> str:
	return f"₹{amount:,}"


def _recommended_compensation_amount(query_text: str, severity: int, stance: str) -> int:
	q = (query_text or "").lower()
	if any(term in q for term in ["death", "died", "killed", "fatal", "murder"]):
		base = 1200000
	elif any(term in q for term in ["grievous", "maim", "fracture", "permanent"]):
		base = 400000
	elif any(term in q for term in ["hurt", "injury", "assault", "attack"]):
		base = 200000
	else:
		base = 150000

	if severity >= 100:
		base = max(base, 1500000)
	elif severity >= 80:
		base = max(base, 900000)
	elif severity >= 10:
		base = max(base, 300000)

	if stance == "defence":
		base = int(base * 0.5)
	elif stance == "judge":
		base = int(base * 0.8)

	return int(round(base / 50000.0) * 50000)


def _specific_sentence_request(statutes: List[Dict[str, str]], query_text: str, stance: str, mitigate: bool = False) -> Dict[str, str]:
	profiles = _rank_profiles(statutes)
	if not profiles:
		fallback_amount = _recommended_compensation_amount(query_text, 5, stance)
		return {
			"section": "IPC (retrieved section to be determined)",
			"sentence": "rigorous imprisonment for 3 years",
			"compensation": _format_inr(fallback_amount),
		}

	primary = profiles[0]
	primary_section = f"IPC {primary['section']}"
	primary_severity = _punishment_severity(primary)
	lower_options = [p for p in profiles[1:] if _punishment_severity(p) < primary_severity]
	lesser = lower_options[0] if lower_options else None

	if stance == "prosecution":
		if primary.get("allows_death"):
			sentence = "death penalty, in the alternative imprisonment for life"
		elif primary.get("allows_life"):
			max_years = primary.get("max_years")
			sentence = f"imprisonment for life or, in the alternative, rigorous imprisonment for {max_years} years" if isinstance(max_years, int) else "imprisonment for life"
		else:
			max_years = primary.get("max_years")
			years = max_years if isinstance(max_years, int) else 3
			sentence = f"rigorous imprisonment for {years} years"
		amount = _recommended_compensation_amount(query_text, primary_severity, stance)
		return {"section": primary_section, "sentence": sentence, "compensation": _format_inr(amount)}

	if stance == "defence":
		if primary.get("allows_death"):
			if lesser and isinstance(lesser.get("max_years"), int):
				years = max(5, min(int(lesser["max_years"]), 10))
				sentence = f"conviction, if any, be limited to IPC {lesser['section']} with rigorous imprisonment for {years} years and not death"
				section_text = f"IPC {lesser['section']}"
			else:
				sentence = "if convicted, impose imprisonment for life and not death penalty"
				section_text = primary_section
		elif primary.get("allows_life") and lesser and isinstance(lesser.get("max_years"), int):
			years = max(3, min(int(lesser["max_years"]), 10))
			sentence = f"record only a lesser offence under IPC {lesser['section']} with rigorous imprisonment for {years} years"
			section_text = f"IPC {lesser['section']}"
		else:
			max_years = primary.get("max_years")
			if isinstance(max_years, int):
				years = max(1, int(round(max_years * 0.5)))
				sentence = f"sentence be reduced to imprisonment for {years} years"
			else:
				sentence = "sentence be restricted to the period already undergone"
			section_text = primary_section
		amount = _recommended_compensation_amount(query_text, primary_severity, stance)
		return {"section": section_text, "sentence": sentence, "compensation": _format_inr(amount)}

	if primary.get("allows_death") and not mitigate:
		sentence = "awards death penalty, with imprisonment for life as the statutory alternative"
		section_text = primary_section
		severity = primary_severity
	elif primary.get("allows_death") and mitigate:
		sentence = "awards imprisonment for life instead of death penalty"
		section_text = primary_section
		severity = 80
	elif primary.get("allows_life") and not mitigate:
		sentence = "awards imprisonment for life"
		section_text = primary_section
		severity = primary_severity
	elif primary.get("allows_life") and mitigate and lesser and isinstance(lesser.get("max_years"), int):
		years = max(5, min(int(lesser["max_years"]), 10))
		sentence = f"convicts under IPC {lesser['section']} and awards rigorous imprisonment for {years} years"
		section_text = f"IPC {lesser['section']}"
		severity = _punishment_severity(lesser)
	else:
		max_years = primary.get("max_years")
		if isinstance(max_years, int):
			years = max(1, int(round(max_years * (0.7 if mitigate else 1.0))))
			sentence = f"awards rigorous imprisonment for {years} years"
		else:
			sentence = "awards rigorous imprisonment for 3 years"
		section_text = primary_section
		severity = primary_severity
	amount = _recommended_compensation_amount(query_text, severity, stance)
	return {"section": section_text, "sentence": sentence, "compensation": _format_inr(amount)}


def _has_specific_sentencing(text: str) -> bool:
	if not text:
		return False
	has_sentence = bool(
		re.search(
			r"\b(death penalty|imprisonment for life|life imprisonment|rigorous imprisonment for\s+\d+\s+years?|imprisonment for\s+\d+\s+years?)\b",
			text,
			flags=re.IGNORECASE,
		)
	)
	has_compensation = bool(re.search(r"\b(compensation|fine)\b", text, flags=re.IGNORECASE)) and bool(
		re.search(r"(₹\s*\d[\d,]*|\b(?:INR|Rs\.?|Rupees)\s*\d[\d,]*)", text, flags=re.IGNORECASE)
	)
	return has_sentence and has_compensation


def _defence_convincing_score(defence_output: str) -> int:
	text = (defence_output or "").lower()
	markers = [
		"benefit of doubt",
		"contradiction",
		"inconsisten",
		"no premeditation",
		"sudden fight",
		"lack of intent",
		"mitigat",
		"first-time offender",
		"reform",
		"remorse",
	]
	return sum(1 for marker in markers if marker in text)


def _extract_precedent_mentions(text: str, max_items: int = 6) -> List[str]:
	if not text:
		return []
	pattern = r"([A-Z][A-Za-z0-9@&.,'()\-\s]{3,140}\b(?:v\.?|vs\.?|versus)\b[A-Za-z0-9@&.,'()\-\s]{3,140})"
	found: List[str] = []
	for match in re.findall(pattern, text, flags=re.IGNORECASE):
		cleaned = re.sub(r"\s+", " ", match).strip(" .;,")
		if len(cleaned) < 10:
			continue
		if cleaned.lower() not in {x.lower() for x in found}:
			found.append(cleaned)
		if len(found) >= max_items:
			break
	return found


def _party_precedents(prosecutor_output: str, defence_output: str, limit: int = 6) -> List[Dict[str, str]]:
	names = _extract_precedent_mentions(prosecutor_output, max_items=limit) + _extract_precedent_mentions(defence_output, max_items=limit)
	unique: List[str] = []
	seen = set()
	for name in names:
		key = name.lower()
		if key in seen:
			continue
		seen.add(key)
		unique.append(name)
		if len(unique) >= limit:
			break
	return [{"case_name": n, "judgment": "as cited by parties", "charges": "as argued by counsel"} for n in unique]


def _party_statutes(prosecutor_output: str, defence_output: str, limit: int = 8) -> List[Dict[str, str]]:
	sections = []
	for sec in _extract_ipc_sections(prosecutor_output) + _extract_ipc_sections(defence_output):
		if sec not in sections:
			sections.append(sec)
		if len(sections) >= limit:
			break
	return [
		{
			"section": sec,
			"title": "Section relied upon by parties",
			"description": "Derived from prosecutor and defence submissions",
		}
		for sec in sections
	]


def _extract_requested_sentence(text: str) -> str | None:
	t = (text or "")
	t_low = t.lower()
	if "death penalty" in t_low and ("imprisonment for life" in t_low or "life imprisonment" in t_low):
		return "death penalty, in the alternative imprisonment for life"
	if "death penalty" in t_low:
		return "death penalty"
	if "imprisonment for life" in t_low or "life imprisonment" in t_low:
		return "imprisonment for life"
	match = re.search(r"(rigorous\s+imprisonment\s+for\s+\d+\s+years?|imprisonment\s+for\s+\d+\s+years?)", t, flags=re.IGNORECASE)
	if match:
		return re.sub(r"\s+", " ", match.group(1)).strip().lower()
	return None


def _extract_compensation_amount(text: str) -> int | None:
	t = text or ""
	amounts: List[int] = []
	for token in re.findall(r"₹\s*(\d[\d,]*)", t, flags=re.IGNORECASE):
		amounts.append(int(token.replace(",", "")))
	for token in re.findall(r"\b(?:INR|Rs\.?|Rupees)\s*(\d[\d,]*)", t, flags=re.IGNORECASE):
		amounts.append(int(token.replace(",", "")))
	if not amounts:
		return None
	return max(amounts)


def _judge_relief_from_party_outputs(
	user_case: str,
	prosecutor_output: str,
	defence_output: str,
	statutes: List[Dict[str, str]],
	mitigate: bool,
) -> Dict[str, str]:
	baseline = _specific_sentence_request(statutes, user_case or prosecutor_output, stance="judge", mitigate=mitigate)

	pro_sentence = _extract_requested_sentence(prosecutor_output)
	def_sentence = _extract_requested_sentence(defence_output)
	if mitigate and def_sentence:
		final_sentence = def_sentence
	elif (not mitigate) and pro_sentence:
		final_sentence = pro_sentence
	else:
		final_sentence = baseline["sentence"]

	pro_amount = _extract_compensation_amount(prosecutor_output)
	def_amount = _extract_compensation_amount(defence_output)
	if pro_amount and def_amount:
		chosen = int((pro_amount + (2 * def_amount)) / 3) if mitigate else int(((2 * pro_amount) + def_amount) / 3)
	elif pro_amount:
		chosen = int(pro_amount * (0.75 if mitigate else 0.9))
	elif def_amount:
		chosen = def_amount
	else:
		chosen = _recommended_compensation_amount(user_case or prosecutor_output, 50, "judge")
	chosen = int(round(chosen / 50000.0) * 50000)

	section = baseline["section"]
	party_sections = _party_statutes(prosecutor_output, defence_output, limit=4)
	if party_sections:
		section = f"IPC {party_sections[0]['section']}"

	return {
		"section": section,
		"sentence": final_sentence,
		"compensation": _format_inr(chosen),
	}


def _is_defence_friendly_judgment(judgment: str) -> bool:
	text = (judgment or "").strip().lower()
	if not text:
		return False
	keywords = [
		"acquitted",
		"allowed",
		"partly allowed",
		"partially allowed",
		"set aside",
		"reduced",
		"modified",
		"bail",
	]
	return any(keyword in text for keyword in keywords)


def _defence_precedent_score(item: Dict[str, object], query_ipc: List[str]) -> float:
	judgment = str(item.get("judgment", "") or "").lower()
	ipc_sections = [str(sec).upper().strip() for sec in (item.get("ipc_sections", []) or []) if str(sec).strip()]
	score = float(item.get("score", 0.0))

	if _is_defence_friendly_judgment(judgment):
		score += 0.75
	if any(token in judgment for token in ["acquitted", "set aside"]):
		score += 0.35
	if any(token in judgment for token in ["reduced", "modified", "partly", "partially"]):
		score += 0.25
	if query_ipc and set(query_ipc).intersection(set(ipc_sections)):
		score += 0.25

	return score


def _clean_defence_precedents(precedents: List[Dict[str, object]], limit: int = 5) -> List[Dict[str, str]]:
	cleaned: List[Dict[str, str]] = []
	seen = set()
	for item in precedents:
		case_name = _clean_text(item.get("case_name", "Unknown Case"), max_len=120)
		if not case_name or case_name.lower() in seen:
			continue
		seen.add(case_name.lower())

		judgment = _clean_text(item.get("judgment", "unknown"), max_len=60).lower()
		facts = _clean_text(item.get("facts", ""), max_len=230)
		ipc_sections = [str(sec).upper().strip() for sec in (item.get("ipc_sections", []) or []) if str(sec).strip()]
		charges = ", ".join(f"IPC {sec}" for sec in ipc_sections[:4]) if ipc_sections else "charges not clearly specified"

		cleaned.append(
			{
				"case_name": case_name,
				"judgment": judgment,
				"facts": facts,
				"charges": charges,
			}
		)
		if len(cleaned) >= limit:
			break
	return cleaned


def _format_defence_precedents(precedents: List[Dict[str, str]]) -> str:
	if not precedents:
		return "No reliable defence-friendly precedents available from retrieval."
	items = [
		f"{item['case_name']} ({item['judgment']}; charges: {item.get('charges', 'unknown')}; defence context: {item.get('facts', 'facts unavailable')})"
		for item in precedents
	]
	return "; ".join(items)


def _prosecutor_indicates_strong_guilt(prosecutor_output: str) -> bool:
	text = (prosecutor_output or "").lower()
	if not text:
		return False
	strong_markers = [
		"proved beyond reasonable doubt",
		"clear intent",
		"premeditated",
		"brutal",
		"deliberate",
		"conviction under",
		"death penalty",
		"life imprisonment",
		"mens rea",
		"actus reus",
	]
	return sum(1 for marker in strong_markers if marker in text) >= 2


def _build_defence_prompt(
	user_case: str,
	prosecutor_output: str,
	structured_input: Dict[str, object],
	precedents: List[Dict[str, str]],
	statutes: List[Dict[str, str]],
	strong_guilt: bool,
) -> str:
	defence_relief = _specific_sentence_request(statutes, user_case or prosecutor_output, stance="defence", mitigate=True)
	mode_instruction = (
		"The prosecution narrative appears strong; focus on sentencing mitigation, lesser offence framing where legally supported, lack of premeditation, proportional punishment, and rehabilitation."
		if strong_guilt
		else "Challenge prosecution certainty with legally valid defence points: evidentiary gaps, contradictions, intent/knowledge limits, causation doubts, and proportional application of IPC sections."
	)

	system_instruction = "You are a senior criminal defence counsel in an Indian court. Defend the accused with legal precision, dignity, and human emotion. Write one continuous paragraph only, with no headings, bullets, numbering, or list formatting."

	return f"""
{system_instruction}

You are given the prosecutor's argument as your primary input and must respond as defence counsel.
Do not make weak or irrelevant claims such as saying IPC itself is invalid or wrong.
Use only legally valid defence strategy anchored in the provided IPC sections and retrieved precedents.
Cite prior cases naturally to support defence reasoning and requested relief.
If conviction seems likely, explicitly seek reduction in punishment, lesser sentence, and proportional relief instead of unrealistic acquittal claims.
Return one single paragraph, roughly 10 to 16 sentences, human-like and emotionally aware, in formal courtroom style.
Output plain paragraph text only.
The final prayer must be specific and not generic: identify exact IPC section(s), ask an exact sentencing outcome (for example imprisonment for life instead of death, or imprisonment for specific years), and state a concrete compensation position with amount in INR.
Do not ask for any sentence beyond what the provided IPC sections legally permit.

Defence strategy mode: {mode_instruction}

Original case facts (if available): {user_case or 'Not separately provided.'}

Prosecutor output (primary input to defence): {prosecutor_output}

Structured preprocessing from prosecutor output:
{_format_structured_input(structured_input)}

Defence-supporting precedents:
{_format_defence_precedents(precedents)}

Relevant IPC sections for defence framing:
{_format_statutes(statutes)}

Suggested legally grounded defence relief to anchor your final prayer:
Section focus: {defence_relief['section']}; sentence request: {defence_relief['sentence']}; compensation position: not exceeding {defence_relief['compensation']}.
""".strip()


def _defence_api_failure_fallback(
	prosecutor_output: str,
	precedents: List[Dict[str, str]],
	statutes: List[Dict[str, str]],
	strong_guilt: bool,
) -> str:
	precedent_text = ", ".join(f"{item['case_name']} ({item['judgment']})" for item in precedents[:3])
	statute_text = ", ".join(f"IPC {item['section']} ({item['title']})" for item in statutes[:3])
	if not precedent_text:
		precedent_text = "supportive defence precedents"
	if not statute_text:
		statute_text = "relevant IPC provisions"
	defence_relief = _specific_sentence_request(statutes, prosecutor_output, stance="defence", mitigate=True)

	if strong_guilt:
		paragraph = (
			f"The defence respectfully submits that while the prosecution has narrated the occurrence with force, sentencing must still remain humane, individualized, and proportionate, and this Court may therefore evaluate culpability through the precise contours of {statute_text} rather than through generalized moral outrage alone. "
			f"Comparable judicial guidance in {precedent_text} shows that even in serious prosecutions, courts have moderated punishment where circumstances disclosed absence of premeditated design, scope for reform, and mitigating context surrounding the incident. "
			f"In that spirit, the accused seeks that conviction, if any, be confined to {defence_relief['section']} and that the sentence be restricted to {defence_relief['sentence']}, with compensation, if awarded, capped at {defence_relief['compensation']}, so that justice protects society while still preserving constitutional fairness and the possibility of reformation."
		)
	else:
		paragraph = (
			f"The defence respectfully submits that the prosecution's conclusion of guilt is not the only possible legal view on the present record, and the Court must test the allegations with strict scrutiny under {statute_text} before fastening criminal liability of the gravest kind. "
			f"The authorities in {precedent_text} demonstrate that where intent, causation, or evidentiary consistency remains uncertain, courts have extended benefit of doubt or granted substantial relief to prevent irreversible injustice. "
			f"With deep regard for the victim's suffering and equal concern for fair trial rights of the accused, the defence prays that the Court either grant benefit of doubt on the graver charge or, in the alternative, record conviction only under {defence_relief['section']} with {defence_relief['sentence']}, and if compensation is considered, limit it to {defence_relief['compensation']} in the facts proved."
		)

	return _normalize_single_paragraph(paragraph)


def _format_structured_input(structured: Dict[str, object]) -> str:
	facts = structured.get("facts", []) or []
	arguments = structured.get("arguments", []) or []
	reasoning = structured.get("reasoning", []) or []
	keywords = structured.get("keywords", []) or []

	parts = [
		"Facts: " + (" | ".join(facts) if facts else "not clearly segmented"),
		"Arguments: " + (" | ".join(arguments) if arguments else "not clearly segmented"),
		"Reasoning cues: " + (" | ".join(reasoning) if reasoning else "not clearly segmented"),
		"Keywords: " + (", ".join(keywords[:10]) if keywords else "none"),
	]
	return "\n".join(parts)


def _api_failure_fallback(
	user_case: str,
	precedents: List[Dict[str, str]],
	statutes: List[Dict[str, str]],
) -> str:
	precedent_text = ", ".join(f"{item['case_name']} ({item['judgment']}; {item.get('charges', 'charges not clearly specified')})" for item in precedents[:3])
	statute_text = ", ".join(f"IPC {item['section']} ({item['title']})" for item in statutes[:3])
	if not precedent_text:
		precedent_text = "comparable prosecution precedents"
	if not statute_text:
		statute_text = "relevant IPC provisions"
	prosecution_relief = _specific_sentence_request(statutes, user_case, stance="prosecution")

	paragraph = (
		f"It is respectfully submitted that the victim in the present case suffered grave harm in the incident narrated as {user_case.strip()}, and the emotional and physical devastation caused to the victim and family calls for firm judicial response rooted in both compassion and accountability. "
		f"The gravamen of the offence squarely falls within {statute_text}, and these are the principal penal provisions that precisely characterize the unlawful conduct, the attendant intent or knowledge, and the resulting consequence in law. "
		f"Comparable judicial guidance emerges from {precedent_text}, where courts treated similar factual matrices with seriousness and proceeded on corresponding criminal charges against the accused based on evidence of conduct and culpable mental state. "
		f"In these circumstances, the prosecution prays for conviction under {prosecution_relief['section']} and seeks {prosecution_relief['sentence']} together with appropriate fine and victim compensation of {prosecution_relief['compensation']}, so that sentencing remains lawful, proportionate, and responsive to the suffering proved before this Hon'ble Court."
	)
	return _normalize_single_paragraph(paragraph)


def _statute_query_bonus(query: str, statute: Dict[str, object]) -> float:
	query_low = query.lower()
	section = str(statute.get("section", "")).strip()
	title_desc = f"{statute.get('title', '')} {statute.get('description', '')}".lower()

	bonus = 0.0
	if any(x in query_low for x in ["stab", "knife", "death", "died", "murder"]):
		if section in {"299", "300", "302", "304", "304A"}:
			bonus += 0.5
	if any(x in query_low for x in ["fraud", "cheat", "forgery"]):
		if section in {"415", "420", "467", "468", "471", "406", "409"}:
			bonus += 0.45
	if any(x in query_low for x in ["hurt", "injury", "assault", "attack"]):
		if section in {"323", "324", "325", "326", "307"}:
			bonus += 0.4

	term_overlap = sum(1 for term in set(re.findall(r"[a-zA-Z]{3,}", query_low)) if term in title_desc)
	bonus += term_overlap * 0.01

	if section.isdigit() and int(section) <= 20:
		bonus -= 0.2

	return bonus


def _build_prompt(
	user_case: str,
	structured_input: Dict[str, object],
	precedents: List[Dict[str, str]],
	statutes: List[Dict[str, str]],
) -> str:
	prosecution_relief = _specific_sentence_request(statutes, user_case, stance="prosecution")
	system_instruction = "You are a senior criminal prosecutor in an Indian court. You are presenting final oral arguments for the victim. Speak with controlled emotion, empathy for the victim, and strict legal precision. Write one continuous paragraph only, with no headings, bullets, numbering, or list formatting."

	return f"""
{system_instruction}

Use only the inputs below and do not invent facts, precedents, or IPC sections.
Return one single paragraph, roughly 10 to 16 sentences, in formal courtroom style.
Follow this exact narrative order inside the paragraph (without explicit numbering):
first explain what happened to the victim; then identify the main IPC sections with high precision;
then discuss similar prior cases and the charges against the accused in those cases;
finally request charges, punishment, and compensation (if any) against the current accused.
Punishment and compensation requests must remain within legal limits implied by the provided IPC sections and their descriptions; do not ask for excessive or legally impossible relief.
Do not request both life imprisonment and death penalty unless the provided IPC sections explicitly support such alternatives.
If compensation is requested, keep it proportional and legally permissible based on the provided sections and case gravity.
Mention mens rea and actus reus naturally while preserving human-like flow.
The final prayer must be specific and concrete, not generic: cite exact IPC section(s), exact sentence term (death, life, or years), and exact compensation amount in INR.
Output plain paragraph text only.

Case facts: {user_case}
Structured preprocessing output:
{_format_structured_input(structured_input)}
Precedents: {_format_precedents(precedents)}
IPC sections: {_format_statutes(statutes)}

Suggested legally grounded prosecution prayer to anchor your final sentence:
Section focus: {prosecution_relief['section']}; sentence request: {prosecution_relief['sentence']}; compensation request: {prosecution_relief['compensation']}.
""".strip()


def _query_has_homicide_signals(query: str) -> bool:
	q = (query or "").lower()
	keywords = ["murder", "homicide", "killed", "death", "died", "stab", "knife", "fatal"]
	return any(word in q for word in keywords)


def _prioritize_statutes(query: str, statutes: List[Dict[str, object]]) -> List[Dict[str, object]]:
	if not statutes:
		return []

	if _query_has_homicide_signals(query):
		priority = {"299": 0.25, "300": 0.4, "302": 0.6, "304": 0.35, "304A": 0.25, "307": 0.15}
		ranked = sorted(
			statutes,
			key=lambda row: float(row.get("score", 0.0)) + priority.get(str(row.get("section", "")).strip().upper(), 0.0),
			reverse=True,
		)
		return ranked

	return statutes


def _normalize_single_paragraph(text: str) -> str:
	cleaned = (text or "").replace("\r", "\n")
	cleaned = re.sub(r"\n{2,}", " ", cleaned)
	cleaned = re.sub(r"\n\s*[-*•]\s*", " ", cleaned)
	cleaned = re.sub(r"\n\s*\d+\.\s+", " ", cleaned)
	cleaned = re.sub(r"\b\d+\.\s*(Facts of the Case|Relevant Precedents|Applicable IPC Sections and Charges|Legal Argument \(Prosecution Side\)|Prayer for Relief / Punishment)\b", " ", cleaned, flags=re.IGNORECASE)
	cleaned = re.sub(r"\s+", " ", cleaned).strip()
	if not cleaned:
		return cleaned

	sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
	if len(sentences) > 18:
		sentences = sentences[:18]

	one_paragraph = " ".join(sentences)
	return one_paragraph.strip()


def _sentence_count(text: str) -> int:
	return len([s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()])


def _looks_truncated(text: str) -> bool:
	cleaned = (text or "").strip()
	if not cleaned:
		return True
	if len(cleaned) < 220:
		return True
	if _sentence_count(cleaned) < 4:
		return True
	return cleaned[-1] not in {".", "!", "?"}


def _looks_like_good_oral_argument(text: str) -> bool:
	if not text or len(text.strip()) < 400:
		return False

	low = text.lower()
	if "draft to improve" in low or "clean precedents" in low or "clean ipc sections" in low:
		return False

	if any(h in text for h in [
		"1. Facts of the Case",
		"2. Relevant Precedents",
		"3. Applicable IPC Sections and Charges",
		"4. Legal Argument (Prosecution Side)",
		"5. Prayer for Relief / Punishment",
	]):
		return False

	if "\n- " in text or "\n* " in text:
		return False

	if not re.search(r"\bIPC\s*\d{1,3}[A-Z]?\b", text, flags=re.IGNORECASE):
		return False

	if not re.search(r"\b(imprisonment|life imprisonment|death penalty|compensation|fine|convict)\b", text, flags=re.IGNORECASE):
		return False

	if _sentence_count(text) < 10:
		return False

	if len(re.findall(r"\bIPC\s*299\b", text, flags=re.IGNORECASE)) >= 5:
		return False

	return True


def _argument_quality_score(text: str) -> int:
	if not text:
		return 0

	score = 0
	if len(text.strip()) >= 300:
		score += 2
	if len(text.strip()) >= 500:
		score += 2
	if re.search(r"\bIPC\s*\d{1,3}[A-Z]?\b", text, flags=re.IGNORECASE):
		score += 2
	if re.search(r"\b(convicted|acquitted|sentenced)\b", text, flags=re.IGNORECASE):
		score += 2
	if re.search(r"\b(actus reus|mens rea|intent|knowledge)\b", text, flags=re.IGNORECASE):
		score += 2
	if re.search(r"\b(imprisonment|life imprisonment|death penalty|compensation|fine|convict)\b", text, flags=re.IGNORECASE):
		score += 2
	if "\n- " not in text and "\n* " not in text:
		score += 1
	if _sentence_count(text) >= 10:
		score += 1
	return score


def _rewrite_as_oral_argument(
	query: str,
	structured_input: Dict[str, object],
	original_answer: str,
	precedents: List[Dict[str, str]],
	statutes: List[Dict[str, str]],
) -> str:
	relief = _specific_sentence_request(statutes, query, stance="prosecution")
	rewrite_prompt = f"""
You are a senior criminal prosecutor in an Indian court delivering final oral submissions.
Generate a fresh final argument in a single continuous paragraph, 10 to 16 sentences, persuasive and human.
No headings, no bullets, no list format, and no repetition.
Do not output instruction text; output only the final oral argument paragraph.
Follow this order naturally: what happened to victim -> precise IPC sections -> similar precedents with charges -> requested conviction, punishment, and compensation.
Keep punishment/compensation strictly within legal scope of the provided IPC sections.
The final prayer must be explicit and specific, including exact sentence term and compensation amount in INR.

User case:
{query}

Structured preprocessing:
{_format_structured_input(structured_input)}

Clean precedents:
{_format_precedents(precedents)}

Clean IPC sections:
{_format_statutes(statutes)}

Use this legally grounded target in your final prayer:
{relief['section']} | {relief['sentence']} | compensation {relief['compensation']}
""".strip()

	rewritten = generate_response(rewrite_prompt).strip()
	if rewritten.startswith("[llm] Generation failed"):
		return rewritten
	return _normalize_single_paragraph(rewritten)


def prosecution(user_case: str) -> str:
	if not user_case or not user_case.strip():
		raise ValueError("user_case must be a non-empty string")

	initialize_precedent_retriever()
	initialize_statute_retriever()

	query = user_case.strip()
	structured_input = _build_structured_input(query)
	retrieval_query = str(structured_input.get("retrieval_query", "") or query).strip()
	query_ipc = sorted(set(_extract_ipc_sections(query) + [str(sec).strip().upper() for sec in (structured_input.get("ipc_candidates", []) or []) if str(sec).strip()]))

	precedents = retrieve_precedents(
		query=retrieval_query,
		top_k=8,
		ipc_sections=query_ipc if query_ipc else None,
		require_ipc_match=False,
	)

	ipc_hint = sorted({str(sec).strip() for row in precedents for sec in (row.get("ipc_sections", []) or []) if str(sec).strip()})
	section_hint = sorted(set(ipc_hint + query_ipc))
	statutes_from_query_ipc = retrieve_statutes(query=retrieval_query, top_k=8, section_filter=query_ipc) if query_ipc else []
	statutes_semantic = retrieve_statutes(query=retrieval_query, top_k=8)
	statutes_hint = retrieve_statutes(query=retrieval_query, top_k=8, section_filter=section_hint) if section_hint else []

	statutes_by_section: Dict[str, Dict[str, object]] = {}
	for row in statutes_from_query_ipc + statutes_semantic + statutes_hint:
		section = str(row.get("section", "")).strip()
		if not section:
			continue
		old = statutes_by_section.get(section)
		if old is None or float(row.get("score", 0.0)) > float(old.get("score", 0.0)):
			statutes_by_section[section] = row
	statutes_ranked = sorted(
		statutes_by_section.values(),
		key=lambda x: float(x.get("score", 0.0)) + _statute_query_bonus(query, x) + (0.7 if str(x.get("section", "")).strip().upper() in set(query_ipc) else 0.0),
		reverse=True,
	)[:8]
	statutes_ranked = _prioritize_statutes(query, statutes_ranked)

	required_min = 2
	if len(precedents) < required_min:
		precedents = (precedents + retrieve_precedents(query=retrieval_query, top_k=required_min + 2))[: required_min + 2]
	if len(statutes_ranked) < required_min:
		fallback_statutes = retrieve_statutes(query=retrieval_query, top_k=required_min + 2)
		fallback_by_sec = {str(row.get("section", "")).strip(): row for row in fallback_statutes if str(row.get("section", "")).strip()}
		for section, row in fallback_by_sec.items():
			if section not in statutes_by_section:
				statutes_ranked.append(row)
		statutes_ranked = statutes_ranked[: required_min + 2]

	clean_precedents = _clean_precedents(precedents, limit=4)
	clean_statutes = _clean_statutes(statutes_ranked, limit=4)

	prompt = _build_prompt(query, structured_input, clean_precedents, clean_statutes)
	answer = generate_response(prompt).strip()

	if answer.startswith("[llm] Generation failed"):
		return _api_failure_fallback(query, clean_precedents, clean_statutes)

	answer = _normalize_single_paragraph(answer)
	if _looks_truncated(answer):
		return _api_failure_fallback(query, clean_precedents, clean_statutes)
	if not _has_specific_sentencing(answer):
		return _api_failure_fallback(query, clean_precedents, clean_statutes)

	if not _looks_like_good_oral_argument(answer):
		best = answer
		best_score = _argument_quality_score(best)
		for _ in range(MAX_REWRITE_ATTEMPTS):
			rewritten = _rewrite_as_oral_argument(query, structured_input, best, clean_precedents, clean_statutes)
			if rewritten.startswith("[llm] Generation failed"):
				return _api_failure_fallback(query, clean_precedents, clean_statutes)
			if _looks_truncated(rewritten):
				continue
			score = _argument_quality_score(rewritten)
			if score > best_score or (score == best_score and len(rewritten) > len(best)):
				best = rewritten
				best_score = score
			if _looks_like_good_oral_argument(rewritten):
				best = rewritten
				break
		answer = best if not _looks_truncated(best) else _api_failure_fallback(query, clean_precedents, clean_statutes)
	return answer


def prosecution_arguments(user_case: str) -> str:
	return prosecution(user_case)


def defence_arguments(prosecutor_output: str, user_case: str = "") -> str:
	if not prosecutor_output or not prosecutor_output.strip():
		raise ValueError("prosecutor_output must be a non-empty string")

	initialize_precedent_retriever()
	initialize_statute_retriever()

	prosecutor_text = prosecutor_output.strip()
	query_for_defence = (
		f"Defence response to prosecutor: {prosecutor_text} "
		"Focus on evidentiary doubts, intent limits, mitigation, lesser offence framing, sentence reduction, and proportional punishment under IPC."
	)

	structured_input = _build_structured_input(prosecutor_text)
	query_ipc = sorted(set(_extract_ipc_sections(prosecutor_text) + [str(sec).strip().upper() for sec in (structured_input.get("ipc_candidates", []) or []) if str(sec).strip()]))

	raw_precedents = retrieve_precedents(
		query=query_for_defence,
		top_k=14,
		ipc_sections=query_ipc if query_ipc else None,
		require_ipc_match=False,
	)
	ranked_precedents = sorted(raw_precedents, key=lambda row: _defence_precedent_score(row, query_ipc), reverse=True)
	clean_precedents = _clean_defence_precedents(ranked_precedents, limit=5)

	statutes_from_query_ipc = retrieve_statutes(query=query_for_defence, top_k=8, section_filter=query_ipc) if query_ipc else []
	statutes_semantic = retrieve_statutes(query=query_for_defence, top_k=8)
	statutes_pool: List[Dict[str, object]] = statutes_from_query_ipc + statutes_semantic
	statutes_by_section: Dict[str, Dict[str, object]] = {}
	for row in statutes_pool:
		section = str(row.get("section", "")).strip()
		if not section:
			continue
		old = statutes_by_section.get(section)
		if old is None or float(row.get("score", 0.0)) > float(old.get("score", 0.0)):
			statutes_by_section[section] = row
	statutes_ranked = sorted(
		statutes_by_section.values(),
		key=lambda x: float(x.get("score", 0.0)) + (0.7 if str(x.get("section", "")).strip().upper() in set(query_ipc) else 0.0),
		reverse=True,
	)[:8]
	clean_statutes = _clean_statutes(statutes_ranked, limit=5)

	strong_guilt = _prosecutor_indicates_strong_guilt(prosecutor_text)
	prompt = _build_defence_prompt(user_case, prosecutor_text, structured_input, clean_precedents, clean_statutes, strong_guilt)
	answer = generate_response(prompt).strip()

	if answer.startswith("[llm] Generation failed"):
		return _defence_api_failure_fallback(prosecutor_text, clean_precedents, clean_statutes, strong_guilt)

	answer = _normalize_single_paragraph(answer)
	if _looks_truncated(answer):
		return _defence_api_failure_fallback(prosecutor_text, clean_precedents, clean_statutes, strong_guilt)
	if not _has_specific_sentencing(answer):
		return _defence_api_failure_fallback(prosecutor_text, clean_precedents, clean_statutes, strong_guilt)

	return answer


def _build_judge_prompt(
	user_case: str,
	prosecutor_output: str,
	defence_output: str,
	precedents: List[Dict[str, str]],
	statutes: List[Dict[str, str]],
	mitigate: bool,
) -> str:
	judicial_relief = _judge_relief_from_party_outputs(user_case, prosecutor_output, defence_output, statutes, mitigate)
	mode = "Defence mitigation appears materially persuasive on sentencing; consider reducing severity within statutory limits." if mitigate else "Prosecution case appears stronger on culpability; apply sentence proportionate to proved offence within statutory limits."
	return f"""
You are an impartial criminal court judge in India writing final reasons and order.
You must remain unbiased: evaluate prosecution and defence submissions on legal merit only, and do not favour either side by tone.
Write one continuous paragraph only, 10 to 16 sentences, no headings and no bullets.
Use only the provided facts, precedents, and IPC sections already cited by prosecutor and defence; do not invent law and do not add fresh retrieval.
Your order must be specific and concrete: clearly state conviction/acquittal outcome, exact IPC section(s) finally applied, exact sentence term (death/life or imprisonment for X years), and compensation amount in INR for victim or legal heirs if warranted.
If defence mitigation is convincing, reduce sentencing severity or move to a legally sustainable lesser offence where justified.
Do not impose any punishment that is unsupported by the provided IPC sections.

Judicial evaluation mode: {mode}

Case facts: {user_case or 'Not separately provided'}
Prosecutor submissions: {prosecutor_output}
Defence submissions: {defence_output}
Comparable precedents: {_format_precedents(precedents)}
Applicable IPC sections: {_format_statutes(statutes)}

Legally grounded decision anchor:
Section focus: {judicial_relief['section']}; sentence baseline: {judicial_relief['sentence']}; compensation baseline: {judicial_relief['compensation']}.
""".strip()


def _judge_api_failure_fallback(
	user_case: str,
	prosecutor_output: str,
	defence_output: str,
	precedents: List[Dict[str, str]],
	statutes: List[Dict[str, str]],
) -> str:
	mitigate = _defence_convincing_score(defence_output) >= 2
	judicial_relief = _judge_relief_from_party_outputs(user_case, prosecutor_output, defence_output, statutes, mitigate)
	precedent_text = ", ".join(f"{item['case_name']} ({item['judgment']})" for item in precedents[:3])
	if not precedent_text:
		precedent_text = "precedents cited by both parties"
	paragraph = (
		f"Having considered the prosecution submissions and the defence response in light of the case facts, this Court records that culpability is to be determined strictly under the proved ingredients of {judicial_relief['section']} and not on conjecture. "
		f"The precedents cited by counsel, including {precedent_text}, support a calibrated approach where conviction follows proof, and sentencing remains proportionate to mens rea, actus reus, and resulting harm. "
		f"Accordingly, this Court orders that the accused stands dealt with under {judicial_relief['section']} and {judicial_relief['sentence']}. "
		f"Further, to address victim harm, compensation of {judicial_relief['compensation']} is directed in favour of the victim or legal heirs, subject to lawful mode of disbursal, while ensuring that the sentence and compensation remain within statutory bounds."
	)
	return _normalize_single_paragraph(paragraph)


def judge_arguments(prosecutor_output: str, defence_output: str, user_case: str = "") -> str:
	if not prosecutor_output or not prosecutor_output.strip():
		raise ValueError("prosecutor_output must be a non-empty string")
	if not defence_output or not defence_output.strip():
		raise ValueError("defence_output must be a non-empty string")

	clean_precedents = _party_precedents(prosecutor_output, defence_output, limit=6)
	clean_statutes = _party_statutes(prosecutor_output, defence_output, limit=8)

	mitigate = _defence_convincing_score(defence_output) >= 2
	prompt = _build_judge_prompt(user_case, prosecutor_output, defence_output, clean_precedents, clean_statutes, mitigate)
	answer = generate_response(prompt).strip()

	if answer.startswith("[llm] Generation failed"):
		return _judge_api_failure_fallback(user_case, prosecutor_output, defence_output, clean_precedents, clean_statutes)

	answer = _normalize_single_paragraph(answer)
	if _looks_truncated(answer):
		return _judge_api_failure_fallback(user_case, prosecutor_output, defence_output, clean_precedents, clean_statutes)
	if not _has_specific_sentencing(answer):
		return _judge_api_failure_fallback(user_case, prosecutor_output, defence_output, clean_precedents, clean_statutes)

	return answer


def run_case(user_case: str) -> Dict[str, str]:
	prosecutor_output = prosecution_arguments(user_case)
	defence_output = defence_arguments(prosecutor_output=prosecutor_output, user_case=user_case)
	judge_output = judge_arguments(prosecutor_output=prosecutor_output, defence_output=defence_output, user_case=user_case)
	return {
		"prosecution": prosecutor_output,
		"defence": defence_output,
		"judge": judge_output,
	}


if __name__ == "__main__":
	test_case = "A man stabbed another person with a knife causing death"
	print(prosecution(test_case))
