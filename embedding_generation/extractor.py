from __future__ import annotations

import re
from typing import Dict, List

ACTION_KEYWORDS = [
	"stabbed", "shot", "fired", "assaulted", "attacked", "killed", "murdered",
	"strangled", "poisoned", "burned", "cheated", "forged", "embezzled",
	"kidnapped", "abducted", "raped", "molested", "threatened", "robbed", "stole",
	"extorted", "forged", "misappropriated", "harassed", "cruelty",
]

WEAPON_KEYWORDS = [
	"knife", "dagger", "sword", "gun", "pistol", "revolver", "rifle", "firearm",
	"axe", "lathi", "rod", "stick", "stone", "acid", "poison", "bomb", "explosive",
]

ARGUMENT_MARKERS = [
	"learned counsel", "it is contended", "argued that", "it was argued", "it is submitted", "submission of",
]

REASONING_MARKERS = [
	"court held", "the court held", "observed", "therefore", "thus", "hence", "we find", "in our view", "it is clear that",
]

ISSUE_MARKERS = [
	"issue", "point for determination", "question for consideration", "whether", "for determination",
]

JUDGMENT_MARKERS = [
	"convicted", "acquitted", "sentenced", "appeal is dismissed", "appeal is allowed", "set aside", "imprisonment",
]


def sentence_tokenize(text: str) -> List[str]:
	return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]


def extract_ipc_sections(text: str) -> List[str]:
	found: List[str] = []
	if not text:
		return found

	for grp in re.findall(r"\b(\d{1,3}[A-Z]?(?:\s*/\s*\d{1,3}[A-Z]?)+)\s*IPC\b", text, flags=re.IGNORECASE):
		for sec in re.split(r"\s*/\s*", grp):
			sec = sec.strip().upper()
			if sec and sec not in found:
				found.append(sec)

	pattern = r"\b(?:Section|Sections|u/s|under sections?)\s+([0-9]{1,3}[A-Z]?(?:\s*(?:,|and|/|r/w|read with)\s*[0-9]{1,3}[A-Z]?){0,8})\s*(?:IPC|I\.P\.C\.|of IPC)?\b"
	for match in re.finditer(pattern, text, flags=re.IGNORECASE):
		for sec in re.findall(r"\b\d{1,3}[A-Z]?\b", match.group(1)):
			sec = sec.upper()
			if sec not in found:
				found.append(sec)

	return found


def detect_actions(text: str) -> List[str]:
	t = text.lower()
	return [action for action in ACTION_KEYWORDS if action in t]


def detect_weapon(text: str) -> str:
	t = text.lower()
	for weapon in WEAPON_KEYWORDS:
		if re.search(rf"\b{re.escape(weapon)}\b", t):
			return weapon
	return "unknown"


def detect_outcome(text: str) -> str:
	t = text.lower()
	if any(x in t for x in ["death", "died", "deceased", "murder", "homicide"]):
		return "death"
	if any(x in t for x in ["injury", "injured", "hurt", "grievous"]):
		return "injury"
	if any(x in t for x in ["cheat", "fraud", "forgery", "dishonest"]):
		return "fraud"
	if any(x in t for x in ["cruelty", "harassment", "dowry"]):
		return "cruelty"
	return "unknown"


def detect_intent(text: str) -> str:
	t = text.lower()
	if any(x in t for x in ["intentional", "intention", "premeditated", "knowingly", "motive", "with intent"]):
		return "intentional"
	if any(x in t for x in ["negligent", "accident", "accidental", "rash and negligent"]):
		return "negligent"
	return "unknown"


def detect_judgment(text: str) -> str:
	t = text.lower()
	if any(x in t for x in ["convicted", "found guilty"]):
		return "convicted"
	if any(x in t for x in ["acquitted", "found not guilty"]):
		return "acquitted"
	if "appeal is dismissed" in t:
		return "dismissed"
	if "appeal is allowed" in t:
		return "allowed"
	if "sentence" in t or "imprisonment" in t:
		return "sentenced"
	return "unknown"


def detect_court_year(text: str) -> tuple[str, str]:
	court = "Unknown Court"
	year = "Unknown"

	court_match = re.search(r"\b(Supreme Court|High Court|Sessions Court|District Court)\b", text, flags=re.IGNORECASE)
	if court_match:
		court = court_match.group(1)

	year_match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
	if year_match:
		year = year_match.group(1)

	return court, year


def build_metadata_template(case_name: str, text: str) -> Dict[str, object]:
	court, year = detect_court_year(text)
	return {
		"case_name": case_name,
		"chunk_type": "",
		"ipc_sections": extract_ipc_sections(text),
		"actions": detect_actions(text),
		"weapon": detect_weapon(text),
		"intent": detect_intent(text),
		"outcome": detect_outcome(text),
		"judgment": detect_judgment(text),
		"court": court,
		"year": year,
	}


def sentence_role_score(sentence: str, position: int, total: int) -> Dict[str, float]:
	s = sentence.lower()
	start_bias = 0.15 if position <= max(3, total // 8) else 0.0
	end_bias = 0.2 if position >= max(0, total - max(4, total // 8)) else 0.0

	scores = {
		"facts": start_bias,
		"issue": 0.0,
		"argument": 0.0,
		"reasoning": 0.0,
		"judgment": end_bias,
	}

	if any(k in s for k in ISSUE_MARKERS):
		scores["issue"] += 1.0
	if any(k in s for k in ARGUMENT_MARKERS):
		scores["argument"] += 1.4
	if any(k in s for k in REASONING_MARKERS):
		scores["reasoning"] += 1.3
	if any(k in s for k in JUDGMENT_MARKERS):
		scores["judgment"] += 1.5

	if any(k in s for k in ["incident", "deceased", "complaint", "f.i.r", "facts", "background", "occurred"]):
		scores["facts"] += 0.8

	if scores["argument"] == 0 and any(k in s for k in ["appellant", "respondent", "state", "defence", "defense"]):
		scores["argument"] += 0.5
	if scores["reasoning"] == 0 and "court" in s:
		scores["reasoning"] += 0.4

	return scores
