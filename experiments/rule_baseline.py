from __future__ import annotations

import re
from typing import Dict, List


def map_ipc_sections(fact_text: str) -> List[str]:
    text = (fact_text or "").lower()
    sections: List[str] = []

    if any(x in text for x in ["murder", "killed", "death", "fatal", "poisoned", "stabbed", "fired"]):
        sections.append("302")
    if any(x in text for x in ["culpable", "without premeditation", "sudden fight"]):
        sections.append("304")
    if any(x in text for x in ["grievous", "iron rod", "fracture", "maimed"]):
        sections.append("325")
    if any(x in text for x in ["forged", "forgery", "fake document"]):
        sections.append("467")
    if any(x in text for x in ["cheat", "cheated", "fraud"]):
        sections.append("420")
    if any(x in text for x in ["house", "night", "house trespass", "broke into"]):
        sections.append("449")

    if not sections:
        sections = ["323"]

    seen = set()
    ordered: List[str] = []
    for sec in sections:
        if sec not in seen:
            seen.add(sec)
            ordered.append(sec)
    return ordered


def judge_sentence_from_ipc(ipc_sections: List[str]) -> Dict[str, str]:
    ipc_set = set(ipc_sections)
    if "302" in ipc_set:
        return {
            "sentence": "imprisonment for life",
            "compensation": "Rs 100000",
        }
    if "304" in ipc_set:
        return {
            "sentence": "rigorous imprisonment for 10 years",
            "compensation": "Rs 60000",
        }
    if "325" in ipc_set:
        return {
            "sentence": "rigorous imprisonment for 5 years",
            "compensation": "Rs 40000",
        }
    if "420" in ipc_set or "467" in ipc_set:
        return {
            "sentence": "rigorous imprisonment for 7 years",
            "compensation": "Rs 50000",
        }
    return {
        "sentence": "rigorous imprisonment for 2 years",
        "compensation": "Rs 10000",
    }


def build_rule_outputs(fact_text: str) -> Dict[str, str]:
    ipc_sections = map_ipc_sections(fact_text)
    sentence_info = judge_sentence_from_ipc(ipc_sections)
    joined = ", ".join(f"IPC {x}" for x in ipc_sections)

    prosecution = (
        f"The prosecution submits that the facts disclose offences under {joined} and seeks conviction with "
        f"{sentence_info['sentence']} along with compensation of {sentence_info['compensation']}."
    )
    defence = (
        "The defence seeks reduction in sentence, challenges intention, and requests benefit under lesser offence "
        "principles wherever legally sustainable."
    )
    judge = (
        f"After considering the record, the Court convicts under {joined} and awards {sentence_info['sentence']} "
        f"with compensation of {sentence_info['compensation']}."
    )
    return {
        "prosecution": re.sub(r"\s+", " ", prosecution).strip(),
        "defence": re.sub(r"\s+", " ", defence).strip(),
        "judge": re.sub(r"\s+", " ", judge).strip(),
    }
