"""
rules.py - Stage 1: Keyword-based heuristics for hybrid classification.
"""

import re
from typing import Dict, Optional

# ── Keyword lists ─────────────────────────────────────────────────────────────

REASONING_KEYWORDS = [
    "court observed", "we find", "we hold", "in our view", "in our opinion",
    "it is clear that", "it is evident", "it is well settled", "therefore",
    "thus", "hence", "accordingly", "consequently", "we therefore", "we accordingly",
    "the court held", "this court held", "we are of the view", "we are of the opinion",
    "on a proper reading", "the ratio", "the legal position", "no merit",
    "without merit", "has no substance", "for the above reasons",
    "for the foregoing reasons", "we do not agree", "we agree", "we reject",
    "it follows that", "it must be held", "we are satisfied",
]

ARGUMENT_KEYWORDS = [
    "argued", "contended", "submitted", "claimed", "pleaded", "asserted",
    "according to", "the defence", "the prosecution", "counsel for",
    "learned counsel", "it was argued", "it was submitted", "it was contended",
    "it was urged", "the contention", "the argument", "the plea",
    "on behalf of", "the appellant argued", "the respondent submitted",
    "the petitioner contended", "alleged", "denied", "disputed",
    "the ground that", "the allegation", "relied upon", "reliance was placed",
    "the case of the", "opposed", "the objection",
]

FACT_KEYWORDS = [
    "fir", "first information report", "post-mortem", "post mortem",
    "injury", "wound", "witness", "examined", "evidence", "exhibit",
    "deposed", "testified", "arrested", "detained", "charged", "convicted",
    "acquitted", "sentenced", "filed", "registered", "recorded", "produced",
    "recovered", "seized", "found guilty", "found dead", "was shot",
    "was killed", "was beaten", "was assaulted", "the deceased", "the complainant",
    "the victim", "the incident", "the occurrence", "incorporated", "appointed",
    "promoted", "terminated", "dismissed from service", "enrolled", "discharged",
]

# Compile patterns
_REASONING_RE = re.compile(
    "|".join(r"\b" + re.escape(k) + r"\b" for k in REASONING_KEYWORDS),
    re.IGNORECASE,
)
_ARGUMENT_RE = re.compile(
    "|".join(r"\b" + re.escape(k) + r"\b" for k in ARGUMENT_KEYWORDS),
    re.IGNORECASE,
)
_FACT_RE = re.compile(
    "|".join(r"\b" + re.escape(k) + r"\b" for k in FACT_KEYWORDS),
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"
    r"|\b\d{1,2}\s+(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{4}\b",
    re.IGNORECASE,
)


def score_sentence(sentence: str) -> Dict[str, int]:
    """Return keyword match counts per label."""
    r = len(_REASONING_RE.findall(sentence))
    a = len(_ARGUMENT_RE.findall(sentence))
    f = len(_FACT_RE.findall(sentence))
    if _DATE_RE.search(sentence):
        f += 1
    return {"REASONING": r, "ARGUMENT": a, "FACT": f}


def get_rule_label(sentence: str) -> Optional[str]:
    """
    Return a label if there is a clear keyword signal, else None.
    Priority: REASONING > ARGUMENT > FACT
    """
    scores = score_sentence(sentence)
    r, a, f = scores["REASONING"], scores["ARGUMENT"], scores["FACT"]
    if r >= 2 and r >= a:
        return "REASONING"
    if a >= 2 and a > r:
        return "ARGUMENT"
    if r == 1 and r > a:
        return "REASONING"
    if a == 1 and a > r:
        return "ARGUMENT"
    return None


def get_rule_strength(sentence: str) -> int:
    """Return total keyword match count as a strength signal."""
    scores = score_sentence(sentence)
    return scores["REASONING"] + scores["ARGUMENT"] + scores["FACT"]
