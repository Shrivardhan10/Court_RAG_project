"""
cleaner.py - Stage 0: Text Cleaning for Legal Case Files
"""

import re
from typing import List


# ── Line-level drop patterns ──────────────────────────────────────────────────

_COURT_LINE = re.compile(
    r"(High Court|Supreme Court|District Court|Sessions Court|Tribunal|"
    r"Appellate\s+Court|Civil\s+Court)",
    re.IGNORECASE,
)

_DATE_LINE = re.compile(
    r"^\s*(\d{1,2}\s+\w+\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\s*$"
)

_JUDGE_LINE = re.compile(
    r"(The\s+Judgment\s+was\s+delivered\s+by|Judgment\s+delivered\s+by"
    r"|HONOURABLE|HON'BLE|CORAM\s*:|\bDivon\b|\bDixon\b)",
    re.IGNORECASE,
)

_JUDGE_SIGNATURE = re.compile(
    r"^\s*[A-Z][a-zA-Z\s\.\,]+,?\s+(J\.|C\.J\.|JJ\.)\s*$"
)

# Case/appeal/petition number lines: "Cr.A. No 834 of 2002", "W.P. No. 117 of 1973"
_CASE_NUMBER_LINE = re.compile(
    r"^\s*[A-Za-z.\s]*(No\.?\s*\d+|Appeal\s+No|Petition\s+No|Case\s+No)"
    r"[\s\w./,-]*\d{4}\s*$",
    re.IGNORECASE,
)

# Explicit petition/appeal/writ prefixes
_CASE_PREFIX_LINE = re.compile(
    r"^\s*(C\.A\.|Cr\.A\.|Civil\s+Appeal|Criminal\s+Appeal|Writ\s+Petition|W\.P\."
    r"|S\.L\.P\.|Transfer\s+Petition|O\.A\.|R\.S\.A\.|C\.R\.P\.|Petition\s+No\."
    r"|Appeal\s+No\.)\s*(No\.?\s*)?\d",
    re.IGNORECASE,
)

_PROCEDURAL_LINE = re.compile(
    r"^\s*(Leave\s+granted|Delay\s+condoned"
    r"|The\s+appeal\s+is\s+dismissed|The\s+appeal\s+is\s+allowed"
    r"|Petition\s+dismissed|Petition\s+allowed"
    r"|Appeal\s+dismissed|Appeal\s+allowed"
    r"|Order\s+accordingly)\s*\.?\s*$",
    re.IGNORECASE,
)

# Case title: "X v Y" or "X v. Y" — short line with no sentence punctuation
_CASE_TITLE_LINE = re.compile(
    r"^\s*[A-Z][^.!?]{3,80}\bv\.?\s+[A-Z][^.!?]{2,80}\s*$"
)

# ── Inline strip patterns ─────────────────────────────────────────────────────

_CITATION_AIR    = re.compile(r"\(AIR\s*\d{4}\s+\w+\.?\s+\d+\)", re.IGNORECASE)
_CITATION_SCC    = re.compile(
    r"[\(\[]\d{4}[\)\]]?\s+\d+\s+\w+\s+\d+"
    r"|[\(\[]\d{4}\s+\d+\s+\w+\s+\d+[\)\]]",
    re.IGNORECASE,
)
_CITATION_INDLAW = re.compile(r"\d{4}\s+Indlaw\s+\w+\s+\d+", re.IGNORECASE)
_CITATION_AC     = re.compile(r"\(\d{4}\s+[A-Z][\w\.]+\s+\d+\)")
_XML_TOKENS      = re.compile(r"<[^>]+>")
_TRAILING_PARA   = re.compile(r"\s+\d+\.\s*$")


def _should_drop_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _COURT_LINE.search(s):
        return True
    if _DATE_LINE.match(s):
        return True
    if _JUDGE_LINE.search(s):
        return True
    if _JUDGE_SIGNATURE.match(s):
        return True
    if _CASE_NUMBER_LINE.match(s):
        return True
    if _CASE_PREFIX_LINE.match(s):
        return True
    if _PROCEDURAL_LINE.match(s):
        return True
    if _CASE_TITLE_LINE.match(s):
        return True
    return False


def clean_text(text: str) -> str:
    lines: List[str] = text.splitlines()
    kept: List[str] = []

    for line in lines:
        if _should_drop_line(line):
            continue
        line = _CITATION_AIR.sub("", line)
        line = _CITATION_SCC.sub("", line)
        line = _CITATION_INDLAW.sub("", line)
        line = _CITATION_AC.sub("", line)
        line = _XML_TOKENS.sub("", line)
        line = _TRAILING_PARA.sub("", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        kept.append(line)

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
