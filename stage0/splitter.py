"""
splitter.py - Stage 0: Sentence Splitting for Legal Case Text
"""

import re
from typing import List


# ── Paragraph marker ─────────────────────────────────────────────────────────
_PARA_SPLIT = re.compile(r"(?:^|\n)\s*\d+\.\s+")

# ── Structured list items: (i), (ii), (a), (b), (1), etc. ───────────────────
_LIST_ITEM = re.compile(r"(?=\(\s*(?:[ivxlcdmIVXLCDM]+|[a-zA-Z]|\d+)\s*\))")

# ── Abbreviation placeholders (protect before splitting) ─────────────────────
_ABBREV_SUBS = [
    (re.compile(r"\bv\.\s"),        "v__ "),
    (re.compile(r"\bNo\.\s"),       "No__ "),
    (re.compile(r"\bu/s\.\s"),      "us__ "),
    (re.compile(r"\bS\.\s"),        "S__ "),
    (re.compile(r"\bCl\.\s"),       "Cl__ "),
    (re.compile(r"\bIPC\.\s"),      "IPC__ "),
    (re.compile(r"\bArt\.\s"),      "Art__ "),
    (re.compile(r"\bsec\.\s"),      "sec__ "),
    (re.compile(r"\bvs\.\s"),       "vs__ "),
    (re.compile(r"\bMr\.\s"),       "Mr__ "),
    (re.compile(r"\bMrs\.\s"),      "Mrs__ "),
    (re.compile(r"\bDr\.\s"),       "Dr__ "),
    (re.compile(r"\bProf\.\s"),     "Prof__ "),
    (re.compile(r"\bSt\.\s"),       "St__ "),
    (re.compile(r"\bDept\.\s"),     "Dept__ "),
    (re.compile(r"\bGovt\.\s"),     "Govt__ "),
    (re.compile(r"\bCorp\.\s"),     "Corp__ "),
    (re.compile(r"\bLtd\.\s"),      "Ltd__ "),
    (re.compile(r"\bu\.p\.\s", re.IGNORECASE), "UP__ "),
    (re.compile(r"\bp\.w\.d\.\s", re.IGNORECASE), "PWD__ "),
    # Single uppercase letter abbreviations: "A. ", "B. " etc.
    (re.compile(r"\b([A-Z])\.\s"), r"\1__ "),
]

_RESTORE_SUBS = [
    ("v__ ",    "v. "),
    ("No__ ",   "No. "),
    ("us__ ",   "u/s. "),
    ("S__ ",    "S. "),
    ("Cl__ ",   "Cl. "),
    ("IPC__ ",  "IPC. "),
    ("Art__ ",  "Art. "),
    ("sec__ ",  "sec. "),
    ("vs__ ",   "vs. "),
    ("Mr__ ",   "Mr. "),
    ("Mrs__ ",  "Mrs. "),
    ("Dr__ ",   "Dr. "),
    ("Prof__ ", "Prof. "),
    ("St__ ",   "St. "),
    ("Dept__ ", "Dept. "),
    ("Govt__ ", "Govt. "),
    ("Corp__ ", "Corp. "),
    ("Ltd__ ",  "Ltd. "),
    ("UP__ ",   "U.P. "),
    ("PWD__ ",  "P.W.D. "),
]

# ── Sentence boundary patterns ────────────────────────────────────────────────
# 1. After .!? followed by whitespace + capital/quote
_BOUNDARY_SPACE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\u2018\u201c])")
# 2. Lowercase letter immediately followed by . then uppercase (no space)
_BOUNDARY_NOSPACE = re.compile(r"(?<=[a-z])\.(?=[A-Z])")

# ── Long sentence splitters ───────────────────────────────────────────────────
_LONG_SPLIT = re.compile(
    r",\s+(?=(?:that|which|where|when|if|as|but|and|or|however|therefore|thus)\s)",
    re.IGNORECASE,
)


def _protect_abbrevs(text: str) -> str:
    for pattern, repl in _ABBREV_SUBS:
        text = pattern.sub(repl, text)
    return text


def _restore_abbrevs(text: str) -> str:
    for placeholder, original in _RESTORE_SUBS:
        text = text.replace(placeholder, original)
        # Also handle trailing (end of string, no trailing space)
        text = text.replace(placeholder.rstrip(), original.rstrip())
    # Restore single-letter abbreviations: "X__ " → "X. " and "X__" → "X."
    text = re.sub(r"\b([A-Z])__\s", r"\1. ", text)
    text = re.sub(r"\b([A-Z])__$", r"\1.", text)
    # Catch any remaining __ placeholders
    text = re.sub(r"__", ".", text)
    return text


def _word_count(s: str) -> int:
    return len(s.split())


def _split_long(sentence: str) -> List[str]:
    """Split sentences > 40 words on comma + conjunction boundaries."""
    if _word_count(sentence) <= 40:
        return [sentence]
    parts = _LONG_SPLIT.split(sentence)
    result: List[str] = []
    for p in parts:
        p = p.strip()
        if p:
            result.append(p)
    return result if len(result) > 1 else [sentence]


def _split_paragraph(text: str) -> List[str]:
    """Split one paragraph chunk into sentences."""
    # Protect abbreviations
    protected = _protect_abbrevs(text)

    # Split on no-space boundary first (e.g. "1951.When")
    protected = _BOUNDARY_NOSPACE.sub(". ", protected)

    # Split on structured list items: (i), (ii), (a), (b)
    list_parts = _LIST_ITEM.split(protected)

    sentences: List[str] = []
    for part in list_parts:
        part = part.strip()
        if not part:
            continue
        # Split on space boundary
        chunks = _BOUNDARY_SPACE.split(part)
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk:
                sentences.append(chunk)

    # Restore abbreviations
    sentences = [_restore_abbrevs(s) for s in sentences]
    return sentences


def split_sentences(text: str) -> List[str]:
    """
    Split cleaned legal text into sentences.

    1. Split on numbered paragraph markers.
    2. Protect abbreviations (v., No., S., etc.) from false splits.
    3. Split on structured list items (i), (ii), (a), (b).
    4. Split on sentence boundaries (. ! ? followed by capital).
    5. Split no-space boundaries: "1951.When" → two sentences.
    6. Restore abbreviations.
    7. Force-split sentences > 40 words on comma+conjunction.
    8. Merge fragments < 6 words into previous sentence.
    """
    if not text or not text.strip():
        return []

    # Step 1: split on paragraph markers
    chunks = _PARA_SPLIT.split(text)
    para_chunks = [c.strip() for c in chunks if c.strip()]

    raw: List[str] = []
    for chunk in para_chunks:
        raw.extend(_split_paragraph(chunk))

    # Step 2: force-split long sentences
    expanded: List[str] = []
    for sent in raw:
        expanded.extend(_split_long(sent))

    # Step 3: strip, discard empty, merge short fragments (< 6 words)
    result: List[str] = []
    for sent in expanded:
        sent = sent.strip()
        if not sent:
            continue
        if _word_count(sent) < 6:
            if result:
                result[-1] = result[-1].rstrip() + " " + sent
        else:
            result.append(sent)

    return result
