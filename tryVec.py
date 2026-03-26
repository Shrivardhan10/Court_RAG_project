import os
import re
import json
from pathlib import Path
from PyPDF2 import PdfReader
from tqdm import tqdm

INPUT_DIR = "tryData"        # folder with your 850 PDFs
OUTPUT_DIR = "processed_cases"

# Section-specific chunk budgets tuned for legal reasoning retrieval.
CHUNK_WORDS_BY_SECTION = {
    "title": 80,
    "facts": 180,
    "issues": 140,
    "prosecution_arguments": 150,
    "defense_arguments": 150,
    "court_analysis": 170,
    "decision": 120,
    "other": 160,
}

OVERLAP_SENTENCES_BY_SECTION = {
    "title": 0,
    "facts": 1,
    "issues": 0,
    "prosecution_arguments": 0,
    "defense_arguments": 0,
    "court_analysis": 1,
    "decision": 0,
    "other": 1,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- SECTION DEFINITIONS ----------
SECTION_PATTERNS = {
    "facts": [
        r"^facts?$",
        r"^brief facts?$",
        r"^background$",
        r"^factual matrix$",
    ],
    "issues": [
        r"^issues?$",
        r"^points? for determination$",
        r"^question[s]? for consideration$",
    ],
    "prosecution_arguments": [
        r"^prosecution arguments?$",
        r"^submissions on behalf of the state$",
        r"^arguments for the state$",
        r"^contentions? of the respondent$",
    ],
    "defense_arguments": [
        r"^defen[cs]e arguments?$",
        r"^submissions on behalf of the accused$",
        r"^arguments for the appellant$",
        r"^contentions? of the appellant$",
    ],
    "court_analysis": [
        r"^analysis$",
        r"^reasoning$",
        r"^discussion$",
        r"^our findings$",
        r"^consideration$",
    ],
    "decision": [
        r"^decision$",
        r"^order$",
        r"^final order$",
        r"^result$",
        r"^relief$",
        r"^sentence$",
        r"^conclusion$",
    ],
}

ROLE_BY_SECTION = {
    "title": ["prosecution", "defense", "judge"],
    "facts": ["prosecution", "defense", "judge"],
    "issues": ["prosecution", "defense", "judge"],
    "prosecution_arguments": ["prosecution", "judge"],
    "defense_arguments": ["defense", "judge"],
    "court_analysis": ["judge", "prosecution", "defense"],
    "decision": ["judge", "prosecution", "defense"],
    "other": ["prosecution", "defense", "judge"],
}

SIDE_HINT_BY_SECTION = {
    "prosecution_arguments": "prosecution",
    "defense_arguments": "defense",
    "court_analysis": "court",
    "decision": "court",
}

ROLE_PRIORITIES = {
    "prosecution": [
        "facts",
        "issues",
        "prosecution_arguments",
        "court_analysis",
        "decision",
        "title",
        "defense_arguments",
        "other",
    ],
    "defense": [
        "facts",
        "issues",
        "defense_arguments",
        "court_analysis",
        "decision",
        "title",
        "prosecution_arguments",
        "other",
    ],
    "judge": [
        "facts",
        "issues",
        "prosecution_arguments",
        "defense_arguments",
        "court_analysis",
        "decision",
        "title",
        "other",
    ],
}


# ---------- TEXT CLEANING ----------
def _is_noise_line(line):
    lower = line.lower().strip()
    if not lower:
        return True

    # Common scanned-report artifacts.
    if re.fullmatch(r"[a-h\s\d]+", lower):
        return True
    if "supreme court reports" in lower:
        return True
    if re.fullmatch(r"h\d+\s*\[\d{4}\].*", lower):
        return True
    if re.fullmatch(r"\[\d{4}\].*s\.c\.r\..*", lower):
        return True
    if re.fullmatch(r"\d+\s+[a-z\s\.\[\]\-]+", lower) and "v" not in lower:
        return True

    return False


def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for raw_line in text.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if _is_noise_line(line):
            continue
        lines.append(line)

    # Keep paragraph boundaries by converting long punctuation boundaries to new lines.
    merged = "\n".join(lines)
    merged = re.sub(r"(?<=[.;:])\s+(?=[A-Z][a-z])", "\n", merged)
    merged = re.sub(r"\n{3,}", "\n\n", merged)
    return merged.strip()


def _extract_case_title(lines):
    candidates = []
    for line in lines[:40]:
        normalized = re.sub(r"\s+", " ", line).strip()
        lower = normalized.lower()
        if " v " in lower or " vs " in lower or "versus" in lower:
            if "supreme court reports" in lower:
                continue
            if re.fullmatch(r"[a-z]?\d+\s*\[\d{4}\].*", lower):
                continue
            candidates.append(normalized)

    if candidates:
        # Pick the shortest plausible caption line.
        return sorted(candidates, key=len)[0][:220]

    for line in lines[:20]:
        normalized = re.sub(r"\s+", " ", line).strip()
        if len(normalized.split()) >= 4:
            return normalized[:220]

    return "Untitled Case"


def _looks_like_heading(line):
    stripped = line.strip().strip(":")
    if not stripped:
        return False
    words = stripped.split()
    if len(words) > 10:
        return False
    if stripped.isupper():
        return True
    if re.fullmatch(r"\d+[.)]?\s+[A-Za-z].*", stripped):
        return True
    if stripped.lower() in {
        "facts",
        "issue",
        "issues",
        "analysis",
        "reasoning",
        "decision",
        "order",
        "conclusion",
    }:
        return True
    return False


def _map_heading_to_section(line):
    normalized = re.sub(r"[^a-z0-9\s]", "", line.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)

    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.fullmatch(pattern, normalized):
                return section

    # Soft heading cues used in many judgments.
    if "learned counsel for the state" in normalized or "public prosecutor" in normalized:
        return "prosecution_arguments"
    if "learned counsel for the appellant" in normalized or "for the accused" in normalized:
        return "defense_arguments"
    if "it was contended" in normalized or "it is submitted" in normalized:
        return "prosecution_arguments"
    if "it was argued" in normalized or "on behalf of the appellant" in normalized:
        return "defense_arguments"
    if "we hold" in normalized or "we find" in normalized:
        return "court_analysis"
    if "accordingly" in normalized or "appeal is" in normalized:
        return "decision"
    return None


# ---------- SECTION SPLITTING ----------
def split_sections(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return []

    sections = []
    title = _extract_case_title(lines)
    sections.append({"section": "title", "text": title})

    current_section = "facts"
    buffer = []

    for line in lines:
        # Paragraph number cues in judgments help preserve argument units.
        numbered_para = re.match(r"^\d+[.)]?\s+", line) is not None

        maybe_section = None
        if _looks_like_heading(line):
            maybe_section = _map_heading_to_section(line)

        if maybe_section:
            if buffer:
                sections.append({"section": current_section, "text": " ".join(buffer).strip()})
                buffer = []
            current_section = maybe_section
            continue

        inferred = _map_heading_to_section(line)
        if inferred in {"prosecution_arguments", "defense_arguments", "court_analysis", "decision"}:
            if buffer:
                sections.append({"section": current_section, "text": " ".join(buffer).strip()})
                buffer = []
            current_section = inferred

        if numbered_para and buffer and len(" ".join(buffer).split()) > 120:
            sections.append({"section": current_section, "text": " ".join(buffer).strip()})
            buffer = []

        buffer.append(line)

    if buffer:
        sections.append({"section": current_section, "text": " ".join(buffer).strip()})

    return [section for section in sections if section["text"]]


# ---------- SMART CHUNKING ----------
def _split_into_paragraphs(text):
    if not text:
        return []

    raw_parts = re.split(r"\n\s*\n", text)
    paragraphs = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        # Further split on paragraph numbering while retaining grouped meaning.
        numbered = re.split(r"(?=(?:^|\s)\d+[.)]\s)", part)
        for item in numbered:
            normalized = re.sub(r"\s+", " ", item).strip()
            if normalized:
                paragraphs.append(normalized)

    return paragraphs


def _chunk_paragraphs(paragraphs, chunk_words, paragraph_overlap):
    if not paragraphs:
        return []

    chunks: list[tuple[str, int, int]] = []
    current = []
    current_words = 0
    start_idx = 0

    for idx, paragraph in enumerate(paragraphs):
        paragraph_words = len(paragraph.split())

        if current and current_words + paragraph_words > chunk_words:
            chunks.append((" ".join(current).strip(), start_idx + 1, idx))
            overlap = current[-paragraph_overlap:] if paragraph_overlap > 0 else []
            current = list(overlap)
            current_words = sum(len(item.split()) for item in current)
            start_idx = max(0, idx - len(overlap))

        if not current:
            start_idx = idx
        current.append(paragraph)
        current_words += paragraph_words

    if current:
        chunks.append((" ".join(current).strip(), start_idx + 1, len(paragraphs)))

    return [chunk for chunk in chunks if chunk]


def build_chunk_records(case_id, sections):
    records = []
    chunk_counter = 1

    for section_item in sections:
        section = section_item["section"]
        text = section_item["text"]
        paragraphs = _split_into_paragraphs(text)
        chunk_words = CHUNK_WORDS_BY_SECTION.get(section, CHUNK_WORDS_BY_SECTION["other"])
        overlap = OVERLAP_SENTENCES_BY_SECTION.get(section, OVERLAP_SENTENCES_BY_SECTION["other"])
        chunks = _chunk_paragraphs(paragraphs, chunk_words=chunk_words, paragraph_overlap=overlap)

        for chunk_text, para_start, para_end in chunks:
            role_tags = ROLE_BY_SECTION.get(section, ROLE_BY_SECTION["other"])
            records.append(
                {
                    "case_id": case_id,
                    "chunk_id": f"{case_id}_CH{chunk_counter:03d}",
                    "section": section,
                    "side_hint": SIDE_HINT_BY_SECTION.get(section, "neutral"),
                    "role_tags": role_tags,
                    "paragraph_start": para_start,
                    "paragraph_end": para_end,
                    "word_count": len(chunk_text.split()),
                    "text": chunk_text,
                }
            )
            chunk_counter += 1

    return records


def _sorted_for_role(records, role):
    priority = ROLE_PRIORITIES[role]
    rank = {section: idx for idx, section in enumerate(priority)}

    filtered = [record for record in records if role in record.get("role_tags", [])]
    return sorted(filtered, key=lambda item: (rank.get(item["section"], 999), item["chunk_id"]))


def _write_role_views(case_id, chunk_records):
    for role in ("prosecution", "defense", "judge"):
        role_records = _sorted_for_role(chunk_records, role)
        txt_output = os.path.join(OUTPUT_DIR, f"{case_id}_{role}.txt")
        jsonl_output = os.path.join(OUTPUT_DIR, f"{case_id}_{role}.jsonl")

        with open(txt_output, "w", encoding="utf-8") as txt_fp:
            for chunk in role_records:
                txt_fp.write(f"\n--- {chunk['chunk_id']} ---\n")
                txt_fp.write(f"SECTION: {chunk['section']}\n")
                txt_fp.write(f"ROLE: {role}\n")
                txt_fp.write(f"WORDS: {chunk['word_count']}\n")
                txt_fp.write(chunk["text"] + "\n")

        with open(jsonl_output, "w", encoding="utf-8") as jsonl_fp:
            for chunk in role_records:
                payload = dict(chunk)
                payload["target_role"] = role
                jsonl_fp.write(json.dumps(payload, ensure_ascii=True) + "\n")


# ---------- PDF TO TEXT ----------
def extract_pdf_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        page_texts = []

        for page in reader.pages:
            if page.extract_text():
                page_texts.append(page.extract_text())

        return clean_text("\n".join(page_texts))

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None


# ---------- MAIN PROCESS ----------
def process_all_pdfs():
    pdf_files = list(Path(INPUT_DIR).glob("*.pdf"))

    for idx, pdf_path in enumerate(tqdm(pdf_files, desc="Processing PDFs"), start=1):
        raw_text = extract_pdf_text(pdf_path)
        if not raw_text:
            continue

        sections = split_sections(raw_text)

        case_id = f"case{idx:04d}"
        chunk_records = build_chunk_records(case_id, sections)
        if not chunk_records:
            continue

        txt_output = os.path.join(OUTPUT_DIR, f"{case_id}.txt")
        jsonl_output = os.path.join(OUTPUT_DIR, f"{case_id}.jsonl")

        with open(txt_output, "w", encoding="utf-8") as txt_fp:
            for chunk in chunk_records:
                txt_fp.write(f"\n--- {chunk['chunk_id']} ---\n")
                txt_fp.write(f"SECTION: {chunk['section']}\n")
                txt_fp.write(f"ROLE_TAGS: {', '.join(chunk['role_tags'])}\n")
                txt_fp.write(chunk["text"] + "\n")

        with open(jsonl_output, "w", encoding="utf-8") as jsonl_fp:
            for chunk in chunk_records:
                jsonl_fp.write(json.dumps(chunk, ensure_ascii=True) + "\n")

        _write_role_views(case_id, chunk_records)

    print("Processing complete!")


if __name__ == "__main__":
    process_all_pdfs()