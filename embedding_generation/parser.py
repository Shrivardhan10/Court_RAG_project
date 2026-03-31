from __future__ import annotations

import re
from pathlib import Path
from typing import List

from PyPDF2 import PdfReader


def read_pdf_text(file_path: Path) -> str:
	reader = PdfReader(str(file_path))
	pages = [page.extract_text() or "" for page in reader.pages]
	return "\n".join(pages)


def clean_text(text: str) -> str:
	text = text.replace("\r", "\n")
	cleaned_lines: List[str] = []

	for line in text.split("\n"):
		line = re.sub(r"\s+", " ", line).strip()
		if not line:
			continue

		lower = line.lower()
		if re.match(r"^\[\d{4}\]", line):
			continue
		if re.match(r"^[A-H]$", line):
			continue
		if "supreme court reports" in lower:
			continue
		if re.search(r"\b(scc|air|scr)\b", lower) and len(line.split()) <= 12:
			continue
		if re.match(r"^(appearance|appearances|present)\s*:?", lower):
			continue
		if re.match(r"^(for the )?(appellant|respondent|petitioner|defendant|plaintiff)\s*:?", lower):
			continue
		if re.search(r"\b(advocate|senior advocate|aor|counsel)\b", lower) and len(line.split()) < 16:
			continue

		cleaned_lines.append(line)

	text = "\n".join(cleaned_lines)
	text = re.sub(r"\[(?:para|paragraph)\s*\d+[^\]]*\]", "", text, flags=re.IGNORECASE)
	text = re.sub(r"\(\d{4}\)\s*\d+\s*(SCC|SCR)\s*\d+", "", text, flags=re.IGNORECASE)
	text = re.sub(r"\n{3,}", "\n\n", text)
	return text.strip()


def extract_case_name(text: str, fallback: str) -> str:
	for line in text.split("\n")[:25]:
		compact = re.sub(r"\s+", " ", line).strip()
		if re.search(r"\b(v\.|vs\.?|versus)\b", compact, flags=re.IGNORECASE):
			return compact[:200]
	return fallback.replace("_", " ")
