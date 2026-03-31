from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

try:
	from .chunker import build_multi_granular_chunks
	from .parser import clean_text, extract_case_name, read_pdf_text
except ImportError:
	from embedding_generation.chunker import build_multi_granular_chunks
	from embedding_generation.parser import clean_text, extract_case_name, read_pdf_text


INPUT_DIR = "cases"
OUTPUT_DIR = "processed_cases"


def process_pdf(file_path: Path, output_dir: Path) -> None:
	case_id = file_path.stem
	raw_text = read_pdf_text(file_path)
	cleaned_text = clean_text(raw_text)

	if not cleaned_text:
		cleaned_text = "No extractable legal text."

	case_name = extract_case_name(cleaned_text, case_id)
	chunks = build_multi_granular_chunks(case_name=case_name, full_text=cleaned_text)

	if len(chunks) < 4:
		# hard fallback to at least a few usable chunks
		fallback_chunks = build_multi_granular_chunks(case_name=case_name, full_text=(cleaned_text + " " + cleaned_text[:1500]))
		if len(fallback_chunks) > len(chunks):
			chunks = fallback_chunks

	payload = {
		"case_id": case_id,
		"case_name": case_name,
		"chunk_count": len(chunks),
		"chunks": chunks,
	}

	output_path = output_dir / f"{case_id}.json"
	output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
	input_dir = Path(INPUT_DIR)
	output_dir = Path(OUTPUT_DIR)
	output_dir.mkdir(parents=True, exist_ok=True)

	pdf_files = sorted(input_dir.glob("*.pdf"))
	for file_path in tqdm(pdf_files, desc="Processing PDFs"):
		process_pdf(file_path, output_dir)


if __name__ == "__main__":
	main()
