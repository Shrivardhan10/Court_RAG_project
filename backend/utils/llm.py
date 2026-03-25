from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


PRIMARY_MODEL = "google/flan-t5-small"
FALLBACK_MODEL = "sshleifer/tiny-gpt2"
MAX_NEW_TOKENS = 256
MAX_INPUT_TOKENS = 512


_model = None
_tokenizer = None
_mode: Optional[str] = None


def _looks_low_quality(response: str) -> bool:
	text = (response or "").strip()
	if not text:
		return True
	if text.startswith("[llm]"):
		return True
	if len(text) < 20:
		return True
	# Common degenerate outputs observed with very small models.
	if text.lower() in {"1.", "n/a", "na", "none"}:
		return True
	return False


def _looks_like_copied_case_header(response: str) -> bool:
	text = (response or "").lower()
	if not text:
		return False
	header_markers = [
		"supreme court of india",
		"appeal (crl.)",
		"the judgment was delivered by",
		"rank 1 |",
		"rank 2 |",
		"score=",
	]
	hits = sum(1 for marker in header_markers if marker in text)
	return hits >= 2


def _load_model():
	global _model, _tokenizer, _mode
	if _model is not None and _tokenizer is not None:
		return _model, _tokenizer, _mode

	print(f"[llm] Loading LLM model on CPU: {PRIMARY_MODEL}")
	try:
		_tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL)
		_model = AutoModelForSeq2SeqLM.from_pretrained(PRIMARY_MODEL)
		_model.to("cpu")
		_model.eval()
		_mode = "seq2seq"
		print("[llm] Primary model loaded.")
	except Exception as primary_error:
		print(f"[llm] Primary model failed: {primary_error}")
		print(f"[llm] Falling back to CPU model: {FALLBACK_MODEL}")
		_tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
		_model = AutoModelForCausalLM.from_pretrained(FALLBACK_MODEL)
		_model.to("cpu")
		_model.eval()
		_mode = "causal"
		print("[llm] Fallback model loaded.")

	return _model, _tokenizer, _mode


def generate_response(prompt: str) -> str:
	if not prompt or not prompt.strip():
		raise ValueError("Prompt must be a non-empty string.")

	model, tokenizer, mode = _load_model()

	try:
		base_prompt = prompt.strip()
		for attempt in range(2):
			current_prompt = base_prompt
			if attempt == 1:
				current_prompt = (
					f"{base_prompt}\n\n"
					"Instruction: Return a complete, legally reasoned answer with at least 5 lines. "
					"Do not return a fragment."
				)

			encoded = tokenizer(
				current_prompt,
				return_tensors="pt",
				truncation=True,
				max_length=MAX_INPUT_TOKENS,
			)

			with torch.no_grad():
				if mode == "seq2seq":
					output_ids = model.generate(
						input_ids=encoded["input_ids"],
						attention_mask=encoded.get("attention_mask"),
						max_new_tokens=MAX_NEW_TOKENS,
						min_new_tokens=48,
						do_sample=False,
						num_beams=4,
						length_penalty=1.0,
						early_stopping=True,
					)
					response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
				else:
					if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
						tokenizer.pad_token = tokenizer.eos_token

					output_ids = model.generate(
						input_ids=encoded["input_ids"],
						attention_mask=encoded.get("attention_mask"),
						max_new_tokens=MAX_NEW_TOKENS,
						min_new_tokens=48,
						do_sample=False,
						num_beams=4,
						pad_token_id=tokenizer.pad_token_id,
						early_stopping=True,
					)
					full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
					response = (
						full_text[len(current_prompt) :].strip()
						if full_text.startswith(current_prompt)
						else full_text.strip()
					)

			if not _looks_low_quality(response):
				return response

		if response and _looks_like_copied_case_header(response):
			rewrite_prompt = (
				"Rewrite the following into a concise legal argument in your own words. "
				"Keep legal meaning, remove procedural/caption metadata, and provide reasoned points:\n\n"
				f"{response}"
			)
			encoded = tokenizer(
				rewrite_prompt,
				return_tensors="pt",
				truncation=True,
				max_length=MAX_INPUT_TOKENS,
			)
			with torch.no_grad():
				output_ids = model.generate(
					input_ids=encoded["input_ids"],
					attention_mask=encoded.get("attention_mask"),
					max_new_tokens=MAX_NEW_TOKENS,
					min_new_tokens=48,
					do_sample=False,
					num_beams=4,
					early_stopping=True,
					pad_token_id=(
						tokenizer.pad_token_id
						if tokenizer.pad_token_id is not None
						else tokenizer.eos_token_id
					),
				)
			rewritten = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
			if rewritten and not _looks_low_quality(rewritten):
				return rewritten

		return response if response else "[llm] No response generated."
	except Exception as error:
		return f"[llm] Generation failed: {error}"
