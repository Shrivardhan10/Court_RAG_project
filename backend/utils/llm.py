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
		encoded = tokenizer(
			prompt,
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
					do_sample=False,
				)
				response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
			else:
				if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
					tokenizer.pad_token = tokenizer.eos_token

				output_ids = model.generate(
					input_ids=encoded["input_ids"],
					attention_mask=encoded.get("attention_mask"),
					max_new_tokens=MAX_NEW_TOKENS,
					do_sample=False,
					pad_token_id=tokenizer.pad_token_id,
				)
				full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
				response = full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text.strip()

		if not response:
			return "[llm] No response generated."
		return response
	except Exception as error:
		return f"[llm] Generation failed: {error}"
