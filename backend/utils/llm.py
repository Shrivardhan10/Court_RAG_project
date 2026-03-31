from __future__ import annotations

import os
from pathlib import Path

import google.generativeai as genai
from dotenv import find_dotenv, load_dotenv


MODEL_NAME = "gemini-1.5-flash"
FALLBACK_MODELS = ["models/gemini-2.0-flash", "models/gemini-flash-latest"]
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 1000
REQUEST_TIMEOUT_SECONDS = 45


_model = None
_configured_api_key = None
_active_model_name = None


def _configure_client() -> None:
	global _configured_api_key
	local_dotenv = Path(__file__).resolve().parents[1] / ".env"
	dotenv_path = ""
	if local_dotenv.exists():
		dotenv_path = str(local_dotenv)
		load_dotenv(dotenv_path=dotenv_path, override=False)
	else:
		dotenv_path = find_dotenv(usecwd=True)
	if dotenv_path:
		load_dotenv(dotenv_path=dotenv_path, override=False)
	else:
		load_dotenv(override=False)
	api_key = os.getenv("GEMINI_API_KEY", "").strip()
	if not api_key:
		if dotenv_path:
			print(f"[llm] GEMINI_API_KEY missing after loading .env at: {dotenv_path}")
		else:
			print("[llm] GEMINI_API_KEY missing and .env file was not found via find_dotenv().")
		raise RuntimeError("Gemini initialization failed: GEMINI_API_KEY is not set. Add it to your .env file.")

	if _configured_api_key != api_key:
		try:
			genai.configure(api_key=api_key)
			_configured_api_key = api_key
		except Exception as error:
			raise RuntimeError(f"Gemini initialization failed during configure(): {error}") from error


def _get_model():
	global _model, _active_model_name
	if _model is None:
		_configure_client()
		try:
			_model = genai.GenerativeModel(MODEL_NAME)
			_active_model_name = MODEL_NAME
		except Exception as error:
			raise RuntimeError(f"Gemini initialization failed for model '{MODEL_NAME}': {error}") from error
	return _model


def _switch_model(model_name: str):
	global _model, _active_model_name
	_model = genai.GenerativeModel(model_name)
	_active_model_name = model_name
	return _model


def generate_response(prompt: str) -> str:
	if not prompt or not prompt.strip():
		raise ValueError("Prompt must be a non-empty string.")

	try:
		model = _get_model()
		response = model.generate_content(
			prompt.strip(),
			generation_config={
				"temperature": TEMPERATURE,
				"max_output_tokens": MAX_OUTPUT_TOKENS,
			},
			request_options={"timeout": REQUEST_TIMEOUT_SECONDS},
		)
		text = (getattr(response, "text", "") or "").strip()
		if text:
			return text

		return "[llm] Generation failed: empty response from Gemini API."
	except Exception as error:
		err_text = str(error).lower()
		if "not found" in err_text or "unsupported for generatecontent" in err_text:
			for candidate in FALLBACK_MODELS:
				try:
					model = _switch_model(candidate)
					response = model.generate_content(
						prompt.strip(),
						generation_config={
							"temperature": TEMPERATURE,
							"max_output_tokens": MAX_OUTPUT_TOKENS,
						},
						request_options={"timeout": REQUEST_TIMEOUT_SECONDS},
					)
					text = (getattr(response, "text", "") or "").strip()
					if text:
						return text
				except Exception:
					continue
		return f"[llm] Generation failed: {error}"
