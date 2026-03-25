from __future__ import annotations

from utils.llm import generate_response
from utils.prompts import get_prosecution_prompt


def run_prosecution_agent(facts: str, statutes: str, precedents: str) -> str:
	prompt = get_prosecution_prompt(facts=facts, statutes=statutes, precedents=precedents)
	return generate_response(prompt)
