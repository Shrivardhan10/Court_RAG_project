from __future__ import annotations

from utils.llm import generate_response
from utils.prompts import get_defense_prompt


def run_defense_agent(
	facts: str,
	statutes: str,
	precedents: str,
	prosecution_output: str,
) -> str:
	prompt = get_defense_prompt(
		facts=facts,
		statutes=statutes,
		precedents=precedents,
		prosecution_output=prosecution_output,
	)
	return generate_response(prompt)
