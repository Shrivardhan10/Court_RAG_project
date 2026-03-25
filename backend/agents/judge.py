from __future__ import annotations

from utils.llm import generate_response
from utils.prompts import get_judge_prompt


def run_judge_agent(
	facts: str,
	prosecution_output: str,
	defense_output: str,
	statutes: str,
	precedents: str,
) -> str:
	prompt = get_judge_prompt(
		facts=facts,
		prosecution_output=prosecution_output,
		defense_output=defense_output,
		statutes=statutes,
		precedents=precedents,
	)
	return generate_response(prompt)
