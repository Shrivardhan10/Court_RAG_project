from __future__ import annotations


def get_fact_extraction_prompt(case_text: str) -> str:
	return (
		"You are a legal fact extractor. Extract structured facts from the case text.\n"
		"Return only these fields in the exact format:\n"
		"act: ...\n"
		"intent: ...\n"
		"weapon: ...\n"
		"outcome: ...\n\n"
		f"Case text:\n{case_text.strip()}"
	)


def get_prosecution_prompt(facts: str, statutes: str) -> str:
	return (
		"You are the prosecution lawyer.\n"
		"Argue why the accused is guilty based on facts and relevant statutes.\n"
		"Be concise and structured with these sections:\n"
		"1) Charges\n2) Key Evidence\n3) Statutory Basis\n4) Conclusion\n\n"
		f"Facts:\n{facts.strip()}\n\n"
		f"Relevant Statutes:\n{statutes.strip()}"
	)


def get_defense_prompt(facts: str, prosecution_output: str) -> str:
	return (
		"You are the defense lawyer.\n"
		"Counter the prosecution argument and reduce criminal liability where possible.\n"
		"Be concise and structured with these sections:\n"
		"1) Weaknesses in Prosecution Case\n2) Alternative Interpretation\n3) Mitigating Factors\n4) Relief Sought\n\n"
		f"Facts:\n{facts.strip()}\n\n"
		f"Prosecution Argument:\n{prosecution_output.strip()}"
	)


def get_judge_prompt(
	facts: str,
	prosecution_output: str,
	defense_output: str,
	statutes: str,
) -> str:
	return (
		"You are the judge. Compare prosecution and defense arguments and give a reasoned verdict.\n"
		"Return in this structure:\n"
		"Findings:\n- ...\n"
		"Reasoning:\n- ...\n"
		"Final Verdict:\n- ...\n"
		"Suggested Sentence/Relief:\n- ...\n\n"
		f"Facts:\n{facts.strip()}\n\n"
		f"Relevant Statutes:\n{statutes.strip()}\n\n"
		f"Prosecution Argument:\n{prosecution_output.strip()}\n\n"
		f"Defense Argument:\n{defense_output.strip()}"
	)
