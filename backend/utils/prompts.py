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


def get_prosecution_prompt(facts: str, statutes: str, precedents: str) -> str:
	return (
		"You are the prosecution lawyer.\n"
		"Argue why the accused is guilty based on facts, relevant statutes, and relevant precedents.\n"
		"Important rules:\n"
		"- Use facts as the primary basis of argument.\n"
		"- Use precedents only as citations like [P1], [P2], [P3], not as copied text.\n"
		"- Do not repeat case captions, court names, or appeal numbers.\n"
		"Write complete sentences only. Do not output placeholders or fragments.\n"
		"Be concise and structured with these sections:\n"
		"1) Charges\n"
		"2) Key Evidence from Facts\n"
		"3) Statutory Basis\n"
		"4) Precedent Support (citation + one-line relevance)\n"
		"5) Conclusion\n\n"
		f"Facts:\n{facts.strip()}\n\n"
		f"Relevant Statutes:\n{statutes.strip()}\n\n"
		f"Relevant Precedents:\n{precedents.strip()}"
	)


def get_defense_prompt(
	facts: str,
	statutes: str,
	precedents: str,
	prosecution_output: str,
) -> str:
	return (
		"You are the defense lawyer.\n"
		"Counter the prosecution argument and reduce criminal liability where possible.\n"
		"Important rules:\n"
		"- Attack prosecution on intent, causation, witness reliability, and charge selection where applicable.\n"
		"- Distinguish precedents by factual differences; cite only as [P1], [P2], [P3].\n"
		"- Do not copy case captions or procedural metadata.\n"
		"Write complete sentences only. Do not copy precedent text verbatim.\n"
		"Be concise and structured with these sections:\n"
		"1) Weaknesses in Prosecution Case\n"
		"2) Statutory Counter-Interpretation\n"
		"3) Precedent Distinguishing (citation + factual mismatch)\n"
		"4) Mitigating Factors\n"
		"5) Relief Sought\n\n"
		f"Facts:\n{facts.strip()}\n\n"
		f"Relevant Statutes:\n{statutes.strip()}\n\n"
		f"Relevant Precedents:\n{precedents.strip()}\n\n"
		f"Prosecution Argument:\n{prosecution_output.strip()}"
	)


def get_judge_prompt(
	facts: str,
	prosecution_output: str,
	defense_output: str,
	statutes: str,
	precedents: str,
) -> str:
	return (
		"You are the judge. Compare prosecution and defense arguments and give a reasoned verdict.\n"
		"Assess both statutory fit and precedent similarity before concluding.\n"
		"Do not output copied precedent text; refer to citations in bracket form only like [P1].\n"
		"Do not paste raw precedent passages; summarize legal relevance in your own words.\n"
		"Return in this structure:\n"
		"Findings:\n- ...\n"
		"Reasoning:\n- ...\n"
		"Accepted/Rejected Citations:\n- ...\n"
		"Final Verdict:\n- ...\n"
		"Suggested Sentence/Relief:\n- ...\n\n"
		f"Facts:\n{facts.strip()}\n\n"
		f"Relevant Statutes:\n{statutes.strip()}\n\n"
		f"Relevant Precedents:\n{precedents.strip()}\n\n"
		f"Prosecution Argument:\n{prosecution_output.strip()}\n\n"
		f"Defense Argument:\n{defense_output.strip()}"
	)
