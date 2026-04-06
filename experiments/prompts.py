from __future__ import annotations

from typing import Dict, List


def build_single_llm_no_rag_prompt(fact_text: str) -> str:
    return f"""
You are an Indian criminal court judge.
Based only on the provided case facts, write one judicial-style paragraph (8 to 14 sentences).
Requirements:
1) mention applicable IPC section(s),
2) provide a specific sentence (death/life/imprisonment for X years),
3) provide compensation/fine in INR,
4) avoid bullet points and headings.

Case facts:
{fact_text}
""".strip()


def build_single_llm_with_rag_prompt(
    fact_text: str,
    precedents: List[Dict[str, str]],
    statutes: List[Dict[str, str]],
) -> str:
    precedent_text = "; ".join(
        f"{p.get('case_name', 'Unknown')} ({p.get('judgment', 'unknown')})" for p in precedents
    ) or "No precedent available"
    statute_text = "; ".join(
        f"IPC {s.get('section', '')}: {s.get('title', '')}" for s in statutes
    ) or "No statute available"

    return f"""
You are an Indian criminal court judge.
Write one judicial-style paragraph (8 to 14 sentences), using the retrieved legal context.
Requirements:
1) mention final IPC section(s),
2) provide specific sentence (death/life/imprisonment for X years),
3) provide compensation/fine in INR,
4) stay consistent with provided retrieval context,
5) no bullet points or headings.

Case facts:
{fact_text}

Retrieved precedents:
{precedent_text}

Retrieved IPC context:
{statute_text}
""".strip()
