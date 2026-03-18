from agents.defense import run_defense_agent
from agents.judge import run_judge_agent
from agents.prosecution import run_prosecution_agent
from retrieval import precedent_retriever, statute_retriever
from utils.llm import generate_response
from utils.prompts import get_fact_extraction_prompt


INPUT_CASE_TEXT = (
	"The accused struck the victim multiple times with an iron rod during a dispute. "
	"Witnesses stated the attack was intentional and directed at vital body parts. "
	"The victim later died from head injuries caused by the assault."
)


def _format_statutes_for_prompt(statutes: list[dict]) -> str:
	if not statutes:
		return "No statutes retrieved."
	parts = []
	for statute in statutes:
		statute_id = statute.get("filename", "")
		title = statute.get("title", "")
		description = (statute.get("description", "") or "").replace("\n", " ").strip()
		parts.append(f"- {statute_id} | {title}: {description[:300]}")
	return "\n".join(parts)


def _print_statutes(statutes: list[dict]) -> None:
	print("\n========== STATUTES ==========")
	if not statutes:
		print("No statutes retrieved.")
		return

	for statute in statutes:
		description = (statute.get("description", "") or "").replace("\n", " ").strip()
		print(f"\nRank: {statute.get('rank')}")
		print(f"Statute ID: {statute.get('filename', '')}")
		print(f"Title: {statute.get('title', '')}")
		print(f"Description: {description[:200]}")
		print(f"Score: {statute.get('score', 0.0):.4f}")


def _print_precedents(precedents: list[dict]) -> None:
	print("\n========== PRECEDENTS ==========")
	if not precedents:
		print("No precedents retrieved.")
		return

	for rank, precedent in enumerate(precedents, start=1):
		preview = (precedent.get("text_preview", "") or "").replace("\n", " ").strip()
		print(f"\nRank: {rank}")
		print(f"Case ID: {precedent.get('case_id', '')}")
		print(f"Score: {precedent.get('score', 0.0):.4f}")
		print(f"Preview: {preview}")


def run_legal_pipeline(case_text: str) -> None:
	if not case_text or not case_text.strip():
		raise ValueError("Input case text must be a non-empty string.")

	print("[main] Starting legal RAG + agents pipeline (CPU only)...")

	statute_init = statute_retriever.initialize_statute_retriever()
	if statute_init["created"]:
		print("[main] Statute index missing. Built new statute index.")
	else:
		print("[main] Loaded existing statute index.")
	print(f"[main] Statutes loaded: {statute_init['statute_count']}")

	precedent_init = precedent_retriever.initialize_precedent_retriever()
	if precedent_init["created"]:
		print("[main] Precedent index missing. Built new precedent index.")
	else:
		print("[main] Loaded existing precedent index.")
	print(f"[main] Cases loaded: {precedent_init['case_count']}")

	print("[main] Extracting facts using LLM...")
	fact_prompt = get_fact_extraction_prompt(case_text)
	facts = generate_response(fact_prompt)

	print("[main] Retrieving statutes based on extracted facts...")
	statutes = statute_retriever.retrieve_statutes(facts, top_k=3)

	print("[main] Retrieving precedents based on extracted facts...")
	precedents = precedent_retriever.retrieve_precedents(facts, top_k=3)

	statutes_for_prompt = _format_statutes_for_prompt(statutes)

	print("[main] Running prosecution agent...")
	prosecution_output = run_prosecution_agent(facts=facts, statutes=statutes_for_prompt)

	print("[main] Running defense agent...")
	defense_output = run_defense_agent(facts=facts, prosecution_output=prosecution_output)

	print("[main] Running judge agent...")
	judge_output = run_judge_agent(
		facts=facts,
		prosecution_output=prosecution_output,
		defense_output=defense_output,
		statutes=statutes_for_prompt,
	)

	print("\n========== FACTS ==========")
	print(facts)

	_print_statutes(statutes)
	_print_precedents(precedents)

	print("\n========== PROSECUTION ARGUMENT ==========")
	print(prosecution_output)

	print("\n========== DEFENSE ARGUMENT ==========")
	print(defense_output)

	print("\n========== FINAL VERDICT ==========")
	print(judge_output)


if __name__ == "__main__":
	run_legal_pipeline(INPUT_CASE_TEXT)
