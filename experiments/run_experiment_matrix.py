from __future__ import annotations

import argparse
import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prosecution_pipeline as pp  # type: ignore[import-not-found]
from experiments.prompts import build_single_llm_no_rag_prompt, build_single_llm_with_rag_prompt
from experiments.rule_baseline import build_rule_outputs


DEFAULT_VARIANTS = [
    "full_system",
    "no_precedent",
    "no_ipc_retrieval",
    "no_defence_agent",
    "single_llm_no_rag",
    "single_llm_with_rag",
    "rule_only_ipc_mapper",
]

IPC_PATTERN = re.compile(r"\b(?:IPC|Section|Sections)?\s*(\d{1,3}[A-Z]?)\b", flags=re.IGNORECASE)
YEARS_PATTERN = re.compile(r"\bimprisonment\s+for\s+(\d{1,2})\s+years?\b", flags=re.IGNORECASE)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _extract_ipc(text: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for m in IPC_PATTERN.finditer(text or ""):
        sec = m.group(1).upper().strip()
        if sec and sec not in seen:
            seen.add(sec)
            out.append(sec)
    return out


def _case_identifier(row: Dict[str, Any], candidate: Dict[str, Any]) -> str:
    case_id = str(candidate.get("case_id", "")).strip()
    if case_id:
        return case_id.upper()
    case_name = str(candidate.get("case_name", "")).strip()
    if case_name:
        return case_name.upper().replace(" ", "_")
    return str(row.get("case_id", "unknown")).upper()


def _normalize_precedents(precedents: Iterable[Dict[str, Any]], row: Dict[str, Any]) -> List[Dict[str, str]]:
    norm: List[Dict[str, str]] = []
    for cand in precedents:
        norm.append(
            {
                "case_id": _case_identifier(row, cand),
                "case_name": str(cand.get("case_name", "Unknown Case")),
                "judgment": str(cand.get("judgment", "unknown")),
            }
        )
    return norm


def _normalize_statutes(statutes: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    norm: List[Dict[str, str]] = []
    for cand in statutes:
        sec = str(cand.get("section", "")).strip().upper()
        if not sec:
            continue
        norm.append(
            {
                "section": sec,
                "title": str(cand.get("title", "")),
            }
        )
    return norm


def _retrieval_snapshot(fact_text: str, use_precedent: bool, use_ipc: bool) -> Dict[str, Any]:
    retrieved_precedents: List[Dict[str, Any]] = []
    retrieved_statutes: List[Dict[str, Any]] = []

    if use_precedent:
        try:
            pp.initialize_precedent_retriever()
            retrieved_precedents = pp.retrieve_precedents(query=fact_text, top_k=8)
        except Exception:
            retrieved_precedents = []

    if use_ipc:
        try:
            pp.initialize_statute_retriever()
            retrieved_statutes = pp.retrieve_statutes(query=fact_text, top_k=8)
        except Exception:
            retrieved_statutes = []

    case_ids = []
    for p in retrieved_precedents:
        cid = str(p.get("case_id", "")).strip()
        cname = str(p.get("case_name", "")).strip().upper().replace(" ", "_")
        case_ids.append((cid or cname))

    ipc_sections = []
    for s in retrieved_statutes:
        sec = str(s.get("section", "")).strip().upper()
        if sec:
            ipc_sections.append(sec)

    seen_case = set()
    uniq_case_ids = []
    for cid in case_ids:
        if cid and cid not in seen_case:
            seen_case.add(cid)
            uniq_case_ids.append(cid)

    seen_ipc = set()
    uniq_ipc = []
    for sec in ipc_sections:
        if sec not in seen_ipc:
            seen_ipc.add(sec)
            uniq_ipc.append(sec)

    return {
        "retrieved_case_ids": uniq_case_ids,
        "retrieved_ipc_sections": uniq_ipc,
        "avg_semantic_similarity": None,
    }


@contextmanager
def _patched_retrieval(disable_precedent: bool = False, disable_ipc: bool = False):
    orig_precedent = pp.retrieve_precedents
    orig_statute = pp.retrieve_statutes
    try:
        if disable_precedent:
            pp.retrieve_precedents = lambda *args, **kwargs: []  # type: ignore[assignment]
        if disable_ipc:
            pp.retrieve_statutes = lambda *args, **kwargs: []  # type: ignore[assignment]
        yield
    finally:
        pp.retrieve_precedents = orig_precedent  # type: ignore[assignment]
        pp.retrieve_statutes = orig_statute  # type: ignore[assignment]


def _single_llm_no_rag(fact_text: str) -> Dict[str, str]:
    prompt = build_single_llm_no_rag_prompt(fact_text)
    judge = pp.generate_response(prompt).strip()
    return {"prosecution": "", "defence": "", "judge": judge}


def _single_llm_with_rag(fact_text: str) -> Dict[str, str]:
    pp.initialize_precedent_retriever()
    pp.initialize_statute_retriever()
    precedents = _normalize_precedents(pp.retrieve_precedents(query=fact_text, top_k=5), {"case_id": "unknown"})
    statutes = _normalize_statutes(pp.retrieve_statutes(query=fact_text, top_k=5))
    prompt = build_single_llm_with_rag_prompt(fact_text=fact_text, precedents=precedents, statutes=statutes)
    judge = pp.generate_response(prompt).strip()
    return {"prosecution": "", "defence": "", "judge": judge}


def _no_defence_agent(fact_text: str) -> Dict[str, str]:
    prosecution = pp.prosecution_arguments(fact_text)
    defence = (
        "The defence disputes intention and requests consideration of lesser offence principles and a proportionate reduced sentence."
    )
    judge = pp.judge_arguments(prosecutor_output=prosecution, defence_output=defence, user_case=fact_text)
    return {"prosecution": prosecution, "defence": defence, "judge": judge}


def _full_system(fact_text: str) -> Dict[str, str]:
    prosecution = pp.prosecution_arguments(fact_text)
    defence = pp.defence_arguments(prosecutor_output=prosecution, user_case=fact_text)
    judge = pp.judge_arguments(prosecutor_output=prosecution, defence_output=defence, user_case=fact_text)
    return {"prosecution": prosecution, "defence": defence, "judge": judge}


def _run_three_agent_variant(fact_text: str) -> Tuple[Dict[str, str], Dict[str, float]]:
    t0 = time.perf_counter()
    prosecution = pp.prosecution_arguments(fact_text)
    t1 = time.perf_counter()
    defence = pp.defence_arguments(prosecutor_output=prosecution, user_case=fact_text)
    t2 = time.perf_counter()
    judge = pp.judge_arguments(prosecutor_output=prosecution, defence_output=defence, user_case=fact_text)
    t3 = time.perf_counter()
    return (
        {"prosecution": prosecution, "defence": defence, "judge": judge},
        {
            "total_sec": t3 - t0,
            "prosecution_sec": t1 - t0,
            "defence_sec": t2 - t1,
            "judge_sec": t3 - t2,
        },
    )


def _no_precedent(fact_text: str) -> Dict[str, str]:
    with _patched_retrieval(disable_precedent=True, disable_ipc=False):
        return _full_system(fact_text)


def _no_ipc_retrieval(fact_text: str) -> Dict[str, str]:
    with _patched_retrieval(disable_precedent=False, disable_ipc=True):
        return _full_system(fact_text)


def _rule_only(fact_text: str) -> Dict[str, str]:
    return build_rule_outputs(fact_text)


def _judge_label(judge_text: str) -> str:
    ipc = _extract_ipc(judge_text)
    sentence = "unknown"
    lower = (judge_text or "").lower()
    if "death penalty" in lower:
        sentence = "death"
    elif "life imprisonment" in lower or "imprisonment for life" in lower:
        sentence = "life"
    else:
        m = YEARS_PATTERN.search(lower)
        if m:
            sentence = f"{m.group(1)}y"
    return f"{'-'.join(ipc) if ipc else 'NOIPC'}_{sentence}"


def _run_variant(variant: str, fact_text: str) -> Tuple[Dict[str, str], Dict[str, float]]:
    start = time.perf_counter()

    if variant == "full_system":
        return _run_three_agent_variant(fact_text)

    if variant == "no_precedent":
        with _patched_retrieval(disable_precedent=True, disable_ipc=False):
            return _run_three_agent_variant(fact_text)
    elif variant == "no_ipc_retrieval":
        with _patched_retrieval(disable_precedent=False, disable_ipc=True):
            return _run_three_agent_variant(fact_text)
    elif variant == "no_defence_agent":
        t0 = time.perf_counter()
        outputs = _no_defence_agent(fact_text)
        t1 = time.perf_counter()
        total = t1 - t0
        return outputs, {"total_sec": total, "prosecution_sec": None, "defence_sec": None, "judge_sec": None}
    elif variant == "single_llm_no_rag":
        outputs = _single_llm_no_rag(fact_text)
    elif variant == "single_llm_with_rag":
        outputs = _single_llm_with_rag(fact_text)
    elif variant == "rule_only_ipc_mapper":
        outputs = _rule_only(fact_text)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    total = time.perf_counter() - start
    return outputs, {"total_sec": total, "prosecution_sec": None, "defence_sec": None, "judge_sec": None}


def _variant_retrieval_flags(variant: str) -> Tuple[bool, bool]:
    if variant == "no_precedent":
        return False, True
    if variant == "no_ipc_retrieval":
        return True, False
    if variant == "single_llm_no_rag":
        return False, False
    if variant == "rule_only_ipc_mapper":
        return False, False
    return True, True


def run_matrix(
    benchmark_rows: Sequence[Dict[str, Any]],
    variants: Sequence[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fp:
        for row in benchmark_rows:
            case_id = str(row.get("case_id", "unknown"))
            fact_text = str(row.get("fact_text", "")).strip()
            if not fact_text:
                continue

            for variant in variants:
                outputs, latency = _run_variant(variant, fact_text)
                use_prec, use_ipc = _variant_retrieval_flags(variant)
                retrieval = _retrieval_snapshot(fact_text, use_precedent=use_prec, use_ipc=use_ipc)

                result_row = {
                    "case_id": case_id,
                    "variant": variant,
                    "paraphrase_group_id": row.get("paraphrase_group_id"),
                    "expected": row.get("expected", {}),
                    "retrieval": retrieval,
                    "outputs": outputs,
                    "evaluation": {
                        "judge_label": _judge_label(outputs.get("judge", "")),
                        "cited_case_ids": [],
                    },
                    "latency": latency,
                }
                fp.write(json.dumps(result_row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comparison matrix for Court_RAG variants.")
    parser.add_argument(
        "--benchmark",
        default="experiments/data/sample_benchmark.jsonl",
        help="JSONL benchmark file with case facts and expected labels.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants to run.",
    )
    parser.add_argument(
        "--output",
        default="experiments/reports/experiment_runs.jsonl",
        help="Output JSONL file for metrics evaluator.",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    output_path = Path(args.output)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows = _load_jsonl(benchmark_path)
    run_matrix(rows, variants=variants, output_path=output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
