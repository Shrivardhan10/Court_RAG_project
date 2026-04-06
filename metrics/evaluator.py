from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


IPC_PATTERN = re.compile(
    r"\b(?:IPC\s*|I\.P\.C\.\s*|Section(?:s)?\s+|u/s\s*|under\s+section(?:s)?\s+)(\d{1,3}[A-Z]?)\b",
    flags=re.IGNORECASE,
)
YEARS_PATTERN = re.compile(
    r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+years?\b",
    flags=re.IGNORECASE,
)

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


@dataclass
class RetrievalMetrics:
    ipc_top_k_accuracy: float
    ipc_precision: float
    ipc_recall: float
    ipc_f1: float
    precedent_recall_at_k: float
    precedent_mrr: float
    precedent_ndcg_at_k: float
    avg_semantic_similarity: Optional[float]


@dataclass
class GenerationMetrics:
    prosecutor_ipc_correctness: float
    defence_ipc_correctness: float
    judge_ipc_correctness: float
    prosecutor_quality_score: float
    defence_quality_score: float
    judge_quality_score: float
    prosecutor_reasoning_score: Optional[float]
    defence_reasoning_score: Optional[float]
    judge_reasoning_score: Optional[float]
    hallucination_rate: Optional[float]


@dataclass
class EndToEndMetrics:
    judgment_accuracy: float
    judgment_ipc_recall: float
    judgment_ipc_precision: float
    judgment_ipc_jaccard: float
    sentencing_accuracy: float
    consistency_score: float
    charge_correction_rate: float
    robustness_stability: Optional[float]
    avg_total_latency_sec: Optional[float]
    avg_prosecution_latency_sec: Optional[float]
    avg_defence_latency_sec: Optional[float]
    avg_judge_latency_sec: Optional[float]


def _to_set(values: Any) -> Set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        return {values.strip().upper()} if values.strip() else set()
    return {str(v).strip().upper() for v in values if str(v).strip()}


def _extract_ipc(text: str) -> Set[str]:
    if not text:
        return set()
    return {m.group(1).strip().upper() for m in IPC_PATTERN.finditer(text) if m.group(1).strip()}


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return _safe_div(len(a & b), len(a | b))


def _recall_at_k(relevant: Set[str], ranked: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = {str(x).strip().upper() for x in ranked[:k]}
    return _safe_div(len(relevant & top), len(relevant))


def _mrr(relevant: Set[str], ranked: Sequence[str]) -> float:
    if not relevant:
        return 0.0
    for i, item in enumerate(ranked, start=1):
        if str(item).strip().upper() in relevant:
            return 1.0 / i
    return 0.0


def _dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def _ndcg_at_k(relevant: Set[str], ranked: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    ranked_top = [str(x).strip().upper() for x in ranked[:k]]
    rels = [1 if x in relevant else 0 for x in ranked_top]
    ideal = sorted(rels, reverse=True)
    return _safe_div(_dcg(rels), _dcg(ideal)) if any(ideal) else 0.0


def _parse_sentence_profile(text: str) -> Dict[str, Any]:
    lower = (text or "").lower()
    if "death penalty" in lower or re.search(r"\bdeath\b", lower):
        return {"type": "death", "years": None}
    if "life imprisonment" in lower or "imprisonment for life" in lower:
        return {"type": "life", "years": None}

    years: List[int] = []
    for token in YEARS_PATTERN.findall(lower):
        tok = token.lower()
        if tok.isdigit():
            years.append(int(tok))
        elif tok in NUMBER_WORDS:
            years.append(NUMBER_WORDS[tok])
    if years:
        return {"type": "term", "years": max(years)}
    return {"type": "unknown", "years": None}


def _sentence_count(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    parts = re.split(r"(?<=[.!?])\s+", text)
    return len([p for p in parts if p.strip()])


def _has_compensation_amount(text: str) -> bool:
    return bool(re.search(r"(₹\s*\d[\d,]*|\b(?:INR|Rs\.?|Rupees)\s*\d[\d,]*)", text or "", flags=re.IGNORECASE))


def _quality_score(text: str) -> float:
    t = text or ""
    if not t.strip():
        return 0.0
    checks = [
        1.0 if _extract_ipc(t) else 0.0,
        1.0 if _parse_sentence_profile(t).get("type") != "unknown" else 0.0,
        1.0 if _has_compensation_amount(t) else 0.0,
        1.0 if "\n\n" not in t else 0.0,
        1.0 if _sentence_count(t) >= 4 else 0.0,
    ]
    return sum(checks) / float(len(checks))


def _mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return mean(valid) if valid else None


def _ipc_precision_recall(predicted: Set[str], expected: Set[str]) -> tuple[float, float, float]:
    tp = len(predicted & expected)
    precision = _safe_div(tp, len(predicted)) if predicted else (1.0 if not expected else 0.0)
    recall = _safe_div(tp, len(expected)) if expected else (1.0 if not predicted else 0.0)
    jaccard = _jaccard(predicted, expected)
    return precision, recall, jaccard


def _is_failed_row(row: Dict[str, Any]) -> bool:
    outputs = row.get("outputs", {}) or {}
    text = " ".join(str(outputs.get(key, "")) for key in ["prosecution", "defence", "judge"]).lower()
    fail_markers = [
        "[llm] generation failed",
        "quota exceeded",
        "429",
        "rate-limit",
    ]
    return any(marker in text for marker in fail_markers)


def _filter_valid_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if not _is_failed_row(row)]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def compute_retrieval_metrics(rows: Sequence[Dict[str, Any]], k: int) -> RetrievalMetrics:
    ipc_hits = []
    ipc_precisions = []
    ipc_recalls = []
    ipc_f1s = []

    case_recalls = []
    case_mrr = []
    case_ndcg = []
    sem_sims = []

    for row in rows:
        gt_ipc = _to_set(row.get("expected", {}).get("ipc_sections"))
        ret_ipc = _to_set(row.get("retrieval", {}).get("retrieved_ipc_sections"))

        if gt_ipc:
            top_k_list = list(ret_ipc)[:k]
            ipc_hits.append(1.0 if gt_ipc & set(top_k_list) else 0.0)

            tp = len(gt_ipc & ret_ipc)
            precision = _safe_div(tp, len(ret_ipc))
            recall = _safe_div(tp, len(gt_ipc))
            f1 = _safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0
            ipc_precisions.append(precision)
            ipc_recalls.append(recall)
            ipc_f1s.append(f1)

        relevant_cases = _to_set(row.get("expected", {}).get("relevant_case_ids"))
        ranked_cases = [str(x).strip().upper() for x in row.get("retrieval", {}).get("retrieved_case_ids", [])]

        if relevant_cases:
            case_recalls.append(_recall_at_k(relevant_cases, ranked_cases, k))
            case_mrr.append(_mrr(relevant_cases, ranked_cases))
            case_ndcg.append(_ndcg_at_k(relevant_cases, ranked_cases, k))

        sim = row.get("retrieval", {}).get("avg_semantic_similarity")
        if isinstance(sim, (int, float)):
            sem_sims.append(float(sim))

    return RetrievalMetrics(
        ipc_top_k_accuracy=mean(ipc_hits) if ipc_hits else 0.0,
        ipc_precision=mean(ipc_precisions) if ipc_precisions else 0.0,
        ipc_recall=mean(ipc_recalls) if ipc_recalls else 0.0,
        ipc_f1=mean(ipc_f1s) if ipc_f1s else 0.0,
        precedent_recall_at_k=mean(case_recalls) if case_recalls else 0.0,
        precedent_mrr=mean(case_mrr) if case_mrr else 0.0,
        precedent_ndcg_at_k=mean(case_ndcg) if case_ndcg else 0.0,
        avg_semantic_similarity=mean(sem_sims) if sem_sims else None,
    )


def compute_generation_metrics(rows: Sequence[Dict[str, Any]]) -> GenerationMetrics:
    pro_scores = []
    def_scores = []
    judge_scores = []

    pro_reason = []
    def_reason = []
    judge_reason = []

    hallucinations = []
    pro_quality = []
    def_quality = []
    judge_quality = []

    for row in rows:
        expected_ipc = _to_set(row.get("expected", {}).get("ipc_sections"))
        outputs = row.get("outputs", {})

        pro_text = str(outputs.get("prosecution", ""))
        def_text = str(outputs.get("defence", ""))
        judge_text = str(outputs.get("judge", ""))

        pro_ipc = _extract_ipc(pro_text)
        def_ipc = _extract_ipc(def_text)
        judge_ipc = _extract_ipc(judge_text)

        pro_quality.append(_quality_score(pro_text))
        def_quality.append(_quality_score(def_text))
        judge_quality.append(_quality_score(judge_text))

        if expected_ipc:
            pro_scores.append(_safe_div(len(pro_ipc & expected_ipc), len(expected_ipc)))
            def_scores.append(_safe_div(len(def_ipc & expected_ipc), len(expected_ipc)))
            judge_scores.append(_safe_div(len(judge_ipc & expected_ipc), len(expected_ipc)))

        pr = row.get("evaluation", {}).get("prosecutor_reasoning_score")
        dr = row.get("evaluation", {}).get("defence_reasoning_score")
        jr = row.get("evaluation", {}).get("judge_reasoning_score")
        if isinstance(pr, (int, float)):
            pro_reason.append(float(pr))
        if isinstance(dr, (int, float)):
            def_reason.append(float(dr))
        if isinstance(jr, (int, float)):
            judge_reason.append(float(jr))

        cited = _to_set(row.get("evaluation", {}).get("cited_case_ids"))
        retrieved = _to_set(row.get("retrieval", {}).get("retrieved_case_ids"))
        if cited:
            unsupported = len(cited - retrieved)
            hallucinations.append(_safe_div(unsupported, len(cited)))

    return GenerationMetrics(
        prosecutor_ipc_correctness=mean(pro_scores) if pro_scores else 0.0,
        defence_ipc_correctness=mean(def_scores) if def_scores else 0.0,
        judge_ipc_correctness=mean(judge_scores) if judge_scores else 0.0,
        prosecutor_quality_score=mean(pro_quality) if pro_quality else 0.0,
        defence_quality_score=mean(def_quality) if def_quality else 0.0,
        judge_quality_score=mean(judge_quality) if judge_quality else 0.0,
        prosecutor_reasoning_score=mean(pro_reason) if pro_reason else None,
        defence_reasoning_score=mean(def_reason) if def_reason else None,
        judge_reasoning_score=mean(judge_reason) if judge_reason else None,
        hallucination_rate=mean(hallucinations) if hallucinations else None,
    )


def compute_end_to_end_metrics(rows: Sequence[Dict[str, Any]]) -> EndToEndMetrics:
    judgment_acc = []
    judgment_recall = []
    judgment_precision = []
    judgment_jaccard = []
    sentencing_acc = []
    consistency = []
    corrections = []

    total_lat = []
    pro_lat = []
    def_lat = []
    judge_lat = []

    robustness_groups: Dict[str, List[str]] = {}

    for row in rows:
        expected_ipc = _to_set(row.get("expected", {}).get("ipc_sections"))
        outputs = row.get("outputs", {})
        pro_text = str(outputs.get("prosecution", ""))
        judge_text = str(outputs.get("judge", ""))

        pro_ipc = _extract_ipc(pro_text)
        judge_ipc = _extract_ipc(judge_text)

        if expected_ipc:
            judgment_acc.append(1.0 if judge_ipc == expected_ipc else 0.0)
            precision, recall, jaccard = _ipc_precision_recall(judge_ipc, expected_ipc)
            judgment_precision.append(precision)
            judgment_recall.append(recall)
            judgment_jaccard.append(jaccard)

            prosecution_correct = pro_ipc == expected_ipc
            judge_correct = judge_ipc == expected_ipc
            if not prosecution_correct:
                corrections.append(1.0 if judge_correct else 0.0)

        consistency.append(_jaccard(pro_ipc, judge_ipc))

        expected_sentence = row.get("expected", {}).get("sentence", {})
        if expected_sentence:
            pred_profile = _parse_sentence_profile(judge_text)
            exp_type = str(expected_sentence.get("type", "")).lower()
            exp_years = expected_sentence.get("years")
            ok = pred_profile["type"] == exp_type
            if ok and exp_type == "term" and isinstance(exp_years, int):
                ok = pred_profile.get("years") == exp_years
            sentencing_acc.append(1.0 if ok else 0.0)

        lat = row.get("latency", {})
        if isinstance(lat.get("total_sec"), (int, float)):
            total_lat.append(float(lat["total_sec"]))
        if isinstance(lat.get("prosecution_sec"), (int, float)):
            pro_lat.append(float(lat["prosecution_sec"]))
        if isinstance(lat.get("defence_sec"), (int, float)):
            def_lat.append(float(lat["defence_sec"]))
        if isinstance(lat.get("judge_sec"), (int, float)):
            judge_lat.append(float(lat["judge_sec"]))

        group_id = row.get("paraphrase_group_id")
        final_label = row.get("evaluation", {}).get("judge_label")
        if group_id and final_label:
            robustness_groups.setdefault(str(group_id), []).append(str(final_label))

    stability_values = []
    for labels in robustness_groups.values():
        if not labels:
            continue
        dominant = max(set(labels), key=labels.count)
        stability_values.append(_safe_div(labels.count(dominant), len(labels)))

    return EndToEndMetrics(
        judgment_accuracy=mean(judgment_acc) if judgment_acc else 0.0,
        judgment_ipc_recall=mean(judgment_recall) if judgment_recall else 0.0,
        judgment_ipc_precision=mean(judgment_precision) if judgment_precision else 0.0,
        judgment_ipc_jaccard=mean(judgment_jaccard) if judgment_jaccard else 0.0,
        sentencing_accuracy=mean(sentencing_acc) if sentencing_acc else 0.0,
        consistency_score=mean(consistency) if consistency else 0.0,
        charge_correction_rate=mean(corrections) if corrections else 0.0,
        robustness_stability=mean(stability_values) if stability_values else None,
        avg_total_latency_sec=mean(total_lat) if total_lat else None,
        avg_prosecution_latency_sec=mean(pro_lat) if pro_lat else None,
        avg_defence_latency_sec=mean(def_lat) if def_lat else None,
        avg_judge_latency_sec=mean(judge_lat) if judge_lat else None,
    )


def compute_ablation(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, Dict[str, int]] = {}
    for row in rows:
        variant = str(row.get("variant", "full_system"))
        b = buckets.setdefault(variant, {"count": 0, "correct": 0})
        b["count"] += 1

        expected_ipc = _to_set(row.get("expected", {}).get("ipc_sections"))
        judge_ipc = _extract_ipc(str(row.get("outputs", {}).get("judge", "")))
        if expected_ipc and judge_ipc == expected_ipc:
            b["correct"] += 1

    result: Dict[str, Dict[str, float]] = {}
    for variant, agg in buckets.items():
        count = agg["count"]
        result[variant] = {
            "n": float(count),
            "judgment_accuracy": _safe_div(agg["correct"], count),
        }
    return result


def compute_variant_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        variant = str(row.get("variant", "full_system"))
        by_variant.setdefault(variant, []).append(row)

    result: Dict[str, Dict[str, Any]] = {}
    for variant, variant_rows in sorted(by_variant.items()):
        valid_rows = _filter_valid_rows(variant_rows)
        result[variant] = {
            "n": len(variant_rows),
            "n_valid": len(valid_rows),
            "end_to_end_all": to_dict(compute_end_to_end_metrics(variant_rows)),
            "end_to_end_valid_only": to_dict(compute_end_to_end_metrics(valid_rows)),
            "generation_valid_only": to_dict(compute_generation_metrics(valid_rows)),
        }
    return result


def compute_multi_agent_comparisons(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    valid_rows = _filter_valid_rows(rows)
    index: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in valid_rows:
        case_id = str(row.get("case_id", ""))
        variant = str(row.get("variant", ""))
        if case_id and variant:
            index[(case_id, variant)] = row

    def _delta(base_variant: str, alt_variant: str) -> Dict[str, Optional[float]]:
        acc_deltas: List[float] = []
        recall_deltas: List[float] = []
        jaccard_deltas: List[float] = []

        case_ids = sorted({case for case, _ in index.keys()})
        for case_id in case_ids:
            base = index.get((case_id, base_variant))
            alt = index.get((case_id, alt_variant))
            if not base or not alt:
                continue

            exp = _to_set(base.get("expected", {}).get("ipc_sections"))
            base_judge = _extract_ipc(str(base.get("outputs", {}).get("judge", "")))
            alt_judge = _extract_ipc(str(alt.get("outputs", {}).get("judge", "")))
            if not exp:
                continue

            base_prec, base_rec, base_jac = _ipc_precision_recall(base_judge, exp)
            alt_prec, alt_rec, alt_jac = _ipc_precision_recall(alt_judge, exp)
            _ = (base_prec, alt_prec)
            acc_deltas.append((1.0 if base_judge == exp else 0.0) - (1.0 if alt_judge == exp else 0.0))
            recall_deltas.append(base_rec - alt_rec)
            jaccard_deltas.append(base_jac - alt_jac)

        return {
            "judgment_accuracy_delta": mean(acc_deltas) if acc_deltas else None,
            "judge_ipc_recall_delta": mean(recall_deltas) if recall_deltas else None,
            "judge_ipc_jaccard_delta": mean(jaccard_deltas) if jaccard_deltas else None,
            "paired_cases": float(len(acc_deltas)),
        }

    return {
        "full_system_vs_no_defence_agent": _delta("full_system", "no_defence_agent"),
        "full_system_vs_single_llm_with_rag": _delta("full_system", "single_llm_with_rag"),
        "full_system_vs_single_llm_no_rag": _delta("full_system", "single_llm_no_rag"),
    }


def to_dict(obj: Any) -> Dict[str, Any]:
    return {k: v for k, v in obj.__dict__.items()}


def render_markdown_report(
    retrieval: RetrievalMetrics,
    generation: GenerationMetrics,
    end_to_end: EndToEndMetrics,
    ablation: Dict[str, Dict[str, float]],
    valid_only_end_to_end: EndToEndMetrics,
    variant_metrics: Dict[str, Dict[str, Any]],
    comparisons: Dict[str, Dict[str, Optional[float]]],
    total_rows: int,
    valid_rows: int,
    input_path: Path,
) -> str:
    lines = [
        "# Metrics Report",
        "",
        f"Input file: {input_path}",
        "",
        "## Retrieval",
        f"- IPC Top-k Accuracy: {retrieval.ipc_top_k_accuracy:.3f}",
        f"- IPC Precision: {retrieval.ipc_precision:.3f}",
        f"- IPC Recall: {retrieval.ipc_recall:.3f}",
        f"- IPC F1: {retrieval.ipc_f1:.3f}",
        f"- Precedent Recall@k: {retrieval.precedent_recall_at_k:.3f}",
        f"- Precedent MRR: {retrieval.precedent_mrr:.3f}",
        f"- Precedent nDCG@k: {retrieval.precedent_ndcg_at_k:.3f}",
        f"- Avg Semantic Similarity: {retrieval.avg_semantic_similarity if retrieval.avg_semantic_similarity is not None else 'N/A'}",
        "",
        "## Generation",
        f"- Prosecutor IPC Correctness: {generation.prosecutor_ipc_correctness:.3f}",
        f"- Defence IPC Correctness: {generation.defence_ipc_correctness:.3f}",
        f"- Judge IPC Correctness: {generation.judge_ipc_correctness:.3f}",
        f"- Prosecutor Quality Score: {generation.prosecutor_quality_score:.3f}",
        f"- Defence Quality Score: {generation.defence_quality_score:.3f}",
        f"- Judge Quality Score: {generation.judge_quality_score:.3f}",
        f"- Prosecutor Reasoning Score: {generation.prosecutor_reasoning_score if generation.prosecutor_reasoning_score is not None else 'N/A'}",
        f"- Defence Reasoning Score: {generation.defence_reasoning_score if generation.defence_reasoning_score is not None else 'N/A'}",
        f"- Judge Reasoning Score: {generation.judge_reasoning_score if generation.judge_reasoning_score is not None else 'N/A'}",
        f"- Hallucination Rate: {generation.hallucination_rate if generation.hallucination_rate is not None else 'N/A'}",
        "",
        "## End-to-End",
        f"- Judgment Accuracy: {end_to_end.judgment_accuracy:.3f}",
        f"- Judgment IPC Recall: {end_to_end.judgment_ipc_recall:.3f}",
        f"- Judgment IPC Precision: {end_to_end.judgment_ipc_precision:.3f}",
        f"- Judgment IPC Jaccard: {end_to_end.judgment_ipc_jaccard:.3f}",
        f"- Sentencing Accuracy: {end_to_end.sentencing_accuracy:.3f}",
        f"- Consistency Score (Prosecutor vs Judge IPC): {end_to_end.consistency_score:.3f}",
        f"- Charge Correction Rate: {end_to_end.charge_correction_rate:.3f}",
        f"- Robustness Stability: {end_to_end.robustness_stability if end_to_end.robustness_stability is not None else 'N/A'}",
        f"- Avg Total Latency (s): {end_to_end.avg_total_latency_sec if end_to_end.avg_total_latency_sec is not None else 'N/A'}",
        f"- Avg Prosecution Latency (s): {end_to_end.avg_prosecution_latency_sec if end_to_end.avg_prosecution_latency_sec is not None else 'N/A'}",
        f"- Avg Defence Latency (s): {end_to_end.avg_defence_latency_sec if end_to_end.avg_defence_latency_sec is not None else 'N/A'}",
        f"- Avg Judge Latency (s): {end_to_end.avg_judge_latency_sec if end_to_end.avg_judge_latency_sec is not None else 'N/A'}",
        "",
        "## Valid-Only Scope",
        f"- Rows: total={total_rows}, valid={valid_rows}",
        f"- Judgment Accuracy (valid only): {valid_only_end_to_end.judgment_accuracy:.3f}",
        f"- Judgment IPC Recall (valid only): {valid_only_end_to_end.judgment_ipc_recall:.3f}",
        f"- Judgment IPC Jaccard (valid only): {valid_only_end_to_end.judgment_ipc_jaccard:.3f}",
        f"- Sentencing Accuracy (valid only): {valid_only_end_to_end.sentencing_accuracy:.3f}",
        "",
        "## Variant End-to-End (Valid Only)",
        "",
    ]

    for variant, stats in sorted(variant_metrics.items()):
        e2e_valid = stats.get("end_to_end_valid_only", {})
        lines.append(
            f"- {variant}: n={stats.get('n', 0)}, n_valid={stats.get('n_valid', 0)}, "
            f"judgment_accuracy={float(e2e_valid.get('judgment_accuracy', 0.0)):.3f}, "
            f"judge_ipc_recall={float(e2e_valid.get('judgment_ipc_recall', 0.0)):.3f}, "
            f"judge_ipc_jaccard={float(e2e_valid.get('judgment_ipc_jaccard', 0.0)):.3f}"
        )

    lines.extend([
        "",
        "## Full-System Comparative Deltas",
    ])
    for name, vals in sorted(comparisons.items()):
        lines.append(
            f"- {name}: paired_cases={int(vals.get('paired_cases', 0.0) or 0)}, "
            f"judgment_accuracy_delta={vals.get('judgment_accuracy_delta')}, "
            f"judge_ipc_recall_delta={vals.get('judge_ipc_recall_delta')}, "
            f"judge_ipc_jaccard_delta={vals.get('judge_ipc_jaccard_delta')}"
        )

    lines.append("")
    lines.append("## Ablation")

    if not ablation:
        lines.append("- No ablation variants found in input.")
    else:
        for variant, stats in sorted(ablation.items()):
            lines.append(
                f"- {variant}: n={int(stats['n'])}, judgment_accuracy={stats['judgment_accuracy']:.3f}"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Court_RAG multi-agent metrics from JSONL runs.")
    parser.add_argument("--input", required=True, help="Path to JSONL file with evaluation rows.")
    parser.add_argument("--k", type=int, default=5, help="Cutoff k for @k retrieval metrics.")
    parser.add_argument(
        "--output-dir",
        default="metrics/reports",
        help="Directory to write summary.json and report.md",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(input_path)
    valid_rows = _filter_valid_rows(rows)
    retrieval = compute_retrieval_metrics(rows, k=args.k)
    generation = compute_generation_metrics(rows)
    end_to_end = compute_end_to_end_metrics(rows)
    end_to_end_valid_only = compute_end_to_end_metrics(valid_rows)
    ablation = compute_ablation(rows)
    ablation_valid_only = compute_ablation(valid_rows)
    variant_metrics = compute_variant_metrics(rows)
    comparisons = compute_multi_agent_comparisons(rows)

    summary = {
        "retrieval": to_dict(retrieval),
        "generation": to_dict(generation),
        "end_to_end": to_dict(end_to_end),
        "end_to_end_valid_only": to_dict(end_to_end_valid_only),
        "ablation": ablation,
        "ablation_valid_only": ablation_valid_only,
        "variant_metrics": variant_metrics,
        "multi_agent_comparisons": comparisons,
        "scope": {
            "total_rows": len(rows),
            "valid_rows": len(valid_rows),
            "excluded_failed_rows": len(rows) - len(valid_rows),
        },
        "config": {"k": args.k, "input": str(input_path)},
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(
        render_markdown_report(
            retrieval,
            generation,
            end_to_end,
            ablation,
            end_to_end_valid_only,
            variant_metrics,
            comparisons,
            len(rows),
            len(valid_rows),
            input_path,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {output_dir / 'summary.json'}")
    print(f"Saved: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
