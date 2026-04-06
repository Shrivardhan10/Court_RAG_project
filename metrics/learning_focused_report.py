from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_learning_focused(summary: Dict[str, Any]) -> Dict[str, Any]:
    variants = summary.get("variant_metrics", {})
    full = variants.get("full_system", {}).get("generation_valid_only", {})
    no_def = variants.get("no_defence_agent", {}).get("generation_valid_only", {})
    rule = variants.get("rule_only_ipc_mapper", {}).get("generation_valid_only", {})

    end_valid = summary.get("end_to_end_valid_only", {})
    scope = summary.get("scope", {})

    def _num(d: Dict[str, Any], k: str) -> float:
        val = d.get(k)
        return float(val) if isinstance(val, (int, float)) else 0.0

    focused = {
        "scope": {
            "total_rows": scope.get("total_rows", 0),
            "valid_rows": scope.get("valid_rows", 0),
            "excluded_failed_rows": scope.get("excluded_failed_rows", 0),
        },
        "learning_core_metrics": {
            "prosecutor_quality_score": _num(full, "prosecutor_quality_score"),
            "defence_quality_score": _num(full, "defence_quality_score"),
            "judge_quality_score": _num(full, "judge_quality_score"),
            "defence_ipc_correctness": _num(full, "defence_ipc_correctness"),
            "judge_ipc_correctness": _num(full, "judge_ipc_correctness"),
        },
        "supporting_reasoning_metrics": {
            "judgment_ipc_recall_valid_only": _num(end_valid, "judgment_ipc_recall"),
            "judgment_ipc_jaccard_valid_only": _num(end_valid, "judgment_ipc_jaccard"),
            "consistency_score_valid_only": _num(end_valid, "consistency_score"),
        },
        "multi_agent_vs_baselines": {
            "full_vs_no_defence": {
                "defence_quality_delta": _num(full, "defence_quality_score") - _num(no_def, "defence_quality_score"),
                "defence_ipc_correctness_delta": _num(full, "defence_ipc_correctness") - _num(no_def, "defence_ipc_correctness"),
            },
            "full_vs_rule_only": {
                "prosecutor_quality_delta": _num(full, "prosecutor_quality_score") - _num(rule, "prosecutor_quality_score"),
                "judge_quality_delta": _num(full, "judge_quality_score") - _num(rule, "judge_quality_score"),
                "defence_quality_delta": _num(full, "defence_quality_score") - _num(rule, "defence_quality_score"),
            },
        },
        "note": "Outcome-exact metrics are intentionally excluded from this focused report because this view is for courtroom argument-learning quality.",
    }
    return focused


def render_markdown(focused: Dict[str, Any]) -> str:
    core = focused["learning_core_metrics"]
    support = focused["supporting_reasoning_metrics"]
    f_nd = focused["multi_agent_vs_baselines"]["full_vs_no_defence"]
    f_rule = focused["multi_agent_vs_baselines"]["full_vs_rule_only"]
    scope = focused["scope"]

    lines = [
        "# Learning-Focused Metrics Report",
        "",
        "## Scope",
        f"- total_rows: {scope['total_rows']}",
        f"- valid_rows: {scope['valid_rows']}",
        f"- excluded_failed_rows: {scope['excluded_failed_rows']}",
        "",
        "## Learning Core Metrics (Full System)",
        f"- prosecutor_quality_score: {core['prosecutor_quality_score']:.3f}",
        f"- defence_quality_score: {core['defence_quality_score']:.3f}",
        f"- judge_quality_score: {core['judge_quality_score']:.3f}",
        f"- defence_ipc_correctness: {core['defence_ipc_correctness']:.3f}",
        f"- judge_ipc_correctness: {core['judge_ipc_correctness']:.3f}",
        "",
        "## Supporting Reasoning Metrics",
        f"- judgment_ipc_recall_valid_only: {support['judgment_ipc_recall_valid_only']:.3f}",
        f"- judgment_ipc_jaccard_valid_only: {support['judgment_ipc_jaccard_valid_only']:.3f}",
        f"- consistency_score_valid_only: {support['consistency_score_valid_only']:.3f}",
        "",
        "## Multi-Agent Deltas",
        f"- full_vs_no_defence.defence_quality_delta: {f_nd['defence_quality_delta']:.3f}",
        f"- full_vs_no_defence.defence_ipc_correctness_delta: {f_nd['defence_ipc_correctness_delta']:.3f}",
        f"- full_vs_rule_only.prosecutor_quality_delta: {f_rule['prosecutor_quality_delta']:.3f}",
        f"- full_vs_rule_only.judge_quality_delta: {f_rule['judge_quality_delta']:.3f}",
        f"- full_vs_rule_only.defence_quality_delta: {f_rule['defence_quality_delta']:.3f}",
        "",
        "## Note",
        focused["note"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate learning-focused metrics summary from full summary.json")
    parser.add_argument("--summary", required=True, help="Path to existing summary.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for focused report")
    args = parser.parse_args()

    summary = _read_json(Path(args.summary))
    focused = build_learning_focused(summary)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary_learning_focused.json").write_text(json.dumps(focused, indent=2), encoding="utf-8")
    (out / "report_learning_focused.md").write_text(render_markdown(focused), encoding="utf-8")

    print(f"Saved: {out / 'summary_learning_focused.json'}")
    print(f"Saved: {out / 'report_learning_focused.md'}")


if __name__ == "__main__":
    main()
