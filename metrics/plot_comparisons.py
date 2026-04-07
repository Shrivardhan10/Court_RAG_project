from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np


def _load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_metric(summary: Dict[str, Any], variant: str, group: str, metric: str) -> float:
    block = summary.get("variant_metrics", {}).get(variant, {})
    data = block.get(group, {})
    val = data.get(metric, 0.0)
    return float(val) if isinstance(val, (int, float)) else 0.0


def _bar_plot(path: Path, title: str, labels: List[str], values: List[float], y_label: str) -> None:
    plt.figure(figsize=(11, 5.5))
    bars = plt.bar(labels, values, color=["#2D6A4F" if x == "full_system" else "#6C757D" for x in labels])
    plt.title(title)
    plt.ylabel(y_label)
    plt.ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _delta_plot(path: Path, title: str, labels: List[str], values: List[float]) -> None:
    plt.figure(figsize=(10.5, 5.5))
    colors = ["#1B9AAA" if v >= 0 else "#D1495B" for v in values]
    bars = plt.bar(labels, values, color=colors)
    plt.axhline(0.0, color="#333333", linewidth=1)
    plt.title(title)
    plt.ylabel("Delta (full_system - no_defence_agent)")
    y_pad = max(0.1, max(abs(v) for v in values) * 0.2) if values else 0.1
    plt.ylim(min(values) - y_pad, max(values) + y_pad)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, values):
        offset = 0.01 * (1 if value >= 0 else -1)
        va = "bottom" if value >= 0 else "top"
        plt.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.3f}", ha="center", va=va, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison graphs from metrics summary.json")
    parser.add_argument("--summary", required=True, help="Path to metrics summary.json")
    parser.add_argument("--output-dir", default="experiments/reports/graphs", help="Directory to save PNG graphs")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(summary_path)
    variants = list(summary.get("variant_metrics", {}).keys())

    # Graph 1: Learning quality comparison across all systems
    judge_quality = [
        _variant_metric(summary, v, "generation_valid_only", "judge_quality_score")
        for v in variants
    ]
    _bar_plot(
        output_dir / "all_systems_judge_quality.png",
        "Judge Quality Score: Full System vs Baselines",
        variants,
        judge_quality,
        "Judge quality score",
    )

    # Graph 2: Defence quality comparison across all systems
    defence_quality = [
        _variant_metric(summary, v, "generation_valid_only", "defence_quality_score")
        for v in variants
    ]
    _bar_plot(
        output_dir / "all_systems_defence_quality.png",
        "Defence Quality Score: Full System vs Baselines",
        variants,
        defence_quality,
        "Defence quality score",
    )

    # Graph 3: Judge IPC recall comparison across all systems
    judge_recall = [
        _variant_metric(summary, v, "end_to_end_valid_only", "judgment_ipc_recall")
        for v in variants
    ]
    _bar_plot(
        output_dir / "all_systems_judge_ipc_recall.png",
        "Judge IPC Recall: Full System vs Baselines",
        variants,
        judge_recall,
        "Judge IPC recall",
    )

    # Graph 4: Full system vs no defence agent deltas
    deltas = summary.get("multi_agent_comparisons", {}).get("full_system_vs_no_defence_agent", {})
    delta_labels = [
        "accuracy_delta",
        "ipc_recall_delta",
        "ipc_jaccard_delta",
        "severity_delta",
        "compensation_delta",
    ]
    delta_values = [
        float(deltas.get("judgment_accuracy_delta", 0.0) or 0.0),
        float(deltas.get("judge_ipc_recall_delta", 0.0) or 0.0),
        float(deltas.get("judge_ipc_jaccard_delta", 0.0) or 0.0),
        float(deltas.get("judge_sentence_severity_delta", 0.0) or 0.0),
        float(deltas.get("judge_compensation_delta", 0.0) or 0.0),
    ]
    _delta_plot(
        output_dir / "full_vs_no_defence_deltas.png",
        "Full System vs No Defence Agent: Key Deltas",
        delta_labels,
        delta_values,
    )

    # Graph 5: Full vs no defence severity outcome counts
    counts_labels = ["less_severe", "same_severity", "harsher"]
    counts_values = [
        float(deltas.get("full_system_less_severe_count", 0.0) or 0.0),
        float(deltas.get("same_severity_count", 0.0) or 0.0),
        float(deltas.get("full_system_harsher_count", 0.0) or 0.0),
    ]
    _bar_plot(
        output_dir / "full_vs_no_defence_severity_counts.png",
        "Full System vs No Defence: Severity Outcome Counts",
        counts_labels,
        counts_values,
        "Case count",
    )

    print(f"Saved graphs to: {output_dir}")


if __name__ == "__main__":
    main()
