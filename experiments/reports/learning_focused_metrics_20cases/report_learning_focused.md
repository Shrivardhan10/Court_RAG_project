# Learning-Focused Metrics Report

## Scope
- total_rows: 140
- valid_rows: 100
- excluded_failed_rows: 40

## Learning Core Metrics (Full System)
- prosecutor_quality_score: 1.000
- defence_quality_score: 0.850
- judge_quality_score: 1.000
- defence_ipc_correctness: 0.350
- judge_ipc_correctness: 0.408

## Supporting Reasoning Metrics
- judgment_ipc_recall_valid_only: 0.317
- judgment_ipc_jaccard_valid_only: 0.120
- consistency_score_valid_only: 0.655

## Multi-Agent Deltas
- full_vs_no_defence.defence_quality_delta: 0.650
- full_vs_no_defence.defence_ipc_correctness_delta: 0.350
- full_vs_rule_only.prosecutor_quality_delta: 0.200
- full_vs_rule_only.judge_quality_delta: 0.200
- full_vs_rule_only.defence_quality_delta: 0.650

## Note
Outcome-exact metrics are intentionally excluded from this focused report because this view is for courtroom argument-learning quality.
