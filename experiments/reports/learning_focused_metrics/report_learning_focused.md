# Learning-Focused Metrics Report

## Scope
- total_rows: 21
- valid_rows: 15
- excluded_failed_rows: 6

## Learning Core Metrics (Full System)
- prosecutor_quality_score: 1.000
- defence_quality_score: 0.867
- judge_quality_score: 1.000
- defence_ipc_correctness: 0.500
- judge_ipc_correctness: 1.000

## Supporting Reasoning Metrics
- judgment_ipc_recall_valid_only: 0.867
- judgment_ipc_jaccard_valid_only: 0.374
- consistency_score_valid_only: 0.604

## Multi-Agent Deltas
- full_vs_no_defence.defence_quality_delta: 0.667
- full_vs_no_defence.defence_ipc_correctness_delta: 0.500
- full_vs_rule_only.prosecutor_quality_delta: 0.200
- full_vs_rule_only.judge_quality_delta: 0.200
- full_vs_rule_only.defence_quality_delta: 0.667

## Note
Outcome-exact metrics are intentionally excluded from this focused report because this view is for courtroom argument-learning quality.
