# Experiment + Metrics Workflow

This is the single source of truth for running your baseline matrix and generating final metrics artifacts.

## Variants in the matrix

- full_system
- no_precedent
- no_ipc_retrieval
- no_defence_agent
- single_llm_no_rag
- single_llm_with_rag
- rule_only_ipc_mapper

## What each baseline means

- no_precedent: precedence retrieval is disabled.
- no_ipc_retrieval: IPC retrieval is disabled.
- no_defence_agent: defence output is replaced with a fixed minimal placeholder (no generated defence reasoning).
- single_llm_no_rag: one-shot judgment from facts only.
- single_llm_with_rag: one-shot judgment using retrieved context.
- rule_only_ipc_mapper: keyword rules + templates (no LLM reasoning).

## Metrics you are focusing on

For your educational courtroom objective, prioritize these learning-focused metrics:

- prosecutor_quality_score
- defence_quality_score
- judge_quality_score
- defence_ipc_correctness
- judge_ipc_correctness
- judgment_ipc_recall_valid_only
- consistency_score_valid_only
- full_vs_no_defence.defence_quality_delta
- full_vs_rule_only.*_quality_delta

Keep outcome-exact metrics (strict final IPC match, sentencing exact match) as secondary.

## Required input benchmark format

JSONL, one case per line:

- case_id
- fact_text
- expected.ipc_sections
- expected.relevant_case_ids (optional)
- expected.sentence (optional)
- paraphrase_group_id (optional)

Sample benchmark:

- experiments/data/sample_benchmark.jsonl

## Final commands to run

Run from project root.

### 1) Generate final experiment run JSONL

```powershell
python experiments/run_experiment_matrix.py --benchmark experiments/data/sample_benchmark.jsonl --output experiments/reports/experiment_runs_full_matrix.jsonl
```

### 2) Generate full metrics JSON + MD

```powershell
python metrics/evaluator.py --input experiments/reports/experiment_runs_full_matrix.jsonl --k 5 --output-dir experiments/reports/full_matrix_metrics_v3
```

Outputs:

- experiments/reports/full_matrix_metrics_v3/summary.json
- experiments/reports/full_matrix_metrics_v3/report.md

### 3) Generate learning-focused metrics JSON + MD

```powershell
python metrics/learning_focused_report.py --summary experiments/reports/full_matrix_metrics_v3/summary.json --output-dir experiments/reports/learning_focused_metrics
```

Outputs:

- experiments/reports/learning_focused_metrics/summary_learning_focused.json
- experiments/reports/learning_focused_metrics/report_learning_focused.md

## Notes on valid rows

- single_llm_no_rag and single_llm_with_rag may fail due to API quota/rate limits.
- learning-focused reports include `total_rows`, `valid_rows`, and `excluded_failed_rows` so comparisons remain transparent.
