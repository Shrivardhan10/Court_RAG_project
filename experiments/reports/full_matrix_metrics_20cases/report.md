# Metrics Report

Input file: experiments\reports\experiment_runs_20cases.jsonl

## Retrieval
- IPC Top-k Accuracy: 0.257
- IPC Precision: 0.046
- IPC Recall: 0.252
- IPC F1: 0.077
- Precedent Recall@k: 0.000
- Precedent MRR: 0.000
- Precedent nDCG@k: 0.000
- Avg Semantic Similarity: N/A

## Generation
- Prosecutor IPC Correctness: 0.301
- Defence IPC Correctness: 0.088
- Judge IPC Correctness: 0.226
- Prosecutor Quality Score: 0.686
- Defence Quality Score: 0.386
- Judge Quality Score: 0.800
- Prosecutor Reasoning Score: N/A
- Defence Reasoning Score: N/A
- Judge Reasoning Score: N/A
- Hallucination Rate: N/A

## End-to-End
- Judgment Accuracy: 0.029
- Judgment IPC Recall: 0.226
- Judgment IPC Precision: 0.111
- Judgment IPC Jaccard: 0.086
- Sentencing Accuracy: 0.136
- Consistency Score (Prosecutor vs Judge IPC): 0.754
- Charge Correction Rate: 0.007
- Robustness Stability: 0.2857142857142857
- Avg Total Latency (s): 0.3978888271509537
- Avg Prosecution Latency (s): 0.22517490167325985
- Avg Defence Latency (s): 0.38217348666706435
- Avg Judge Latency (s): 0.11964712000529593

## Valid-Only Scope
- Rows: total=140, valid=100
- Judgment Accuracy (valid only): 0.040
- Judgment IPC Recall (valid only): 0.317
- Judgment IPC Jaccard (valid only): 0.120
- Sentencing Accuracy (valid only): 0.190

## Variant End-to-End (Valid Only)

- full_system: n=20, n_valid=20, judgment_accuracy=0.000, judge_ipc_recall=0.408, judge_ipc_jaccard=0.081
- no_defence_agent: n=20, n_valid=20, judgment_accuracy=0.000, judge_ipc_recall=0.408, judge_ipc_jaccard=0.081
- no_ipc_retrieval: n=20, n_valid=20, judgment_accuracy=0.000, judge_ipc_recall=0.383, judge_ipc_jaccard=0.081
- no_precedent: n=20, n_valid=20, judgment_accuracy=0.050, judge_ipc_recall=0.142, judge_ipc_jaccard=0.142
- rule_only_ipc_mapper: n=20, n_valid=20, judgment_accuracy=0.150, judge_ipc_recall=0.242, judge_ipc_jaccard=0.217
- single_llm_no_rag: n=20, n_valid=0, judgment_accuracy=0.000, judge_ipc_recall=0.000, judge_ipc_jaccard=0.000
- single_llm_with_rag: n=20, n_valid=0, judgment_accuracy=0.000, judge_ipc_recall=0.000, judge_ipc_jaccard=0.000

## Full-System Comparative Deltas
- full_system_vs_no_defence_agent: paired_cases=20, judgment_accuracy_delta=0.0, judge_ipc_recall_delta=0.0, judge_ipc_jaccard_delta=0.0, judge_sentence_severity_delta=-36.3, judge_compensation_delta=-157500.0, full_system_less_severe_count=8, full_system_harsher_count=0, same_severity_count=12
- full_system_vs_single_llm_no_rag: paired_cases=0, judgment_accuracy_delta=None, judge_ipc_recall_delta=None, judge_ipc_jaccard_delta=None, judge_sentence_severity_delta=None, judge_compensation_delta=None, full_system_less_severe_count=0, full_system_harsher_count=0, same_severity_count=0
- full_system_vs_single_llm_with_rag: paired_cases=0, judgment_accuracy_delta=None, judge_ipc_recall_delta=None, judge_ipc_jaccard_delta=None, judge_sentence_severity_delta=None, judge_compensation_delta=None, full_system_less_severe_count=0, full_system_harsher_count=0, same_severity_count=0

## Ablation
- full_system: n=20, judgment_accuracy=0.000
- no_defence_agent: n=20, judgment_accuracy=0.000
- no_ipc_retrieval: n=20, judgment_accuracy=0.000
- no_precedent: n=20, judgment_accuracy=0.050
- rule_only_ipc_mapper: n=20, judgment_accuracy=0.150
- single_llm_no_rag: n=20, judgment_accuracy=0.000
- single_llm_with_rag: n=20, judgment_accuracy=0.000
