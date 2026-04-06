# Metrics Report

Input file: experiments\reports\experiment_runs_full_matrix.jsonl

## Retrieval
- IPC Top-k Accuracy: 0.381
- IPC Precision: 0.048
- IPC Recall: 0.286
- IPC F1: 0.080
- Precedent Recall@k: 0.000
- Precedent MRR: 0.000
- Precedent nDCG@k: 0.000
- Avg Semantic Similarity: N/A

## Generation
- Prosecutor IPC Correctness: 0.643
- Defence IPC Correctness: 0.143
- Judge IPC Correctness: 0.619
- Prosecutor Quality Score: 0.686
- Defence Quality Score: 0.390
- Judge Quality Score: 0.800
- Prosecutor Reasoning Score: N/A
- Defence Reasoning Score: N/A
- Judge Reasoning Score: N/A
- Hallucination Rate: N/A

## End-to-End
- Judgment Accuracy: 0.143
- Judgment IPC Recall: 0.619
- Judgment IPC Precision: 0.267
- Judgment IPC Jaccard: 0.267
- Sentencing Accuracy: 0.095
- Consistency Score (Prosecutor vs Judge IPC): 0.717
- Charge Correction Rate: 0.053
- Robustness Stability: 0.19047619047619047
- Avg Total Latency (s): 0.7069435619438688
- Avg Prosecution Latency (s): 0.4886084778182622
- Avg Defence Latency (s): 0.6030684444639418
- Avg Judge Latency (s): 0.23166630000600386

## Valid-Only Scope
- Rows: total=21, valid=15
- Judgment Accuracy (valid only): 0.200
- Judgment IPC Recall (valid only): 0.867
- Judgment IPC Jaccard (valid only): 0.374
- Sentencing Accuracy (valid only): 0.133

## Variant End-to-End (Valid Only)

- full_system: n=3, n_valid=3, judgment_accuracy=0.000, judge_ipc_recall=1.000, judge_ipc_jaccard=0.229
- no_defence_agent: n=3, n_valid=3, judgment_accuracy=0.000, judge_ipc_recall=1.000, judge_ipc_jaccard=0.229
- no_ipc_retrieval: n=3, n_valid=3, judgment_accuracy=0.000, judge_ipc_recall=1.000, judge_ipc_jaccard=0.245
- no_precedent: n=3, n_valid=3, judgment_accuracy=0.333, judge_ipc_recall=0.333, judge_ipc_jaccard=0.333
- rule_only_ipc_mapper: n=3, n_valid=3, judgment_accuracy=0.667, judge_ipc_recall=1.000, judge_ipc_jaccard=0.833
- single_llm_no_rag: n=3, n_valid=0, judgment_accuracy=0.000, judge_ipc_recall=0.000, judge_ipc_jaccard=0.000
- single_llm_with_rag: n=3, n_valid=0, judgment_accuracy=0.000, judge_ipc_recall=0.000, judge_ipc_jaccard=0.000

## Full-System Comparative Deltas
- full_system_vs_no_defence_agent: paired_cases=3, judgment_accuracy_delta=0.0, judge_ipc_recall_delta=0.0, judge_ipc_jaccard_delta=0.0
- full_system_vs_single_llm_no_rag: paired_cases=0, judgment_accuracy_delta=None, judge_ipc_recall_delta=None, judge_ipc_jaccard_delta=None
- full_system_vs_single_llm_with_rag: paired_cases=0, judgment_accuracy_delta=None, judge_ipc_recall_delta=None, judge_ipc_jaccard_delta=None

## Ablation
- full_system: n=3, judgment_accuracy=0.000
- no_defence_agent: n=3, judgment_accuracy=0.000
- no_ipc_retrieval: n=3, judgment_accuracy=0.000
- no_precedent: n=3, judgment_accuracy=0.333
- rule_only_ipc_mapper: n=3, judgment_accuracy=0.667
- single_llm_no_rag: n=3, judgment_accuracy=0.000
- single_llm_with_rag: n=3, judgment_accuracy=0.000
