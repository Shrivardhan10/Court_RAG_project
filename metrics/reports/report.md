# Metrics Report

Input file: metrics\data\sample_eval_runs.jsonl

## Retrieval
- IPC Top-k Accuracy: 0.500
- IPC Precision: 0.167
- IPC Recall: 0.500
- IPC F1: 0.250
- Precedent Recall@k: 0.500
- Precedent MRR: 0.500
- Precedent nDCG@k: 0.500
- Avg Semantic Similarity: 0.7

## Generation
- Prosecutor IPC Correctness: 0.500
- Defence IPC Correctness: 0.000
- Judge IPC Correctness: 0.500
- Prosecutor Reasoning Score: 3.4
- Defence Reasoning Score: 3.3
- Judge Reasoning Score: 3.6
- Hallucination Rate: 0.0

## End-to-End
- Judgment Accuracy: 0.500
- Sentencing Accuracy: 0.500
- Consistency Score (Prosecutor vs Judge IPC): 0.750
- Charge Correction Rate: 0.000
- Robustness Stability: 0.5
- Avg Total Latency (s): 37.2
- Avg Prosecution Latency (s): 20.55
- Avg Defence Latency (s): 16.4
- Avg Judge Latency (s): 0.15000000000000002

## Ablation
- full_system: n=1, judgment_accuracy=1.000
- no_precedent: n=1, judgment_accuracy=0.000
