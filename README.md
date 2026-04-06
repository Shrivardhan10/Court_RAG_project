# Court RAG – Technical System Report

This document is a code-level technical report for the Court RAG project, based entirely on the current repository (Python modules, notebooks, and generated artifacts).

---

## 1. Abstract

Court RAG is an Indian criminal-law reasoning system that combines:

- Offline ingestion of reported criminal judgments into a structured precedent corpus.
- A two-stage text pipeline: Stage 0 cleans and sentence-splits legal text; Stage 1 classifies sentences into legal roles (FACT / ARGUMENT / REASONING).
- Dual retrieval over precedents and IPC statutes using FAISS + BM25 with domain-specific reranking.
- A three-agent online reasoning loop (prosecutor, defence, judge) implemented with Google Gemini, where the judge is constrained to decide only on the basis of what the two parties present.

The project also includes an experiment matrix and metrics suite that evaluate retrieval, generation quality, and end‑to‑end judgment behaviour on a small benchmark.

---

## 2. System Overview

At a high level, the system consists of:

- **Offline precedent ingestion**
  - `embedding_generation/tryVec.py` reads `cases/*.pdf`, cleans them, builds multi‑granular chunks, and saves `processed_cases/*.json`.
  - 290 JSON precedent files are currently present under `processed_cases/` (counted via the filesystem).
- **Retrieval index build**
  - `backend/retrieval/index_precedents.py` builds FAISS + BM25 indexes over `processed_cases/`.
  - `backend/retrieval/index_ipc.py` builds FAISS + BM25 indexes over `ipc.json`.
  - Artifacts are stored under `vector_db/`.
- **Stage 0 + Stage 1 sentence pipeline**
  - `stage0/` cleans and sentence‑splits raw judgments or user narratives.
  - `stage1/` labels sentences as FACT / ARGUMENT / REASONING using a hybrid rules + transformer classifier.
- **Online multi‑agent backend**
  - `backend/prosecution_pipeline.py` orchestrates retrieval + Gemini prompts for:
    - `prosecution_arguments(user_case)`
    - `defence_arguments(prosecutor_output, user_case)`
    - `judge_arguments(prosecutor_output, defence_output, user_case)`
    - `run_case(user_case)` returns a dict with `prosecution`, `defence`, `judge` paragraphs.
- **Experiments and metrics**
  - `experiments/run_experiment_matrix.py` runs a fixed variant matrix on a JSONL benchmark.
  - `metrics/evaluator.py` and `metrics/learning_focused_report.py` compute retrieval, generation, and end‑to‑end metrics and render Markdown reports.

---

## 3. Architecture

### 3.1 Module layout

- **`backend/` (online inference)**
  - `prosecution_pipeline.py`: structured preprocessing, retrieval orchestration, Gemini prompts for prosecutor/defence/judge, sentencing and compensation normalization.
  - `retrieval/`:
    - `precedent_retriever.py`: loads `processed_cases/*.json`, builds and queries FAISS + BM25 precedent index.
    - `statute_retriever.py`: loads `ipc.json`, builds and queries FAISS + BM25 IPC index.
    - `hybrid_retriever.py`: combines semantic and BM25 scores plus IPC/domain bonuses for final reranking.
    - `bm25_index.py`: standalone BM25 implementation (tokenization, IDF, scoring, top‑N helpers).
    - `index_precedents.py`, `index_ipc.py`: CLI entrypoints for building indexes.
  - `utils/llm.py`: single Gemini client wrapper with configurable model name, fallbacks, and timeouts.

- **`embedding_generation/` (offline precedent preparation)**
  - `parser.py`: PDF text extraction via PyPDF2 and header/citation cleanup.
  - `extractor.py`: heuristics to detect IPC sections, actions, weapons, outcomes, intent, judgment, court, and year; per‑sentence role scoring.
  - `chunker.py`: builds multi‑granular chunks (facts, issues, arguments, reasoning, judgment, IPC summary) with length bounds suitable for embedding.
  - `tryVec.py`: batch driver from `cases/*.pdf` to `processed_cases/*.json`.

- **`stage0/` (sentence‑level preprocessing)**
  - `cleaner.py`: removes boilerplate (court headings, dates, case numbers, judge names, citations) and normalizes whitespace.
  - `splitter.py`: robust legal sentence splitter with abbreviation protection and long‑sentence handling.
  - `pipeline.py`: case readers (`process_case`, `process_all_cases`), in‑memory `process_text`, and JSON writers for Stage 0 outputs.

- **`stage1/` (sentence classification)**
  - `rules.py`: keyword rules for each label plus simple date‑driven factual boosts.
  - `classifier.py`: hybrid combiner that always runs the transformer model and lets strong rules override it per sentence.
  - `model.py`: global transformer singleton, GPU placement, batched inference, and keyword‑boosted fallbacks.
  - `dataset.py`, `trainer.py`: JSON training data loader and fine‑tuning script for Legal‑BERT / DistilBERT.
  - `pipeline.py`: Stage 1 orchestration (`process_case_stage1`, `process_sentences_stage1`, `process_all_cases_stage1`).

- **`experiments/` and `metrics/`**
  - `experiments/run_experiment_matrix.py`: runs seven variants (full system and ablations/baselines) over a JSONL benchmark and logs outputs + latencies.
  - `metrics/evaluator.py`: computes retrieval, generation, and end‑to‑end metrics and ablations from JSONL logs.
  - `metrics/learning_focused_report.py`: derives a “learning‑focused” subset of metrics for argument quality.

### 3.2 Execution modes

- **Offline indexing mode**
  1. Run `embedding_generation/tryVec.py` to populate `processed_cases/` from `cases/*.pdf`.
  2. Run `backend/retrieval/index_precedents.py` and `backend/retrieval/index_ipc.py` to populate `vector_db/`.

- **Online multi‑agent inference mode**
  - Import and call `run_case(user_case: str)` from `backend/prosecution_pipeline.py`. This performs Stage 0 + Stage 1 preprocessing (if available), retrieval, and three Gemini calls (prosecutor, defence, judge).

- **Experiment + metrics mode**
  - Use `experiments/run_experiment_matrix.py` over a benchmark JSONL; then feed the resulting JSONL into `metrics/evaluator.py` and `metrics/learning_focused_report.py` to produce `summary.json` and Markdown reports.

---

## 4. Stage 0 (Preprocessing)

Stage 0 is implemented in `stage0/cleaner.py`, `stage0/splitter.py`, and `stage0/pipeline.py`.

### 4.1 Cleaning logic

The cleaner operates line‑by‑line with explicit regular expressions for typical Indian judgment boilerplate:

- **Boilerplate line dropping**
  - Court headers: lines mentioning “High Court”, “Supreme Court”, “District Court”, “Sessions Court”, “Tribunal”, etc.
  - Standalone dates in common formats (`dd Month yyyy`, `dd/mm/yyyy`, `Month dd, yyyy`).
  - Judge name lines, CORAM blocks, and signature lines ending with `J.`, `C.J.`, or `JJ.`.
  - Case numbers and prefixes (`Cr.A. No …`, `Writ Petition No …`, `Civil Appeal No …`).
  - Procedural order lines like “Leave granted”, “Appeal dismissed/allowed”, “Order accordingly”.
  - Single‑line case titles like “X v. Y” without sentence punctuation.

- **Inline cleanup**
  - Removal of common law report citations (AIR, SCC, Indlaw, SCR, AC) via explicit regex patterns.
  - Removal of any XML‑like tokens (`<...>`), paragraph trailing numbers, and reduction of repeated whitespace.
  - Collapsing of 3+ consecutive blank lines into a maximum of two.

The result is a cleaned multi‑line string that preserves substantive narrative while stripping metadata, citations, and formatting noise.

### 4.2 Sentence splitting

Sentence splitting is handled by `stage0/splitter.py` and is tailored to legal text:

- **Abbreviation protection**
  - A large set of patterns (`v.`, `No.`, `u/s.`, `Mr.`, `Dr.`, `Ltd.`, `IPC.`, `Art.`, single‑letter initials, etc.) is temporarily rewritten to placeholders (e.g., `v__`) so that later boundary detection does not split on these periods.

- **Paragraph and list structure**
  - Splits on numbered paragraph markers like `1.`, `2.` at line starts.
  - Detects structured list items such as `(i)`, `(ii)`, `(a)`, `(b)`, `(1)` and begins new segments at these markers.

- **Sentence boundary detection**
  - Splits on punctuation + space + capital/quote (`[.!?]` followed by an uppercase letter or quote).
  - Splits “no‑space” boundaries like `1951.When` by inserting a space after the period.

- **Long‑sentence handling and fragment merging**
  - Sentences longer than 40 words are optionally split on comma + conjunction boundaries (e.g., `, that`, `, which`, `, however`) to make them more manageable.
  - Fragments under 6 words are merged back into the previous sentence to avoid over‑fragmentation.

After restoring abbreviations and merging fragments, Stage 0 returns a list of sentences optimized for downstream classification and prompting.

### 4.3 Stage 0 pipeline and storage

`stage0/pipeline.py` provides the orchestration layer:

- `process_case(file_path: str) -> List[str]`
  - Reads a `.txt` judgment (UTF‑8 with Latin‑1 fallback), applies `clean_text`, then `split_sentences`.
- `process_text(text: str) -> List[str]`
  - In‑memory variant for raw user narratives.
- `save_stage0_output(case_id: str, sentences: List[str], output_dir: str = "data/stage0")`
  - Writes `{ "case_id": ..., "sentences": [...] }` JSON files (directory is created on demand).
- `process_all_cases(folder_path: str, output_dir: str = "data/stage0")`
  - Iterates all `.txt` files, processes them, saves JSON, and logs progress / completion ratios.

Stage 0 is used both as a preprocessing step for Stage 1 and as part of the interactive pipeline when `backend/prosecution_pipeline.py` imports `stage0.pipeline.process_text`.

---

## 5. Stage 1 (Classification)

Stage 1 assigns each sentence a legal role label from `{FACT, ARGUMENT, REASONING}`.

### 5.1 Label space and data utilities

- `stage1/dataset.py` defines `LABEL2ID = {"FACT": 0, "ARGUMENT": 1, "REASONING": 2}` and the inverse mapping.
- `load_training_data(path)` expects JSON of the form `[{"text": "...", "label": "FACT"}, ...]` and returns parallel lists of texts and integer labels.
- When PyTorch is installed, `LegalDataset` wraps tokenized inputs and labels for use with `DataLoader`.

### 5.2 Hybrid rules + ML classifier

The hybrid logic resides in `stage1/rules.py` and `stage1/classifier.py`:

- **Rule scoring (`rules.py`)**
  - Maintains curated keyword lists for REASONING, ARGUMENT, and FACT (e.g., “court observed”, “argued”, “witness”, “post‑mortem”), plus a date pattern.
  - `score_sentence(sentence)` counts regex matches for each class and adds a small FACT boost when dates are present.
  - `get_rule_label` applies a simple decision policy:
    - Prioritise REASONING when its score is clearly dominant.
    - Otherwise ARGUMENT, with FACT used only as a weak fallback.

- **Hybrid combination (`classifier.py`)**
  - `_STRONG_THRESHOLD = 2`.
  - `classify_batch(sentences)`:
    - Computes rule scores for every sentence.
    - Always calls `stage1.model.predict_batch(sentences)`; there is no rules‑only shortcut.
    - For each sentence:
      - If REASONING score ≥ 2 and ≥ ARGUMENT score → label `REASONING`.
      - Else if ARGUMENT score ≥ 2 and > REASONING score → label `ARGUMENT`.
      - Else use the model’s predicted label.

Thus rules act as high‑precision overrides, while the transformer provides the default.

### 5.3 Model and training

`stage1/model.py` encapsulates transformer loading and inference:

- **Model selection and loading**
  - Prefers a fine‑tuned model under `stage1_trained_model/` if the directory exists.
  - Otherwise falls back to base models in order: `nlpaueb/legal-bert-base-uncased` → `distilbert-base-uncased`.
  - Uses `ALLOW_REMOTE_MODEL_DOWNLOAD` (env var) to decide whether remote downloads are allowed (`local_files_only` flag).
  - Moves the model to `cuda` if `torch.cuda.is_available()`, otherwise `cpu`, and keeps it in a global singleton.

- **Batched GPU inference**
  - `INFERENCE_BATCH_SIZE = 32`.
  - `predict_batch(sentences)` tokenizes all inputs and runs them through the model in sub‑batches of at most 32 sentences.
  - For fine‑tuned models, logits are mapped directly to labels via `ID2LABEL`.
  - For base models (no fine‑tuning), a keyword‑boosting layer adjusts labels using reasoning/argument regexes when confidence is ambiguous.
  - On any failure (missing model, runtime error), it falls back to a pure keyword classifier with fixed confidence scores.

- **Training script**
  - `stage1/trainer.py` fine‑tunes the classifier:
    - Loads data via `load_training_data`.
    - Loads a base model and tokenizer (`BASE_MODEL` or `FALLBACK_MODEL`) with `ignore_mismatched_sizes=True`.
    - Trains for a configurable number of epochs with AdamW and CrossEntropyLoss.
    - Reports running loss and accuracy, then saves both model and tokenizer to `stage1_trained_model/`.

### 5.4 Stage 1 pipeline and outputs

`stage1/pipeline.py` integrates Stage 0 outputs with classification:

- `_ensure_model_loaded()` ensures the model singleton is initialised once per process.
- `process_case_stage1(file_path)`:
  - Reads a Stage 0 JSON (`{"case_id": ..., "sentences": [...]}`), filters empty/whitespace‑only strings.
  - Runs `classify_batch` and returns `[{"text": s, "label": l}, ...]`.
- `process_sentences_stage1(sentences)` and `process_text_stage1(text)`
  - In‑memory variants for lists of sentences or raw text (which is first passed through `stage0.pipeline.process_text`).
- `save_stage1_output(case_id, labeled_sentences, output_dir="data/stage1")`
  - Persists labeled outputs as JSON for later analysis.
- `process_all_cases_stage1(folder_path="data/stage0", output_dir="data/stage1")`
  - Runs Stage 1 classification over all Stage 0 JSON files and logs throughput and model availability.

Stage 1 labels are also used by the multi‑agent backend as structured features (facts/arguments/reasoning cues) for prompt construction and retrieval queries.

---

## 6. Model & GPU Optimization

The project uses transformer models in two distinct roles: sentence classification and retrieval embeddings.

- **Sentence classification (Stage 1)**
  - Uses Hugging Face transformers with GPU acceleration when available.
  - Applies batched inference (`INFERENCE_BATCH_SIZE = 32`) to avoid per‑sentence overhead.
  - Keeps a global model/tokenizer singleton to avoid repeated loading.

- **Retrieval embeddings**
  - `backend/retrieval/precedent_retriever.py` and `statute_retriever.py` use SentenceTransformers (`all-MiniLM-L6-v2`) on CPU by default.
  - Embeddings are normalised and indexed in FAISS (`IndexFlatIP`) for inner‑product similarity search.
  - BM25 indexes are built in parallel over tokenized texts for lexical matching.

- **Reranking and domain heuristics**
  - `hybrid_retriever.py` combines normalised semantic and BM25 scores, plus:
    - IPC overlap bonuses between query and record sections for precedents.
    - Chunk‑type bonuses (argument > reasoning > judgment > facts) for precedent chunks.
    - Domain‑specific bonuses for homicide, fraud, and injury sections in IPC retrieval.

The design favours predictable, explainable scoring and batching over aggressive GPU micro‑optimisation, while still exploiting GPU for Stage 1 and (optionally) model training.

---

## 7. Data Flow

This section summarises input → processing → output across the main pipelines.

### 7.1 Precedent ingestion and indexing

- **Input**: Raw PDF judgments under `cases/`.
- **Processing**:
  - `embedding_generation/parser.py` extracts text via PyPDF2 and removes front matter, citations, and appearance lists.
  - `embedding_generation/extractor.py` identifies IPC sections, actions, outcomes, weapons, intent, judgment, court, and year.
  - `embedding_generation/chunker.py` tokenizes into sentences, scores each by role, and builds up to six multi‑sentence chunks (facts, issue, argument, reasoning, judgment, IPC summary) with token count constraints.
  - `embedding_generation/tryVec.py` packages chunks into a JSON payload per case.
- **Output**:
  - `processed_cases/{case_id}.json` with schema:
    - `case_id`, `case_name`, `chunk_count`, and a list of `chunks` each containing `chunk_type`, `text`, `embedding_text`, and `metadata`.
  - There are currently 290 such JSON files.

Indexing then proceeds via:

- **Input**: `processed_cases/*.json` and `ipc.json`.
- **Processing**:
  - `precedent_retriever.build_and_save_precedent_index` normalises legacy and structured formats into flat records, builds embeddings, FAISS index, and BM25 index.
  - `statute_retriever.build_and_save_statute_index` builds section‑level embeddings and BM25 index from IPC metadata.
- **Output**:
  - FAISS indexes and pickled metadata/BM25 structures under `vector_db/`.

### 7.2 Stage 0 → Stage 1 sentence pipeline

- **Input**: Raw `.txt` case files (for batch mode) or raw narrative text (for interactive use).
- **Processing**:
  - Stage 0: cleaning (`clean_text`) and sentence splitting (`split_sentences`).
  - Stage 1: hybrid rules + transformer classification with batched GPU inference.
- **Output**:
  - Batch mode: `data/stage0/{case_id}.json` and `data/stage1/{case_id}.json` (case‑wise sentence lists and labeled sentences).
  - Interactive mode: in‑memory lists used to build structured inputs for retrieval and prompting.

### 7.3 Online three‑agent reasoning pipeline

Implemented in `backend/prosecution_pipeline.py`:

- **Input**: `user_case: str` – free‑form description of a criminal incident or case.

- **Processing**:
  1. **Structured preprocessing**
     - Uses Stage 0 + Stage 1 (if importable) to segment `user_case` into sentences and label them.
     - `_build_structured_input` derives:
       - `facts`, `arguments`, `reasoning` (top sentences per label).
       - `keywords` (filtered word frequencies) and IPC section candidates from text.
       - `retrieval_query` combining these signals.
  2. **Retrieval over precedents and statutes**
     - `initialize_precedent_retriever` / `initialize_statute_retriever` ensure indexes exist.
     - `retrieve_precedents(retrieval_query, top_k=8, ipc_sections=...)` returns structured precedent rows.
     - `retrieve_statutes` is called in several modes (by query, by section filter, by IPC hints) and results are merged and reranked with domain bonuses.
  3. **Prosecutor agent** – `prosecution(user_case)` / `prosecution_arguments`
     - Cleans retrieved precedents/statutes (`_clean_precedents`, `_clean_statutes`).
     - Computes a legally grounded sentencing anchor via `_specific_sentence_request`.
     - Builds a long, instruction‑rich prompt describing narrative order and constraints, then calls `utils.llm.generate_response`.
     - Normalises the output into a single paragraph and enforces:
       - Not truncated (`_looks_truncated`).
       - Contains specific sentencing and compensation (`_has_specific_sentencing`).
       - Resembles a proper oral argument (`_looks_like_good_oral_argument`), otherwise triggers up to two rewrite attempts.
  4. **Defence agent** – `defence_arguments(prosecutor_output, user_case)`
     - Treats the prosecutor’s paragraph as primary input.
     - Re‑derives structured features from the prosecutor text and collects defence‑friendly precedents using `_defence_precedent_score`.
     - Retrieves supporting IPC statutes and builds a defence prompt that either challenges culpability or focuses on mitigation, depending on `_prosecutor_indicates_strong_guilt`.
     - Applies the same truncation and specificity checks; on failure, synthesises a conservative defence paragraph.
  5. **Judge agent** – `judge_arguments(prosecutor_output, defence_output, user_case)`
     - **Does not call any retrieval functions.**
     - Extracts party‑cited precedents (`_party_precedents`) and statutes (`_party_statutes`) directly from the two agent outputs.
     - Computes a judicial relief profile via `_judge_relief_from_party_outputs`, taking into account defence mitigation signals.
     - Builds a judge prompt that must:
       - State final IPC sections applied.
       - Specify exact sentence (death/life/term in years).
       - Specify compensation amount in INR.
     - Enforces the same truncation and specificity checks, falling back to a rule‑based synthesised judgment if the LLM call fails.

- **Output**:
  - `run_case(user_case)` returns a dict:
    - `{"prosecution": <paragraph>, "defence": <paragraph>, "judge": <paragraph>}`.

### 7.4 Experiment + metrics flow

- **Input**: JSONL benchmark (e.g., `experiments/data/sample_benchmark.jsonl`) with fields:
  - `case_id`, `fact_text`, `expected.ipc_sections`, optional `expected.relevant_case_ids`, `expected.sentence`, and `paraphrase_group_id`.
- **Processing**:
  - `experiments/run_experiment_matrix.py` runs seven system variants per row:
    - `full_system`, `no_precedent`, `no_ipc_retrieval`, `no_defence_agent`, `single_llm_no_rag`, `single_llm_with_rag`, `rule_only_ipc_mapper`.
    - Logs `outputs` (prosecution/defence/judge texts), `retrieval` snapshots (retrieved case ids and IPC sections), and `latency` per phase.
  - `metrics/evaluator.py` loads the resulting JSONL and computes metrics, writing `summary.json` and `report.md`.
  - `metrics/learning_focused_report.py` reads a summary and produces a condensed `summary_learning_focused.json` + `report_learning_focused.md`.
- **Output**:
  - In the checked‑in `experiments/reports/full_matrix_metrics_v3/summary.json`, each variant has `n = 3` rows, so the example full‑matrix run covers 3 benchmark fact patterns × 7 variants = 21 runs.

---

## 8. Metrics & Evaluation

This section summarises the implemented metrics and the concrete values present in `experiments/reports/full_matrix_metrics_v3/summary.json` (3‑case benchmark, 7 variants). All numbers below come directly from that file or from the associated code.

### 8.1 Retrieval metrics

From the top‑level `retrieval` block:

- IPC Top‑k Accuracy (k=5): **0.381** – fraction of cases where at least one expected IPC section appears in the top‑5 retrieved sections.
- IPC Precision: **0.048**; IPC Recall: **0.286**; IPC F1: **0.080** – measured over retrieved vs expected IPC sections.
- Precedent Recall@k, MRR, and nDCG@k are all **0.0** in this small run, indicating that expected `relevant_case_ids` were not recovered in the top‑k set on this sample.

### 8.2 Generation metrics (argument quality)

From the top‑level `generation` block:

- Prosecutor IPC correctness: **0.643** (average fraction of expected IPC sections mentioned in the prosecutor output).
- Defence IPC correctness: **0.143**.
- Judge IPC correctness: **0.619**.
- Quality scores (0–1 composite from IPC presence, sentencing specificity, compensation presence, paragraph structure, and minimum sentence count):
  - Prosecutor: **0.686**
  - Defence: **0.390**
  - Judge: **0.800**

### 8.3 End‑to‑end metrics

From `end_to_end` and `end_to_end_valid_only`:

- **All rows (including failures):**
  - Judgment (exact IPC) accuracy: **0.143**.
  - Judgment IPC recall: **0.619**; IPC precision & Jaccard: **0.267**.
  - Sentencing accuracy (exact match to expected type/years): **0.095**.
  - Consistency score (Jaccard between prosecutor and judge IPC sets): **0.717**.
  - Charge correction rate (judge correcting mis‑charged prosecutor cases): **0.053**.
  - Robustness stability across paraphrase groups: **0.190**.
  - Avg total latency per run: **≈0.707 s**; prosecution ≈0.489 s; defence ≈0.603 s; judge ≈0.232 s.

- **Valid‑only rows (excluding LLM failures for some variants):**
  - Judgment accuracy: **0.200**.
  - Judgment IPC recall: **0.867**; IPC precision & Jaccard: **0.374**.
  - Sentencing accuracy: **0.133**.
  - Consistency score: **0.604**.
  - Robustness stability: **0.267**.

### 8.4 Variant‑level behaviour

From `variant_metrics` and `ablation_valid_only`:

- Each variant has `n = 3` rows in this sample run.
- Judgment accuracy (valid‑only, per variant):
  - `full_system`: **0.000**
  - `no_precedent`: **0.333**
  - `no_ipc_retrieval`: **0.000**
  - `no_defence_agent`: **0.000**
  - `rule_only_ipc_mapper`: **0.667**
  - `single_llm_no_rag` and `single_llm_with_rag`: **0.000** with `n_valid = 0` due to API failures.

On this tiny benchmark, the simple rule‑only IPC mapper achieves the highest exact judgment accuracy, while the full multi‑agent system achieves high IPC recall but low exact match. This behaviour is consistent with the code: the rule‑only mapper is deterministic and tightly coupled to IPC heuristics, whereas the multi‑agent pipeline optimises for rich narrative quality and proportional sentencing.

### 8.5 Requested metrics vs. implementation

- **Number of cases processed**
  - Precedent corpus: **290** processed judgments under `processed_cases/` (filesystem count).
  - Benchmark evaluation: **3** fact patterns × 7 variants = 21 runs in `experiment_runs_full_matrix.jsonl`.
- **Average sentences per case**
  - Not currently logged or summarised in the repository. Stage 0 and Stage 1 write per‑case JSONs but no aggregate statistics over sentence counts are computed.
- **Label distribution (FACT/ARGUMENT/REASONING)**
  - Not present: the training JSON for Stage 1 and any corpus‑level label histograms are not checked into this repository.
- **Processing performance**
  - End‑to‑end average latencies are recorded by variant in the metrics summary (see above for aggregate and per‑variant averages).
- **Classification behaviour**
  - Stage 1 sentence classifier is not directly evaluated in the metrics suite; instead, metrics operate at the IPC + sentencing level over judge outputs. However, the configuration and hybrid design of Stage 1 are fully implemented in the codebase and can be instrumented with additional evaluation scripts if desired.

Where metrics are missing (e.g., average sentences per case, Stage 1 label distribution), the current repository does not implement them; they would require additional logging or analysis over Stage 0/Stage 1 JSON outputs.

---

## 9. Design Decisions

Key design choices evident from the code are:

- **Hybrid rules + transformer for sentence roles**
  - Rules act as high‑precision overrides for obvious REASONING/ARGUMENT sentences, while the transformer handles the majority of cases.
  - The classifier always runs the model and only overrides when rule scores exceed a threshold; there is no rules‑only fast path.

- **Separation of concerns across stages**
  - Stage 0 focuses solely on cleaning and splitting, with no classification logic.
  - Stage 1 focuses solely on per‑sentence role labeling.
  - Embedding generation and indexing are separate from the inference backend (`embedding_generation/` vs `backend/`).

- **Retrieval fusion and domain heuristics**
  - FAISS (semantic similarity) and BM25 (lexical overlap) are combined and normalised; domain‑specific bonuses (IPC overlap, homicide/fraud/injury patterns, chunk types) encode legal priors directly in scoring.

- **Judge constrained to party materials**
  - `judge_arguments` never calls `retrieve_precedents` or `retrieve_statutes`; instead, it reconstructs precedents/statutes from citations in the prosecutor/defence paragraphs.
  - All judge prompts clearly instruct the model not to introduce new law beyond what parties rely on.

- **Robustness and safety checks around LLM calls**
  - All three agents normalise outputs to a single paragraph, enforce minimal length and number of sentences, and require explicit sentencing and compensation patterns.
  - On API failure or truncated/incomplete outputs, each agent falls back to a deterministic, template‑driven argument or judgment anchored to retrieved IPC sections and precedents.

---

## 10. Challenges (Code‑Inferred)

The implementation surfaces several domain and engineering challenges:

- **Noisy legal text**
  - Headers, citations, appearance lists, and procedural boilerplate are pervasive in PDFs and `.txt` judgments; Stage 0 and `embedding_generation/parser.py` contain numerous regexes to strip these reliably.

- **Sentence splitting complexity**
  - Legal texts contain dense abbreviations (`v.`, `u/s.`, initials, law‑report citations) and structured lists. The splitter’s placeholder substitution and long‑sentence splitting logic are explicitly designed to avoid over‑splitting on these constructs.

- **Classification ambiguity**
  - Many sentences plausibly belong to multiple roles; the hybrid classifier reflects this by allowing the model to decide unless rules are strongly confident.

- **Retrieval vs. generation trade‑offs**
  - Metrics show high IPC recall but modest exact judgment accuracy on the benchmark, and the rule‑only mapper outperforming the full system in strict accuracy underlines the tension between expressive narrative generation and tight label matching.

- **Performance and reliability of LLM calls**
  - The code includes explicit handling for rate‑limit/empty responses, multiple variants that may fail (`single_llm_no_rag`, `single_llm_with_rag`), and reasoning about valid vs. invalid rows in metrics.

---

## 11. Future Work (Code‑Driven Suggestions)

The following improvements are natural extensions of the current codebase:

- **Stage 0/Stage 1 analytics**
  - Add scripts that iterate over `data/stage0` and `data/stage1` (when present) to compute average sentences per case, label distribution, and confusion matrices for the classifier.

- **Expanded benchmarks**
  - Generalise the experiment and metrics pipeline to larger, more diverse benchmarks beyond the 3‑case example; this would stabilise ablation comparisons and better reflect retrieval/ judgment performance.

- **Tighter coupling between Stage 1 and agent prompts**
  - Currently, Stage 1 roles are used mainly for structured input summaries; additional prompt features could explicitly highlight, for example, REASONING sentences to the judge.

- **GPU‑aware retrieval**
  - Retrieval currently uses CPU SentenceTransformers; moving heavy retrieval embedding to GPU (when available) could reduce latency on larger corpora.

- **Direct evaluation of sentence‑role classification**
  - Extend `metrics/evaluator.py` or a sibling module to ingest labeled sentence datasets and compute per‑label precision/recall/F1 for Stage 1.

---

## 12. Setup and Usage (from Code)

Minimal setup inferred from `requirements.txt` and the backend LLM wrapper:

- **Environment**
  - Python environment with `faiss-cpu`, `numpy`, `sentence-transformers`, `torch`, `google-generativeai`, and `python-dotenv` installed.
  - `backend/.env` (or equivalent) providing `GEMINI_API_KEY` for Gemini access.

- **Typical run sequence**
  1. Generate `processed_cases/` from PDFs via `python -m embedding_generation.tryVec`.
  2. Build retrieval indexes via:
     - `python backend/retrieval/index_precedents.py`
     - `python backend/retrieval/index_ipc.py`
  3. Import and call `run_case` from `backend/prosecution_pipeline.py` in your application or notebook.

This report reflects only what is implemented in the repository at the time of analysis; no additional behaviour is assumed beyond the code and data described above.

