# Court RAG Project

Court RAG is an Indian criminal-law reasoning pipeline with three agents:
- prosecutor
- defence
- judge

The system builds retrieval context from precedents and IPC statutes for prosecution/defence, then the judge issues a final specific verdict using only what those two agents provide.

## Clean Repository Layout

### `backend/` (online inference)
- `prosecution_pipeline.py`
  - main APIs: `prosecution`, `defence_arguments`, `judge_arguments`, `run_case`
  - prompt construction, output normalization, sentencing/compensation specificity checks
  - orchestrates prosecutor → defence → judge flow
- `utils/llm.py`
  - Gemini wrapper, model fallback handling, request config
- `retrieval/`
  - `precedent_retriever.py`: FAISS + BM25 retrieval over `processed_cases/`
  - `statute_retriever.py`: FAISS + BM25 retrieval over `ipc.json`
  - `hybrid_retriever.py`: score fusion + reranking logic
  - `bm25_index.py`: lightweight BM25 utilities
  - `index_precedents.py`: build precedent index artifacts
  - `index_ipc.py`: build IPC index artifacts

### `embedding_generation/` (offline preprocessing stage)
- `parser.py`: read PDFs, clean text, derive case title
- `extractor.py`: extract legal cues (IPC refs, action/outcome/intent/judgment)
- `chunker.py`: create multi-granular legal chunks for retrieval
- `tryVec.py`: batch processor (`cases/*.pdf` → `processed_cases/*.json`)

### `stage0/` (text preprocessing stage)
- `cleaner.py`: text cleanup rules
- `splitter.py`: sentence splitting
- `pipeline.py`: stage0 orchestration utilities
- `eval_stage0.ipynb`: stage0 evaluation notebook

### `stage1/` (sentence labeling stage)
- `rules.py`: rule-based scoring for legal-role labels
- `classifier.py`: hybrid classifier entry
- `model.py`: transformer model load/inference helpers
- `dataset.py`: training data utilities
- `trainer.py`: training script
- `pipeline.py`: stage1 orchestration

### Data/runtime folders
- `cases/`: raw PDF judgments
- `processed_cases/`: parsed + chunked precedent JSONs
- `ipc.json`: IPC knowledge base
- `vector_db/`: persisted FAISS/BM25 indexes
- `prosecution.ipynb`: interactive testing notebook

## End-to-End Pipeline

1. **Offline: embedding generation**
   - Convert source PDFs to structured precedent JSONs.
2. **Offline: index build**
   - Build precedent and IPC retrieval indexes into `vector_db/`.
3. **Online: agent reasoning**
   - **Prosecutor**: retrieves precedents/statutes and argues specific IPC charges + sentence + compensation.
   - **Defence**: responds to prosecutor with mitigation/lesser-offence strategy and specific sentencing relief.
   - **Judge**: no fresh retrieval; analyzes only statutes/precedents cited by prosecutor and defence and issues final verdict.

## Judge Agent Constraint (Implemented)

- Judge agent does **not** call precedent/statute retrievers.
- It extracts legal anchors directly from prosecutor and defence outputs.
- Final output is specific:
  - final IPC section(s)
  - exact sentence (death/life/years)
  - compensation amount in INR when applicable

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `backend/.env`:

```env
GEMINI_API_KEY=your_key_here
```

## Run Commands

### 1) Build processed precedents from PDFs

```powershell
python -m embedding_generation.tryVec
```

### 2) Build retrieval indexes

```powershell
python .\backend\retrieval\index_precedents.py
python .\backend\retrieval\index_ipc.py
```

### 3) Run full agent pipeline

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("backend").resolve()))
from prosecution_pipeline import run_case

result = run_case("A man stabbed another person with a knife causing death")
print(result["prosecution"])
print(result["defence"])
print(result["judge"])
```

## Notebook Test Flow

1. Open `prosecution.ipynb`.
2. Run the helper-definition cell first.
3. Run any `run_case(...)` cell.
4. Verify returned keys include: `prosecution`, `defence`, `judge`.

## Cleanup Summary

- Staged embedding scripts are grouped under `embedding_generation/`.
- Empty legacy folder `backend/agents/` removed.
- Source-level `__pycache__/` folders removed.
- Imports are aligned to current staged layout.
