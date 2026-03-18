# Court RAG Project (Current MVP)

A CPU-only Legal RAG prototype that:
- Retrieves relevant statutes from `data/statutes/`
- Retrieves similar case precedents from `data/cases/`
- Runs simple LLM-based legal agents (Fact Extractor, Prosecution, Defense, Judge)
- Prints a full legal reasoning simulation in `backend/main.py`

---

## Current Features

- **Statute Retriever** (`backend/retrieval/statute_retriever.py`)
  - Embeds statute files using `all-MiniLM-L6-v2`
  - Stores FAISS index in `vector_db/statutes.index`
  - Stores statute metadata in `vector_db/statutes.pkl`

- **Precedent Retriever** (`backend/retrieval/precedent_retriever.py`)
  - Embeds case files using `all-MiniLM-L6-v2`
  - Stores FAISS index in `vector_db/cases.index`
  - Stores case metadata in `vector_db/cases.pkl`

- **LLM + Agent Pipeline**
  - `backend/utils/llm.py`: CPU-only generation wrapper
  - `backend/utils/prompts.py`: prompt templates
  - `backend/agents/`: prosecution, defense, judge agents

- **Main Pipeline** (`backend/main.py`)
  - Extract facts from input case text
  - Retrieve top statutes + precedents
  - Run prosecution, defense, judge
  - Print sections:
    - FACTS
    - STATUTES
    - PRECEDENTS
    - PROSECUTION ARGUMENT
    - DEFENSE ARGUMENT
    - FINAL VERDICT

---

## Project Structure

```text
backend/
  main.py
  agents/
    defense.py
    judge.py
    prosecution.py
  retrieval/
    precedent_retriever.py
    statute_retriever.py
  utils/
    llm.py
    prompts.py

data/
  cases/        # case text files
  statutes/     # statute text files

vector_db/      # generated FAISS indexes + metadata
requirements.txt
```

---

## Setup (for new users)

### 1) Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Run the project

From project root:

```powershell
python backend/main.py
```

On first run, if index files are missing, the app auto-builds:
- `vector_db/statutes.index`
- `vector_db/statutes.pkl`
- `vector_db/cases.index`
- `vector_db/cases.pkl`

Then it runs the legal pipeline and prints top results + agent outputs.

---

## How to use right now

- Edit `INPUT_CASE_TEXT` in `backend/main.py` with your test scenario.
- Run `python backend/main.py`.
- Review retrieved statutes/precedents and final verdict text.

---

## Notes / Limitations (current MVP)

- CPU-only inference (no GPU required).
- LLM output quality depends on lightweight model and prompt quality.
- Retrieval quality depends on dataset text cleanliness and legal domain coverage.
- This is a prototype for research/demo usage, not legal advice.

---

## Troubleshooting

- **Missing model download / slow first run**: first run downloads HuggingFace models.
- **HF rate limit warnings**: set `HF_TOKEN` optionally for better throughput.
- **Windows symlink warning from HF cache**: safe to ignore, but enabling Developer Mode improves cache behavior.
