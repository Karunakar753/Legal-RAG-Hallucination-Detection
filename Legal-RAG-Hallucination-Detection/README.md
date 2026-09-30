# ⚖️ Legal RAG with Hallucination Detection

A Streamlit app that answers questions about a **legal judgment PDF** using Retrieval-Augmented Generation, shows the **evidence with page numbers**, and runs a **hallucination check** that flags potentially unsupported claims.

> **Disclaimer:** This is an educational/research tool, **not legal advice**. Verify every answer against the source judgment.

## Features
- Upload a judgment PDF → text + page numbers (PyMuPDF), page-aware chunking (LangChain splitter)
- Embeddings with `all-MiniLM-L6-v2` (Sentence Transformers), cosine search with FAISS
- LLM (OpenAI) answers **only** from retrieved evidence, cites `[Page N]`, and says so when evidence is insufficient
- Hallucination check: per-claim support score, flags invented case names / provisions / numbers / dates, checks that cited pages really were retrieved
- Graceful handling of invalid, empty, encrypted and scanned PDFs, and of a missing/invalid API key
- PDFs are processed **in memory only** - never written to disk

## Setup & Installation
Requires Python 3.10-3.12.

```powershell
git clone https://github.com/<your-username>/Legal-RAG-Hallucination-Detection.git
cd Legal-RAG-Hallucination-Detection
python -m venv venv
.\venv\Scripts\Activate.ps1          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env               # macOS/Linux: cp .env.example .env
```
Edit `.env` and set `OPENAI_API_KEY`. The embedding model (~90 MB) downloads into `models/` on first use.

## Run
```powershell
python -m streamlit run app.py
```
Tests: `python -m pytest` (the real-model test is skipped automatically when the model can't be downloaded).

## API-key configuration
Priority order: sidebar field (session only) → environment / `.env` → Streamlit Secrets. Keys are never hard-coded.

| Where | How |
|---|---|
| Local | `.env` → `OPENAI_API_KEY=sk-...` (optional `OPENAI_MODEL=gpt-4o-mini`) |
| Streamlit Cloud | App → Settings → **Secrets** → `OPENAI_API_KEY = "sk-..."` |
| Quick test | Paste into the sidebar field |

`.env` and `.streamlit/secrets.toml` are git-ignored; commit only `.env.example` / `.streamlit/secrets.toml.example`.

## Architecture
```text
PDF → Text (per page) → Chunks (page kept) → Embeddings → FAISS (in memory)
                                                             ↓
Question → Embedding → Top-K retrieval → Evidence (+ pages, similarity)
                                             ↓ (if best similarity ≥ 0.20, else "insufficient")
                                        LLM answer (evidence only, [Page N] citations)
                                             ↓
                            Hallucination check → support score + unsupported claims
```
**Hallucination check** (`utils/hallucination.py`): the answer is split into claims; each is scored by 0.6 × semantic similarity to evidence sentences + 0.4 × content-word coverage. Case names, legal provisions and numbers/dates must literally appear in the evidence, and cited pages must be among the retrieved pages - otherwise the claim's score is halved and it is flagged. Thresholds (`SUPPORT_THRESHOLD`, `SEM_LOW/HIGH`, `MIN_RELEVANCE`) are heuristics you can tune.

## Folder structure
```text
Legal-RAG-Hallucination-Detection/
├── app.py                  # Streamlit UI + pipeline orchestration
├── requirements.txt  .env.example  .gitignore  pytest.ini
├── .streamlit/             # config.toml, secrets.toml.example
├── data/documents/         # (optional local samples; git-ignored)
├── data/processed/         # (optional; git-ignored)
├── models/                 # embedding model cache (git-ignored)
├── vectorstore/            # optional persisted FAISS index (git-ignored, off by default)
├── utils/
│   ├── pdf_processor.py    # PyMuPDF extraction, scanned/invalid PDF errors
│   ├── chunker.py          # page-aware chunking
│   ├── embeddings.py       # MiniLM embeddings
│   ├── vector_store.py     # FAISS index (+ optional save/load)
│   ├── retriever.py        # top-K retrieval + sufficiency check
│   ├── llm.py              # OpenAI call, key resolution, error handling
│   ├── hallucination.py    # unsupported-claim detection
│   └── prompts.py          # system prompt, disclaimer
└── tests/                  # pytest suite (+ conftest.py fixtures)
```

## GitHub deployment
```powershell
git init
git add .
git commit -m "Legal RAG hallucination detection app"
git branch -M main
git remote add origin https://github.com/<your-username>/Legal-RAG-Hallucination-Detection.git
git push -u origin main
```
Check `git status` first: `.env`, `venv/` and `.streamlit/secrets.toml` must not appear.

## Streamlit Cloud deployment
1. Push the repo to GitHub.
2. Go to <https://share.streamlit.io> → **New app** → pick the repo, branch `main`, main file `app.py`.
3. **Advanced settings** → choose Python 3.11 or 3.12 and paste your secrets: `OPENAI_API_KEY = "sk-..."`.
4. Deploy. The first launch installs dependencies (PyTorch is large) and downloads the embedding model on first upload.

## Limitations
- **Not legal advice**; output may be wrong or incomplete.
- The hallucination check is a **heuristic**: it can miss subtle misstatements (e.g. reversed holdings) and can flag correct paraphrases. It does not verify facts against the outside world.
- Scanned/image-only PDFs are rejected (no OCR built in). Multi-column layouts and tables may extract imperfectly.
- Chunks never cross page boundaries, so reasoning that spans pages may be split across passages.
- Only the top-K passages are seen by the LLM; a small K can miss relevant text, a large K adds noise.
- Answers depend on the OpenAI model chosen and require network access and API credit.
- Streamlit Cloud free tier has limited memory; very large judgments may be slow.
