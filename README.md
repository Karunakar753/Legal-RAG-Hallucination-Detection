# Casebrief — Legal Judgment Analysis

Casebrief analyzes an uploaded legal judgment with a React interface and a Python API. It retrieves passages from the uploaded PDF, gives page-referenced evidence to an OpenAI-compatible model, and provides a heuristic support check for the generated answer.

> **Not legal advice.** This is an educational/research tool. Machine-generated analysis may be incomplete or wrong. Verify every statement against the judgment and consult a qualified lawyer for legal matters.

## What it does

- Uploads searchable judgment PDFs (up to 25 MB); extracts text and keeps page numbers.
- Creates page-aware chunks and searches them with `all-MiniLM-L6-v2` and FAISS.
- Answers from retrieved passages only, separating submissions from court findings.
- For broad analysis, covers the procedural background, issues, expressly named Acts/provisions, arguments, reasoning, and holding/order where the passages support it.
- Cites source pages, displays retrieved passages, and flags potentially unsupported claims and invalid page citations.
- Keeps PDFs and indexes in server memory only. Up to eight indexed documents are retained per API worker; restarting the server or cache eviction removes them. The API key entered in the UI is sent only with a query and is not stored by the frontend.

The support score and provision checks are heuristics, not proof that an answer is true, complete, or legally correct. The system cannot verify a judgment against external law or legal sources. Scanned/image-only PDFs need OCR before upload.

## Requirements

- Python 3.10–3.12
- Node.js 18 or newer and npm
- An OpenAI or Groq API key for generated answers (retrieval and evidence display work without one)

## Setup

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set either `OPENAI_API_KEY` or `GROQ_API_KEY`. A model can be set using `OPENAI_MODEL` or `GROQ_MODEL`. Alternatively, enter an API key under **Answer settings** in the UI; that value is kept only in browser memory and sent to the API for each question.

Install frontend dependencies:

```powershell
Set-Location frontend
npm install
Set-Location ..
```

The sentence-transformer model (~90 MB) downloads the first time a judgment is indexed.

## Run locally

Open two PowerShell terminals from the project root.

Terminal 1 — API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — React:

```powershell
Set-Location frontend
npm run dev
```

Open <http://127.0.0.1:5173>. The API health endpoint is <http://127.0.0.1:8000/api/health> and interactive API documentation is available at <http://127.0.0.1:8000/docs>.

## Deploy the whole project to Vercel (No background runners or Render needed)

This repository is configured as a unified full-stack application that deploys entirely onto Vercel:
- **Frontend**: Built automatically by Vite into `frontend/dist` and served through Vercel's global CDN.
- **Backend**: Runs as a Python Serverless Function (`api/index.py`) using `@vercel/python`.
- **Zero background runners**: No Render, Railway, or external containers required.
- **Serverless-ready RAG**: Vector search runs with pure NumPy linear algebra, and embeddings use serverless-optimized providers (OpenAI, free Hugging Face Serverless Inference, or fast deterministic hash) to stay strictly within Vercel's 250 MB function size limit.
- **Stateless resilience**: Document embeddings and chunks are preserved across ephemeral serverless invocations via client index serialization and local `/tmp` caching.

### Deploying via Vercel Dashboard (GitHub integration)

1. Push this repository to GitHub.
2. In the [Vercel Dashboard](https://vercel.com/new), import the repository.
3. Keep the **Root Directory** as `./` (do **not** select `frontend`).
4. Vercel automatically uses the settings in `vercel.json`:
   - Build Command: `npm --prefix frontend run build`
   - Output Directory: `frontend/dist`
5. In **Environment Variables**, configure:
   - `OPENAI_API_KEY`: Your OpenAI API key (or `GROQ_API_KEY`).
   - `OPENAI_MODEL`: (Optional, default `gpt-4o-mini`).
   - `EMBEDDING_PROVIDER`: (Optional: `openai`, `huggingface`, or `hash`. Defaults to `openai` if `OPENAI_API_KEY` is present, or Hugging Face Inference API).
6. Click **Deploy**. Both the React UI and the Python `/api` endpoints will be live on the same domain with zero configuration!

### Deploying via Vercel CLI

```bash
npm install -g vercel
vercel
```


## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Architecture

```text
React (frontend/) → FastAPI (api.py) → PDF text/pages → chunks → embeddings → in-memory FAISS
                                                                  ↓
Question → retrieve relevant passages → evidence-only answer with page citations
                                                    ↓
                               claim support heuristics + cited source excerpts
```

The retrieval threshold, chunk sizes, and support score thresholds are heuristic settings. Broad questions invite a structured judgment summary, but the model must state when the retrieved text does not identify an Act or provision. The app does not independently establish that a court applied the correct law.

## Project layout

```text
├── api.py                  # FastAPI upload/query API
├── app.py                  # PDF indexing and question-answer pipeline
├── frontend/               # React + Vite client
├── utils/
│   ├── pdf_processor.py    # In-memory PDF text extraction and page numbers
│   ├── chunker.py          # Page-aware chunking
│   ├── embeddings.py       # MiniLM embeddings
│   ├── vector_store.py     # In-memory FAISS index
│   ├── retriever.py        # Ranked evidence and sufficiency check
│   ├── llm.py              # OpenAI/Groq-compatible generation
│   ├── prompts.py          # Evidence-only legal analysis instructions
│   └── hallucination.py    # Heuristic claim support checks
└── tests/                  # Pipeline and API tests
```
