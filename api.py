"""HTTP API for the React legal-judgment analysis frontend."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import build_index, run_query
from utils.pdf_processor import PDFProcessingError
from utils.retriever import MAX_TOP_K, MIN_TOP_K
from utils.vector_store import VectorStore

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DOCUMENTS = 8
TMP_CACHE_DIR = Path(tempfile.gettempdir()) / "legal_rag_cache"

app = FastAPI(title="Legal Judgment Analysis API", version="1.0.0")
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
# In serverless environments, allow same-origin and localhost
if "*" not in allowed_origins:
    allowed_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

_documents: OrderedDict[str, dict] = OrderedDict()
_documents_lock = threading.Lock()


def _save_tmp_cache(document_id: str, doc: dict) -> None:
    try:
        TMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        store_dict = doc["store"].to_dict()
        cache_data = {
            "filename": doc["filename"],
            "stats": doc["stats"],
            "store": store_dict,
        }
        (TMP_CACHE_DIR / f"{document_id}.json").write_text(
            json.dumps(cache_data, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Could not write serverless tmp cache for %s: %s", document_id, exc)


def _load_tmp_cache(document_id: str) -> dict | None:
    try:
        path = TMP_CACHE_DIR / f"{document_id}.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        store = VectorStore.from_dict(data["store"])
        return {
            "filename": data["filename"],
            "stats": data["stats"],
            "store": store,
        }
    except Exception as exc:
        logger.warning("Could not read serverless tmp cache for %s: %s", document_id, exc)
        return None


def _delete_tmp_cache(document_id: str) -> None:
    try:
        path = TMP_CACHE_DIR / f"{document_id}.json"
        if path.is_file():
            path.unlink(missing_ok=True)
    except Exception:
        pass


class QueryRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=MIN_TOP_K, le=MAX_TOP_K)
    api_key: str | None = Field(default=None, max_length=500)
    model: str | None = Field(default=None, max_length=200)
    index_data: dict | None = None


def _public_document(document_id: str, doc: dict) -> dict:
    data = {
        "document_id": document_id,
        "filename": doc["filename"],
        "stats": doc["stats"],
    }
    if "store" in doc and hasattr(doc["store"], "to_dict"):
        data["index_data"] = doc["store"].to_dict()
    return data


@app.get("/health")
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents", status_code=201)
@app.post("/api/documents", status_code=201)
async def upload_document(file: Annotated[UploadFile, File()]) -> dict:
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Upload a PDF judgment.")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 25 MB upload limit.")

    try:
        doc = build_index(data, filename, 1000, 200)
    except PDFProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to index uploaded judgment")
        raise HTTPException(status_code=500, detail="Could not process this judgment.") from exc
    finally:
        await file.close()

    document_id = uuid.uuid4().hex
    with _documents_lock:
        _documents[document_id] = doc
        while len(_documents) > MAX_DOCUMENTS:
            _documents.popitem(last=False)
    _save_tmp_cache(document_id, doc)
    return _public_document(document_id, doc)


@app.post("/query")
@app.post("/api/query")
def query_judgment(request: QueryRequest) -> dict:
    with _documents_lock:
        doc = _documents.get(request.document_id)
        if doc is not None:
            _documents.move_to_end(request.document_id)

    if doc is None:
        doc = _load_tmp_cache(request.document_id)
        if doc is not None:
            with _documents_lock:
                _documents[request.document_id] = doc

    if doc is None and request.index_data is not None:
        try:
            store = VectorStore.from_dict(request.index_data)
            doc = {
                "filename": "judgment.pdf",
                "stats": {"chunks": len(store.chunks), "pages": max((c.page for c in store.chunks), default=1)},
                "store": store,
            }
            with _documents_lock:
                _documents[request.document_id] = doc
                _save_tmp_cache(request.document_id, doc)
        except Exception as exc:
            logger.warning("Failed to restore doc from index_data: %s", exc)

    if doc is None:
        raise HTTPException(
            status_code=404,
            detail="Judgment not found. It may have expired after a server restart or cache eviction; upload it again.",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    try:
        result = run_query(doc, question, request.top_k, request.api_key, request.model)
    except Exception as exc:
        logger.exception("Failed to answer question from uploaded judgment")
        raise HTTPException(status_code=500, detail="Could not analyze this question.") from exc

    report = asdict(result["report"]) if result["report"] is not None else None
    if report is not None:
        report["unsupported_claims"] = [
            asdict(claim) for claim in result["report"].unsupported_claims
        ]
    return {
        "question": result["question"],
        "answer": result["answer"],
        "error": result["error"],
        "notice": result["notice"],
        "report": report,
        "evidence": [
            {
                "rank": item.rank,
                "page": item.page,
                "score": item.score,
                "text": item.text,
            }
            for item in result["evidence"]
        ],
    }


@app.delete("/documents/{document_id}", status_code=204)
@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str) -> None:
    _delete_tmp_cache(document_id)
    with _documents_lock:
        if _documents.pop(document_id, None) is None:
            raise HTTPException(status_code=404, detail="Judgment not found.")

