"""Legal judgment retrieval and evidence-grounded answer pipeline."""
from __future__ import annotations

import hashlib
from functools import lru_cache

from utils.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_pages
from utils.embeddings import embed_texts, load_embedding_model
from utils.hallucination import check_answer
from utils.llm import LLMError, MissingAPIKeyError, generate_answer
from utils.pdf_processor import PDFProcessingError, extract_pages, get_pdf_stats
from utils.prompts import INSUFFICIENT_EVIDENCE_MESSAGE
from utils.retriever import MAX_TOP_K, MIN_TOP_K, Retriever
from utils.vector_store import VectorStore


def get_embedding_model(api_key: str | None = None):
    """Load the embedding model once per API worker or use request API key."""
    return load_embedding_model(api_key=api_key)


def build_index(
    pdf_bytes: bytes,
    filename: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    api_key: str | None = None,
) -> dict:
    """PDF bytes -> pages -> chunks -> embeddings -> VectorStore; document text stays in memory."""
    pages = extract_pages(pdf_bytes)
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    if not chunks:
        raise PDFProcessingError("No text chunks could be created from this PDF.")
    try:
        model = get_embedding_model(api_key)
    except TypeError:
        model = get_embedding_model()
    embeddings = embed_texts(model, [chunk.text for chunk in chunks])
    return {
        "key": (hashlib.sha256(pdf_bytes).hexdigest(), chunk_size, chunk_overlap),
        "filename": filename,
        "stats": {**get_pdf_stats(pages), "chunks": len(chunks)},
        "store": VectorStore.from_chunks(embeddings, chunks),
    }


def run_query(
    doc: dict,
    question: str,
    top_k: int,
    api_key: str | None,
    model_name: str | None,
) -> dict:
    """Retrieve passages, answer from those passages only, then score answer support."""
    model = get_embedding_model()
    retriever = Retriever(model, doc["store"])
    evidence = retriever.retrieve(question, top_k)
    result = {
        "question": question,
        "evidence": evidence,
        "answer": None,
        "report": None,
        "error": None,
        "notice": None,
    }

    if not retriever.is_sufficient(evidence):
        best = max((item.score for item in evidence), default=0.0)
        result["answer"] = INSUFFICIENT_EVIDENCE_MESSAGE
        result["notice"] = (
            f"No passage was relevant enough (best similarity {best:.2f}); "
            "the LLM was not called, to avoid an ungrounded answer."
        )
        return result

    try:
        answer = generate_answer(question, evidence, api_key=api_key, model=model_name)
    except (MissingAPIKeyError, LLMError) as exc:
        result["error"] = str(exc)
        return result

    result["answer"] = answer
    result["report"] = check_answer(answer, evidence, lambda texts: embed_texts(model, texts))
    return result
