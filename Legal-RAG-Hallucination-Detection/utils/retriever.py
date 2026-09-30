"""Question -> embedding -> FAISS search -> ranked evidence with page numbers."""
from __future__ import annotations

from dataclasses import dataclass

from utils.chunker import Chunk
from utils.embeddings import embed_query
from utils.vector_store import VectorStore

MIN_TOP_K, MAX_TOP_K = 1, 10
MIN_RELEVANCE = 0.20  # best cosine similarity below this => evidence deemed insufficient


@dataclass(frozen=True)
class Evidence:
    rank: int  # 1-based
    chunk: Chunk
    score: float  # cosine similarity

    @property
    def page(self) -> int:
        return self.chunk.page

    @property
    def text(self) -> str:
        return self.chunk.text


class Retriever:
    def __init__(self, model, store: VectorStore, min_relevance: float = MIN_RELEVANCE):
        self.model = model
        self.store = store
        self.min_relevance = min_relevance

    def retrieve(self, question: str, top_k: int = 5) -> list[Evidence]:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")
        top_k = max(MIN_TOP_K, min(MAX_TOP_K, int(top_k)))
        hits = self.store.search(embed_query(self.model, question), top_k)
        return [Evidence(i, chunk, score) for i, (chunk, score) in enumerate(hits, start=1)]

    def is_sufficient(self, evidence: list[Evidence]) -> bool:
        return bool(evidence) and max(e.score for e in evidence) >= self.min_relevance
