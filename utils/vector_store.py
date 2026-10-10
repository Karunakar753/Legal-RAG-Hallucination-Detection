"""FAISS inner-product index + chunk metadata. In-memory by default; save/load are opt-in."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from utils.chunker import Chunk

VECTORSTORE_DIR = Path(__file__).resolve().parent.parent / "vectorstore"


class VectorStore:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, embeddings: np.ndarray, chunks: Sequence[Chunk]) -> None:
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != self.dim:
            raise ValueError(f"Embeddings must have shape (n, {self.dim}); got {emb.shape}")
        if len(emb) != len(chunks):
            raise ValueError("Number of embeddings and chunks must match")
        self.index.add(emb)
        self.chunks.extend(chunks)

    @classmethod
    def from_chunks(cls, embeddings: np.ndarray, chunks: Sequence[Chunk]) -> "VectorStore":
        emb = np.asarray(embeddings)
        if emb.ndim != 2:
            raise ValueError("Embeddings must be a 2-D array")
        store = cls(emb.shape[1])
        store.add(emb, chunks)
        return store

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        query = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"Query must have dimension {self.dim}; got {query.shape[1]}")
        k = max(1, min(int(top_k), len(self.chunks)))
        scores, ids = self.index.search(query, k)
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def save(self, directory: Path = VECTORSTORE_DIR) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path = VECTORSTORE_DIR) -> "VectorStore":
        directory = Path(directory)
        index = faiss.read_index(str(directory / "index.faiss"))
        chunks = [Chunk.from_dict(d) for d in json.loads((directory / "chunks.json").read_text("utf-8"))]
        store = cls(index.d)
        store.index, store.chunks = index, chunks
        return store
