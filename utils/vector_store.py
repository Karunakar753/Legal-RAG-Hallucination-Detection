"""FAISS inner-product index + chunk metadata with pure-NumPy fallback for serverless deployments.
In-memory by default; save/load and dict serialization supported.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Sequence

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False

import numpy as np

from utils.chunker import Chunk

VECTORSTORE_DIR = Path(__file__).resolve().parent.parent / "vectorstore"


class VectorStore:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.chunks: list[Chunk] = []
        self._vectors: list[np.ndarray] = []
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = None

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, embeddings: np.ndarray, chunks: Sequence[Chunk]) -> None:
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != self.dim:
            raise ValueError(f"Embeddings must have shape (n, {self.dim}); got {emb.shape}")
        if len(emb) != len(chunks):
            raise ValueError("Number of embeddings and chunks must match")
        if HAS_FAISS and self.index is not None:
            self.index.add(emb)
        self._vectors.append(emb)
        self.chunks.extend(chunks)

    @property
    def embeddings(self) -> np.ndarray:
        if not self._vectors:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack(self._vectors)

    @classmethod
    def from_chunks(cls, embeddings: np.ndarray, chunks: Sequence[Chunk]) -> "VectorStore":
        emb = np.asarray(embeddings, dtype=np.float32)
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
        if HAS_FAISS and self.index is not None:
            scores, ids = self.index.search(query, k)
            return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

        # Pure NumPy cosine similarity fallback (L2-normalised vectors => dot product)
        all_emb = self.embeddings
        scores = np.dot(all_emb, query.reshape(-1))
        top_indices = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def to_dict(self) -> dict:
        emb_bytes = self.embeddings.tobytes()
        return {
            "dim": self.dim,
            "chunks": [c.to_dict() for c in self.chunks],
            "embeddings_b64": base64.b64encode(emb_bytes).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VectorStore":
        dim = int(data["dim"])
        chunks = [Chunk.from_dict(d) for d in data["chunks"]]
        b64 = data.get("embeddings_b64", "")
        if b64:
            raw = base64.b64decode(b64)
            embeddings = np.frombuffer(raw, dtype=np.float32).reshape(len(chunks), dim)
        else:
            embeddings = np.empty((0, dim), dtype=np.float32)
        store = cls(dim)
        if len(chunks) > 0:
            store.add(embeddings, chunks)
        return store

    def save(self, directory: Path = VECTORSTORE_DIR) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, str(directory / "index.faiss"))
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path = VECTORSTORE_DIR) -> "VectorStore":
        directory = Path(directory)
        chunks = [Chunk.from_dict(d) for d in json.loads((directory / "chunks.json").read_text("utf-8"))]
        if HAS_FAISS and (directory / "index.faiss").exists():
            index = faiss.read_index(str(directory / "index.faiss"))
            store = cls(index.d)
            store.index, store.chunks = index, chunks
            if (directory / "embeddings.npy").exists():
                store._vectors = [np.load(directory / "embeddings.npy")]
            return store
        embeddings = np.load(directory / "embeddings.npy")
        store = cls(embeddings.shape[1])
        store.add(embeddings, chunks)
        return store

