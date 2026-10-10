"""Sentence-Transformers and serverless embeddings. L2-normalised => inner product == cosine.

Supports:
  1. Local SentenceTransformer (when sentence-transformers is installed)
  2. OpenAI text-embedding-3-small (when OPENAI_API_KEY is available or configured)
  3. Hugging Face Serverless Inference API (sentence-transformers/all-MiniLM-L6-v2)
  4. Fast deterministic subword hash embedder (pure NumPy fallback, 0 network, 0 external deps)
"""
from __future__ import annotations

import logging
import os
import re
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class HashEmbedder:
    """Deterministic token-hashing embedder. Zero downloads, zero network calls, pure NumPy."""

    DIM = 384

    def __init__(self, dim: int = DIM):
        self.dim = int(dim)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        texts = list(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", (text or "").lower()):
                idx = zlib.crc32(word.encode("utf-8")) % self.dim
                out[i, idx] += 1.0
        if normalize_embeddings and len(out) > 0:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.maximum(norms, 1e-9)
        return out


class HFInferenceEmbedder:
    """Hugging Face free serverless inference API for all-MiniLM-L6-v2."""

    DIM = 384

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.dim = self.DIM
        self.api_url = f"https://router.huggingface.co/hf-inference/models/sentence-transformers/{model_name}"

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        import json
        import urllib.request

        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        if not token:
            return HashEmbedder(dim=self.dim).encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
            )

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token.strip()}"}
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = json.dumps({"inputs": batch, "options": {"wait_for_model": True}}).encode("utf-8")
            req = urllib.request.Request(self.api_url, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    batch_arr = np.asarray(data, dtype=np.float32)
                    if batch_arr.ndim == 3:
                        batch_arr = np.mean(batch_arr, axis=1)
                    vectors.extend(batch_arr)
            except Exception as exc:
                logger.warning("HF Inference API request failed (%s); falling back to hash embedder", exc)
                fallback = HashEmbedder(dim=self.dim).encode(batch, normalize_embeddings=normalize_embeddings)
                vectors.extend(fallback)

        arr = np.asarray(vectors, dtype=np.float32)
        if normalize_embeddings and len(arr) > 0:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-9)
        return arr


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small embedder."""

    DIM = 1536

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.dim = self.DIM

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        from openai import OpenAI
        from utils.llm import resolve_api_key

        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        key = resolve_api_key(self.api_key)
        client = OpenAI(api_key=key, timeout=30)
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(model=self.model, input=batch)
            for item in resp.data:
                vectors.append(item.embedding)

        arr = np.asarray(vectors, dtype=np.float32)
        if normalize_embeddings and len(arr) > 0:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-9)
        return arr


def load_embedding_model(
    model_name: str = MODEL_NAME,
    cache_dir: Path = MODELS_DIR,
    api_key: str | None = None,
):
    """Load the embedding model based on available environment and dependencies.

    Priority:
      1. Explicit EMBEDDING_PROVIDER environment variable ("sentence-transformers", "openai", "huggingface", "hash")
      2. sentence-transformers (if installed and not on serverless-only mode)
      3. OpenAI text-embedding-3-small (if OPENAI_API_KEY is available)
      4. Hugging Face Serverless Inference API (free all-MiniLM-L6-v2)
      5. Pure-Python HashEmbedder (reliable zero-dependency fallback)
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "").lower().strip()

    if provider == "hash":
        return HashEmbedder()

    if provider == "openai":
        return OpenAIEmbedder(api_key=api_key)

    if provider == "huggingface":
        return HFInferenceEmbedder(model_name=model_name)

    # If provider is explicitly sentence-transformers or unspecified, check if installed:
    if provider in {"", "sentence-transformers", "local"}:
        try:
            from sentence_transformers import SentenceTransformer

            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            return SentenceTransformer(model_name, cache_folder=str(cache_dir))
        except (ImportError, Exception) as exc:
            if provider == "sentence-transformers":
                logger.warning("Could not load sentence-transformers (%s); falling back", exc)

    # In serverless environments without PyTorch:
    from utils.llm import resolve_api_key

    resolved_key = resolve_api_key(api_key)
    if resolved_key and not resolved_key.startswith("gsk_"):
        try:
            return OpenAIEmbedder(api_key=api_key)
        except Exception:
            pass

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if token:
        return HFInferenceEmbedder(model_name=model_name)

    return HashEmbedder()


def embedding_dim(model) -> int:
    getter = (
        getattr(model, "get_embedding_dimension", None)
        or getattr(model, "get_sentence_embedding_dimension", None)
    )
    if getter is not None:
        return int(getter())
    return int(getattr(model, "dim", getattr(model, "DIM", 384)))


def embed_texts(model, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """Return a float32 array of shape (len(texts), dim) with unit-length rows."""
    texts = list(texts)
    if not texts:
        return np.empty((0, embedding_dim(model)), dtype=np.float32)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(vectors, dtype=np.float32)


def embed_query(model, query: str) -> np.ndarray:
    return embed_texts(model, [query])

