"""Sentence-Transformers embeddings (all-MiniLM-L6-v2). L2-normalised => inner product == cosine."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_embedding_model(model_name: str = MODEL_NAME, cache_dir: Path = MODELS_DIR):
    """Load (downloading on first use) the model. Wrap with st.cache_resource in the UI layer."""
    from sentence_transformers import SentenceTransformer

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return SentenceTransformer(model_name, cache_folder=str(cache_dir))


def embedding_dim(model) -> int:
    getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return int(getter())


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
