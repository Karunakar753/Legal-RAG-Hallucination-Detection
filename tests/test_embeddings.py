import numpy as np
import pytest

from utils.embeddings import MODEL_NAME, embed_query, embed_texts
from utils.vector_store import VectorStore
from utils.chunker import Chunk


def test_embed_texts_shape_dtype_and_norm(fake_model):
    emb = embed_texts(fake_model, ["appeal dismissed", "conviction upheld", "costs"])
    assert emb.shape == (3, fake_model.DIM) and emb.dtype == np.float32
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_empty_input_returns_empty_matrix(fake_model):
    assert embed_texts(fake_model, []).shape == (0, fake_model.DIM)


def test_embed_query_is_2d(fake_model):
    assert embed_query(fake_model, "what happened?").shape == (1, fake_model.DIM)


def test_faiss_store_search_save_load(fake_model, tmp_path):
    chunks = [Chunk(0, 1, "murder conviction appeal"), Chunk(1, 2, "costs of the proceedings")]
    store = VectorStore.from_chunks(embed_texts(fake_model, [c.text for c in chunks]), chunks)
    hits = store.search(embed_query(fake_model, "murder appeal"), top_k=5)
    assert hits[0][0].page == 1 and hits[0][1] > hits[1][1]
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 2 and loaded.search(embed_query(fake_model, "costs"), 1)[0][0].page == 2


def test_store_rejects_wrong_dimension(fake_model):
    store = VectorStore(fake_model.DIM)
    with pytest.raises(ValueError):
        store.add(np.zeros((1, 3), dtype=np.float32), [Chunk(0, 1, "x")])


def test_real_minilm_model_if_available():
    """Runs only when the real model can be loaded (needs network on first run)."""
    pytest.importorskip("sentence_transformers")
    from utils.embeddings import load_embedding_model

    try:
        model = load_embedding_model()
    except Exception as exc:
        pytest.skip(f"{MODEL_NAME} unavailable offline: {exc}")
    emb = embed_texts(model, ["The appeal is dismissed.", "The appeal was rejected.", "Bananas are yellow."])
    assert emb.shape == (3, 384)
    assert emb[0] @ emb[1] > emb[0] @ emb[2]
