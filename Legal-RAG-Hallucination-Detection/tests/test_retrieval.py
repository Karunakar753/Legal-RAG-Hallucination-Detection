import pytest

import app
from utils.chunker import chunk_pages
from utils.embeddings import embed_texts
from utils.llm import LLMError, MissingAPIKeyError, generate_answer
from utils.pdf_processor import extract_pages
from utils.retriever import Retriever
from utils.vector_store import VectorStore


@pytest.fixture
def retriever(fake_model, sample_pdf_bytes):
    chunks = chunk_pages(extract_pages(sample_pdf_bytes), 250, 50)
    store = VectorStore.from_chunks(embed_texts(fake_model, [c.text for c in chunks]), chunks)
    return Retriever(fake_model, store)


def test_retrieves_relevant_page_first(retriever):
    hits = retriever.retrieve("appeal dismissed conviction upheld costs", top_k=3)
    assert len(hits) == 3 and hits[0].rank == 1
    assert hits[0].page == 3 and hits[0].score >= hits[-1].score
    assert retriever.is_sufficient(hits)


def test_top_k_is_clamped_and_empty_question_rejected(retriever):
    assert len(retriever.retrieve("appeal", top_k=99)) <= 10
    assert len(retriever.retrieve("appeal", top_k=0)) == 1
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_irrelevant_question_is_insufficient(retriever):
    assert not retriever.is_sufficient(retriever.retrieve("zebra quantum bicycle nebula", 3))


class _FakeClient:
    def __init__(self, text):
        self.chat = type("C", (), {"completions": self})()
        self.text, self.kwargs = text, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        msg = type("M", (), {"content": self.text})()
        return type("R", (), {"choices": [type("Ch", (), {"message": msg})()]})()


def test_llm_prompt_is_evidence_only(retriever):
    client = _FakeClient("The appeal was dismissed [Page 3].")
    ev = retriever.retrieve("appeal outcome", 2)
    assert generate_answer("appeal outcome", ev, client=client) == "The appeal was dismissed [Page 3]."
    system, user = (m["content"] for m in client.kwargs["messages"])
    assert "ONLY" in system and "Never invent" in system
    assert "[Source 1 | Page" in user and client.kwargs["temperature"] == 0


def test_missing_api_key_is_graceful(retriever, monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("utils.llm.ROOT", tmp_path)  # no .env here
    with pytest.raises(MissingAPIKeyError):
        generate_answer("q", retriever.retrieve("appeal", 1))
    assert issubclass(MissingAPIKeyError, LLMError)


def test_end_to_end_pipeline(fake_model, sample_pdf_bytes, monkeypatch):
    """PDF -> chunks -> FAISS -> retrieval -> (stubbed) LLM -> hallucination check, via the app's own functions."""
    monkeypatch.setattr(app, "get_embedding_model", lambda: fake_model)
    doc = app.build_index(sample_pdf_bytes, "judgment.pdf", 300, 50)
    assert doc["stats"]["pages"] == 3 and doc["stats"]["chunks"] >= 3

    monkeypatch.setattr(app, "generate_answer", lambda *a, **k: (
        "The appeal was dismissed and the conviction under Section 302 was upheld [Page 3]. "
        "The Court awarded Rs 5 lakh compensation in Sharma v. Union of India [Page 9]."))
    result = app.run_query(doc, "Was the appeal dismissed?", 4, "sk-test", "test-model")
    assert result["error"] is None and result["evidence"][0].page in {1, 2, 3}
    report = result["report"]
    assert report.invalid_citations == [9]
    flagged = " ".join(c.claim for c in report.unsupported_claims)
    assert "Sharma" in flagged and "dismissed and the conviction" not in flagged


def test_run_query_reports_missing_key_without_crashing(fake_model, sample_pdf_bytes, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "get_embedding_model", lambda: fake_model)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("utils.llm.ROOT", tmp_path)
    doc = app.build_index(sample_pdf_bytes, "j.pdf", 300, 50)
    result = app.run_query(doc, "Was the appeal dismissed?", 3, None, None)
    assert "API key" in result["error"] and result["evidence"] and result["report"] is None
