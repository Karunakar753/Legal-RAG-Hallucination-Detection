from fastapi.testclient import TestClient

import api
import app


def test_health_endpoint():
    response = TestClient(api.app).get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_document_upload_query_and_delete(sample_pdf_bytes, fake_model, monkeypatch):
    monkeypatch.setattr(app, "get_embedding_model", lambda: fake_model)

    def raise_missing_api_key(*args, **kwargs):
        raise app.MissingAPIKeyError("No API key found.")

    monkeypatch.setattr(app, "generate_answer", raise_missing_api_key)
    api._documents.clear()
    client = TestClient(api.app)
    uploaded = client.post(
        "/api/documents",
        files={"file": ("judgment.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert uploaded.status_code == 201
    document = uploaded.json()
    assert document["filename"] == "judgment.pdf"
    assert document["stats"]["pages"] == 3

    answer = client.post(
        "/api/query",
        json={
            "document_id": document["document_id"],
            "question": "Was the appeal dismissed?",
            "top_k": 4,
        },
    )
    assert answer.status_code == 200
    result = answer.json()
    assert result["error"] is not None
    assert result["evidence"]
    assert result["evidence"][0]["page"] in {1, 2, 3}

    deleted = client.delete(f"/api/documents/{document['document_id']}")
    assert deleted.status_code == 204
    missing = client.post(
        "/api/query",
        json={"document_id": document["document_id"], "question": "What was decided?"},
    )
    assert missing.status_code == 404


def test_upload_rejects_non_pdf():
    response = TestClient(api.app).post(
        "/api/documents",
        files={"file": ("judgment.txt", b"not a PDF", "text/plain")},
    )
    assert response.status_code == 415
