
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client():
    return TestClient(api.app)


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["application"] == (
        "Advanced Wikipedia RAG API"
    )


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_empty_question(client):
    response = client.post(
        "/query",
        json={"question": ""},
    )
    assert response.status_code == 422


def test_whitespace_question(client):
    response = client.post(
        "/query",
        json={"question": "   "},
    )
    assert response.status_code == 422


def test_missing_question(client):
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_question_too_long(client):
    response = client.post(
        "/query",
        json={"question": "A" * 1001},
    )
    assert response.status_code == 422


def test_query_success(client, monkeypatch):
    # Mock model inference to keep this test fast.
    def mock_run_rag(question):
        return api.QueryResponse(
            question=question,
            answer="Supervised learning uses labeled data [1].",
            sources=[
                api.SourceResponse(
                    id=1,
                    title="Machine learning",
                    url=(
                        "https://en.wikipedia.org/"
                        "wiki/Machine_learning"
                    ),
                    excerpt="Supervised learning uses labels.",
                    score=0.95,
                )
            ],
            citation_warning=None,
            latency_ms=12.5,
        )

    monkeypatch.setattr(api, "run_rag", mock_run_rag)

    response = client.post(
        "/query",
        json={"question": "Explain supervised learning"},
    )

    assert response.status_code == 200

    data = response.json()
    assert "[1]" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["id"] == 1
    assert "latency_ms" in data


def test_query_failure(client, monkeypatch):
    def mock_failure(question):
        raise RuntimeError("Simulated model failure")

    monkeypatch.setattr(api, "run_rag", mock_failure)

    response = client.post(
        "/query",
        json={"question": "Explain machine learning"},
    )

    assert response.status_code == 503
    assert "RAG inference failed" in response.json()["detail"]
