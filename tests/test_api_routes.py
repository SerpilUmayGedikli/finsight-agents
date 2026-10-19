import json
from datetime import date

from fastapi.testclient import TestClient

from apps.api.dependencies import (
    get_embedder,
    get_llm_client,
    get_settings,
    get_vector_store,
)
from apps.api.main import create_app
from apps.api.settings import Settings
from gateway.llm import FakeLLMClient
from memory.vector_store import ScoredChunk


def make_chunk():
    return ScoredChunk(
        chunk_id="chunk-1",
        text="Revenue grew 12 percent because services expanded.",
        ticker="ACME",
        cik="0000000001",
        accession_number="accession-1",
        form_type="10-K",
        filing_date=date(2025, 1, 1),
        item="Item 7",
        chunk_index=0,
        source_url="https://example.test/filing",
        score=0.91,
    )


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def hybrid_search(self, **kwargs):
        self.calls.append(kwargs)
        return self.chunks[: kwargs["k"]]


class FakeEmbedder:
    def embed_query(self, query):
        return [0.1, 0.2]


def make_client(chunks, llm):
    settings = Settings(anthropic_api_key="test-key", retrieval_k=3)
    store = FakeStore(chunks)
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_embedder] = FakeEmbedder
    app.dependency_overrides[get_llm_client] = lambda: llm
    return TestClient(app), store


def test_search_returns_chunk_fields_and_passes_filters():
    client, store = make_client([make_chunk()], FakeLLMClient(["unused"]))

    response = client.post(
        "/search",
        json={"query": "revenue", "k": 2, "filters": {"ticker": "ACME"}},
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result == {
        "chunk_id": "chunk-1",
        "score": 0.91,
        "text": "Revenue grew 12 percent because services expanded.",
        "ticker": "ACME",
        "form_type": "10-K",
        "item": "Item 7",
        "filing_date": "2025-01-01",
        "source_url": "https://example.test/filing",
    }
    assert store.calls[0]["filters"].ticker == "ACME"
    assert store.calls[0]["k"] == 2


def test_ask_happy_path_returns_answer_and_retrieved_chunks():
    response_body = {
        "answer": "Revenue grew 12 percent.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Revenue grew 12 percent"}],
        "insufficient_context": False,
    }
    llm = FakeLLMClient([json.dumps(response_body)])
    client, store = make_client([make_chunk()], llm)

    response = client.post("/ask", json={"question": "How did revenue change?"})

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "Revenue grew 12 percent."
    assert result["citations"] == [
        {"chunk_id": "chunk-1", "quote": "Revenue grew 12 percent", "verified": True}
    ]
    assert result["retrieved_chunks"][0]["chunk_id"] == "chunk-1"
    assert result["model"] == "fake"
    assert result["attempts"] == 1
    assert result["insufficient_context"] is False
    assert store.calls[0]["k"] == 3


def test_ask_empty_retrieval_does_not_call_llm():
    llm = FakeLLMClient(["must not be used"])
    client, _ = make_client([], llm)

    response = client.post("/ask", json={"question": "Unknown fact?"})

    assert response.status_code == 200
    assert response.json()["insufficient_context"] is True
    assert response.json()["attempts"] == 0
    assert response.json()["retrieved_chunks"] == []
    assert llm.calls == []


def test_ask_parse_failure_returns_502_standard_error():
    llm = FakeLLMClient(["bad", "still bad", "also bad"])
    client, _ = make_client([make_chunk()], llm)

    response = client.post(
        "/ask",
        json={"question": "How did revenue change?"},
        headers={"X-Request-ID": "ask-parse-failure"},
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "answer_parse_error"
    assert response.json()["error"]["request_id"] == "ask-parse-failure"


def test_ingest_job_lifecycle_uses_background_pipeline(monkeypatch):
    import apps.api.routes.ingest as ingest_route

    app = create_app()
    observed_statuses = []

    def fake_pipeline(tickers, form, limit, progress_callback):
        job = next(iter(app.state.ingest_jobs.values()))
        observed_statuses.append(job["status"])
        assert tickers == ["ACME"]
        assert form == "10-K"
        assert limit == 2
        progress_callback(1, 1)

    monkeypatch.setattr(ingest_route, "ingest_tickers", fake_pipeline)
    client = TestClient(app)

    created = client.post("/ingest", json={"tickers": ["ACME"]})
    job_id = created.json()["job_id"]
    status = client.get(f"/ingest/{job_id}")

    assert created.status_code == 202
    assert observed_statuses == ["running"]
    assert status.json() == {
        "job_id": job_id,
        "status": "done",
        "progress": 100,
        "error": None,
    }


def test_unknown_ingest_job_returns_standard_404():
    response = TestClient(create_app()).get("/ingest/not-a-job")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]
