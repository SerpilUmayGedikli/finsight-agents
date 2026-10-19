from fastapi.testclient import TestClient

from apps.api.errors import APIError
from apps.api.main import create_app
from apps.api.settings import get_settings


def test_health_returns_ok():
    response = TestClient(create_app()).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_docs_is_available():
    response = TestClient(create_app()).get("/docs")

    assert response.status_code == 200


def test_settings_read_environment_and_are_cached(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("ANTHROPIC_MODEL", "test-model")
    monkeypatch.setenv("RETRIEVAL_K", "9")
    monkeypatch.setenv("RERANK_DEFAULT", "true")

    settings = get_settings()

    assert settings.anthropic_model == "test-model"
    assert settings.retrieval_k == 9
    assert settings.rerank_default is True
    assert get_settings() is settings
    get_settings.cache_clear()


def test_request_id_is_echoed_and_generated():
    client = TestClient(create_app())
    echoed = client.get("/health", headers={"X-Request-ID": "trace-123"})
    generated = client.get("/health")

    assert echoed.headers["X-Request-ID"] == "trace-123"
    assert generated.headers["X-Request-ID"]
    assert generated.headers["X-Request-ID"] != "trace-123"


def test_unknown_route_returns_json_error_with_request_id():
    response = TestClient(create_app()).get(
        "/does-not-exist",
        headers={"X-Request-ID": "missing-route"},
    )

    assert response.status_code == 404
    assert response.json()["error"] == {
        "code": "not_found",
        "message": "Not Found",
        "request_id": "missing-route",
    }


def test_custom_exception_returns_json_error_shape():
    app = create_app()

    @app.get("/_test/error")
    async def raise_api_error():
        raise APIError("example_error", "An example failure", status_code=409)

    response = TestClient(app).get("/_test/error")

    assert response.status_code == 409
    assert response.json()["error"] == {
        "code": "example_error",
        "message": "An example failure",
        "request_id": response.headers["X-Request-ID"],
    }
