from argparse import Namespace

from memory import cli


def test_search_cli_passes_reranker_when_requested(monkeypatch):
    store = type("FakeStore", (), {"close": lambda self: None})()
    embedder = object()
    reranker = object()
    captured = {}

    class FakeRetriever:
        def __init__(self, received_store, received_embedder, reranker=None):
            captured["store"] = received_store
            captured["embedder"] = received_embedder
            captured["reranker"] = reranker

        def retrieve(self, query, k, filters, rerank):
            captured["retrieve"] = (query, k, filters, rerank)
            return []

    monkeypatch.setattr(cli, "VectorStore", lambda: store)
    monkeypatch.setattr(cli, "get_embedder", lambda: embedder)
    monkeypatch.setattr(cli, "Reranker", lambda: reranker)
    monkeypatch.setattr(cli, "Retriever", FakeRetriever)

    cli._run_search(Namespace(query="risk factors", k=5, rerank=True, ticker="AAPL"))

    assert captured["store"] is store
    assert captured["embedder"] is embedder
    assert captured["reranker"] is reranker
    query, k, filters, rerank = captured["retrieve"]
    assert (query, k, rerank) == ("risk factors", 5, True)
    assert filters.ticker == "AAPL"
    assert store is not None


def test_ingest_closes_owned_store_if_embedder_initialization_fails(monkeypatch):
    store = type(
        "FakeStore",
        (),
        {"closed": False, "close": lambda self: setattr(self, "closed", True)},
    )()
    monkeypatch.setattr(cli, "VectorStore", lambda: store)
    monkeypatch.setattr(
        cli, "get_embedder", lambda: (_ for _ in ()).throw(ValueError("missing key"))
    )

    try:
        cli.ingest_tickers(["AAPL"])
    except ValueError as error:
        assert str(error) == "missing key"
    else:
        raise AssertionError("Expected embedder initialization to fail")

    assert store.closed
