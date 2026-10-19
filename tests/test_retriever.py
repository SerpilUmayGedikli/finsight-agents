from datetime import date

from memory.retriever import Retriever
from memory.vector_store import ScoredChunk, SearchFilters


def make_chunk(index: int, score: float = 0.0) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=f"accession:Item {index}:0",
        text=f"Passage {index}",
        ticker="ACME",
        cik="0000000001",
        accession_number="accession",
        form_type="10-K",
        filing_date=date(2025, 1, 1),
        item=f"Item {index}",
        chunk_index=0,
        source_url="https://example.test/filing",
        score=score,
    )


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2]


class FakeStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def hybrid_search(self, **kwargs):
        self.calls.append(kwargs)
        return self.results[: kwargs["k"]]


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, chunks, top_k):
        self.calls.append((query, chunks, top_k))
        return [
            chunk.model_copy(update={"score": float(index)})
            for index, chunk in enumerate(reversed(chunks[-top_k:]))
        ]


def test_retrieve_embeds_query_and_passes_k_and_filters():
    chunks = [make_chunk(index) for index in range(4)]
    store = FakeStore(chunks)
    embedder = FakeEmbedder()
    filters = SearchFilters(ticker="ACME", item="Item 1")

    results = Retriever(store, embedder).retrieve("risk factors", k=2, filters=filters)

    assert [chunk.chunk_id for chunk in results] == [
        chunk.chunk_id for chunk in chunks[:2]
    ]
    assert embedder.queries == ["risk factors"]
    assert store.calls[0]["k"] == 2
    assert store.calls[0]["filters"] is filters


def test_reranking_fetches_twenty_and_changes_order_and_respects_k():
    chunks = [make_chunk(index) for index in range(10)]
    store = FakeStore(chunks)
    reranker = FakeReranker()

    results = Retriever(store, FakeEmbedder(), reranker).retrieve(
        "competition", k=3, rerank=True
    )

    assert store.calls[0]["k"] == 20
    assert [chunk.chunk_id for chunk in results] == [
        "accession:Item 9:0",
        "accession:Item 8:0",
        "accession:Item 7:0",
    ]
    assert len(results) == 3
    assert reranker.calls[0][2] == 3


def test_rerank_requested_without_reranker_raises():
    try:
        Retriever(FakeStore([]), FakeEmbedder()).retrieve("query", rerank=True)
    except ValueError as error:
        assert "requires a Reranker" in str(error)
    else:
        raise AssertionError("Expected a missing-reranker error")
