import random
from datetime import date

import pytest

from memory.cli import ingest_tickers
from memory.models import Chunk, FilingSection
from memory.vector_store import SearchFilters, VectorStore

pytestmark = pytest.mark.integration


@pytest.fixture
def populated_store():
    store = VectorStore()
    if store._client.collections.exists(store.collection_name):
        store._client.collections.delete(store.collection_name)
    store.ensure_collection()

    chunks = [
        Chunk(
            chunk_id=f"accession-{index}:Item {item}:0",
            text=f"Financial filing passage number {index}.",
            ticker=ticker,
            cik=f"{index:010d}",
            accession_number=f"accession-{index}",
            form_type="10-K",
            filing_date=filing_date,
            item=f"Item {item}",
            chunk_index=0,
            source_url=f"https://example.test/{index}",
        )
        for index, (ticker, item, filing_date) in enumerate(
            [
                ("ACME", "1A", date(2024, 1, 1)),
                ("ACME", "7", date(2024, 6, 15)),
                ("OTHER", "1A", date(2025, 1, 1)),
                ("ACME", "1A", date(2025, 9, 1)),
                ("OTHER", "7", date(2023, 12, 31)),
            ]
        )
    ]
    randomizer = random.Random(12)
    vectors = [[randomizer.random() for _ in range(16)] for _ in chunks]
    store.upsert_chunks(chunks, vectors)

    try:
        yield store, chunks, vectors
    finally:
        store._client.collections.delete(store.collection_name)
        store.close()


def test_upserting_same_chunks_twice_does_not_duplicate(populated_store):
    store, chunks, vectors = populated_store

    assert store.count() == 5
    assert store.upsert_chunks(chunks, vectors) == 5
    assert store.count() == 5


@pytest.mark.parametrize(
    ("filters", "expected_ids"),
    [
        (
            SearchFilters(ticker="ACME"),
            {"accession-0:Item 1A:0", "accession-1:Item 7:0", "accession-3:Item 1A:0"},
        ),
        (
            SearchFilters(item="Item 1A"),
            {"accession-0:Item 1A:0", "accession-2:Item 1A:0", "accession-3:Item 1A:0"},
        ),
        (
            SearchFilters(
                filing_date_from=date(2024, 1, 1), filing_date_to=date(2024, 12, 31)
            ),
            {"accession-0:Item 1A:0", "accession-1:Item 7:0"},
        ),
    ],
)
def test_hybrid_search_filters(populated_store, filters, expected_ids):
    store, _chunks, vectors = populated_store
    results = store.hybrid_search("financial filing", vectors[0], filters=filters)

    assert {result.chunk_id for result in results} == expected_ids


def test_hybrid_search_returns_scored_chunks(populated_store):
    store, _, vectors = populated_store
    results = store.hybrid_search("financial filing", vectors[0], k=5)

    assert results
    assert all(result.score >= 0 for result in results)


def test_reingesting_same_filing_does_not_increase_count(populated_store):
    store, _, _ = populated_store
    section = FilingSection(
        text="A short filing section for ingestion.",
        ticker="ACME",
        cik="0000000042",
        accession_number="ingest-accession",
        form_type="10-K",
        filing_date=date(2025, 1, 1),
        item="Item 1",
        source_url="https://example.test/ingest",
    )

    class FakeEmbedder:
        def embed_documents(self, texts):
            return [[0.25] * 16 for _ in texts]

    fetcher = lambda ticker, form, limit: [object()]
    parser = lambda filing, ticker: [section.model_copy(update={"ticker": ticker})]
    before = store.count()
    ingest_tickers(
        ["ACME"],
        store=store,
        embedder=FakeEmbedder(),
        fetcher=fetcher,
        parser=parser,
    )
    after_first = store.count()
    ingest_tickers(
        ["ACME"],
        store=store,
        embedder=FakeEmbedder(),
        fetcher=fetcher,
        parser=parser,
    )

    assert after_first == before + 1
    assert store.count() == after_first
