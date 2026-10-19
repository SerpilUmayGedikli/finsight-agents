import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

from apps.api.settings import Settings
from memory.chunking import chunk_section
from memory.embeddings import Embedder, get_embedder
from memory.models import Chunk, FilingSection
from memory.reranker import Reranker
from memory.retriever import Retriever
from memory.vector_store import SearchFilters, VectorStore


@dataclass
class IngestStats:
    ticker: str
    filings: int
    sections: int
    chunks: int
    total_objects: int


def fetch_filings(ticker: str, form: str, limit: int):
    identity = Settings().edgar_identity
    if not identity:
        raise RuntimeError(
            "Set EDGAR_IDENTITY to your name and email before fetching SEC filings."
        )

    from edgar import Company, set_identity

    set_identity(identity)
    return list(Company(ticker).get_filings(form=form)[:limit])


def parse_filing(filing, ticker: str) -> list[FilingSection]:
    report = filing.obj()
    items = getattr(report, "items", []) or []
    sections = []
    for item in items:
        text = report.get(item)
        if text and text.strip():
            sections.append(_make_section(filing, ticker, str(item), text))

    if not sections:
        text = filing.text()
        if text and text.strip():
            sections.append(_make_section(filing, ticker, "Filing", text))
    return sections


def _make_section(filing, ticker: str, item: str, text: str) -> FilingSection:
    filing_date = filing.filing_date
    if not isinstance(filing_date, date):
        filing_date = date.fromisoformat(str(filing_date)[:10])
    cik = str(filing.cik).zfill(10)
    return FilingSection(
        text=text,
        ticker=ticker.upper(),
        cik=cik,
        accession_number=filing.accession_no,
        form_type=filing.form,
        filing_date=filing_date,
        item=item,
        source_url=filing.url,
    )


def ingest_tickers(
    tickers: Sequence[str],
    form: str = "10-K",
    limit: int = 2,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    *,
    fetcher: Callable | None = None,
    parser: Callable | None = None,
    batch_size: int = 32,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[IngestStats]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    owns_store = store is None
    store = VectorStore() if store is None else store
    fetcher = fetcher or fetch_filings
    parser = parser or parse_filing
    stats = []
    try:
        embedder = get_embedder() if embedder is None else embedder
        ticker_count = len(tickers)
        for ticker_index, ticker in enumerate(tickers, start=1):
            filings = fetcher(ticker, form, limit)
            sections = [
                section for filing in filings for section in parser(filing, ticker)
            ]
            chunks: list[Chunk] = [
                chunk for section in sections for chunk in chunk_section(section)
            ]
            vectors = []
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                vectors.extend(
                    embedder.embed_documents([chunk.text for chunk in batch])
                )
            if chunks:
                store.upsert_chunks(chunks, vectors)
            stats.append(
                IngestStats(
                    ticker=ticker,
                    filings=len(filings),
                    sections=len(sections),
                    chunks=len(chunks),
                    total_objects=store.count(),
                )
            )
            if progress_callback is not None:
                progress_callback(ticker_index, ticker_count)
    finally:
        if owns_store:
            store.close()
    return stats


def _run_ingest(args: argparse.Namespace) -> None:
    for stats in ingest_tickers(args.tickers, form=args.form, limit=args.limit):
        print(
            f"{stats.ticker}: filings={stats.filings}, sections={stats.sections}, "
            f"chunks={stats.chunks}, total_objects={stats.total_objects}"
        )


def _run_search(args: argparse.Namespace) -> None:
    store = VectorStore()
    try:
        reranker = Reranker() if args.rerank else None
        retriever = Retriever(store, get_embedder(), reranker=reranker)
        filters = SearchFilters(ticker=args.ticker) if args.ticker else None
        results = retriever.retrieve(
            args.query,
            k=args.k,
            filters=filters,
            rerank=args.rerank,
        )
        for rank, result in enumerate(results, start=1):
            excerpt = " ".join(result.text.split())[:200]
            print(
                f"{rank}. {result.ticker} | {result.item} | "
                f"score={result.score:.4f}\n{excerpt}\n"
            )
    finally:
        store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m memory.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser(
        "ingest", help="Fetch, chunk, embed, and store SEC filings"
    )
    ingest.add_argument("tickers", nargs="+")
    ingest.add_argument("--form", default="10-K")
    ingest.add_argument("--limit", type=int, default=2)
    ingest.set_defaults(handler=_run_ingest)

    search = commands.add_parser("search", help="Hybrid-search filing chunks")
    search.add_argument("query")
    search.add_argument("--k", type=int, default=5)
    search.add_argument("--rerank", action="store_true")
    search.add_argument("--ticker")
    search.set_defaults(handler=_run_search)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
