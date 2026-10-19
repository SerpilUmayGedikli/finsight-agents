from datetime import date

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from apps.api.dependencies import get_retriever
from memory.retriever import Retriever
from memory.vector_store import SearchFilters

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1)
    filters: SearchFilters | None = None
    rerank: bool = False


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    text: str
    ticker: str
    form_type: str
    item: str
    filing_date: date
    source_url: str


class SearchResponse(BaseModel):
    results: list[SearchResult]


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
) -> SearchResponse:
    chunks = retriever.retrieve(
        request.query,
        k=request.k,
        filters=request.filters,
        rerank=request.rerank,
    )
    return SearchResponse(
        results=[
            SearchResult(
                chunk_id=chunk.chunk_id,
                score=chunk.score,
                text=chunk.text,
                ticker=chunk.ticker,
                form_type=chunk.form_type,
                item=chunk.item,
                filing_date=chunk.filing_date,
                source_url=chunk.source_url,
            )
            for chunk in chunks
        ]
    )
