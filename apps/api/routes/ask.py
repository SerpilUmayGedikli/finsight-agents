from time import perf_counter

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from apps.api.dependencies import get_llm_client, get_retriever, get_settings
from apps.api.errors import APIError
from apps.api.qa import AnswerParseError, Citation, answer_question
from apps.api.routes.search import SearchResult
from apps.api.settings import Settings
from gateway.llm import LLMClient
from memory.retriever import Retriever
from memory.vector_store import SearchFilters

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    filters: SearchFilters | None = None
    k: int | None = Field(default=None, ge=1)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[SearchResult]
    latency_ms: float
    model: str
    attempts: int
    insufficient_context: bool


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    settings: Settings = Depends(get_settings),
    retriever: Retriever = Depends(get_retriever),
    llm: LLMClient = Depends(get_llm_client),
) -> AskResponse:
    started_at = perf_counter()
    chunks = retriever.retrieve(
        request.question,
        k=request.k if request.k is not None else settings.retrieval_k,
        filters=request.filters,
        rerank=settings.rerank_default,
    )
    retrieved_chunks = [
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

    if not chunks:
        return AskResponse(
            answer="Insufficient context to answer the question.",
            citations=[],
            retrieved_chunks=[],
            latency_ms=round((perf_counter() - started_at) * 1000, 2),
            model=llm.model,
            attempts=0,
            insufficient_context=True,
        )

    try:
        result = answer_question(request.question, chunks, llm)
    except AnswerParseError as exc:
        raise APIError("answer_parse_error", str(exc), status_code=502) from exc

    return AskResponse(
        answer=result.answer,
        citations=result.citations,
        retrieved_chunks=retrieved_chunks,
        latency_ms=round((perf_counter() - started_at) * 1000, 2),
        model=result.model,
        attempts=result.attempts,
        insufficient_context=result.insufficient_context,
    )
