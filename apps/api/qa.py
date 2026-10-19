import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from gateway.llm import LLMClient
from memory.vector_store import ScoredChunk

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    chunk_id: str
    quote: str
    verified: bool = False


class LLMAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_context: bool


class AskResult(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_context: bool
    attempts: int
    model: str


class AnswerParseError(Exception):
    """Raised when the LLM response stays invalid after all retries."""


_SYSTEM_PROMPT = """Answer ONLY from the provided context. If the context does not contain the answer, set insufficient_context=true and say so in the answer.
Respond with ONLY a JSON object with exactly these keys:
- answer: string
- citations: array of objects, each with chunk_id and quote
- insufficient_context: boolean

Each citation quote must be copied VERBATIM from its cited chunk."""
_CODE_FENCE = re.compile(r"\A```[\w-]*\s*(.*?)\s*```\Z", re.DOTALL)


def build_prompt(
    question: str, chunks: list[ScoredChunk]
) -> tuple[str, list[dict[str, str]]]:
    context = []
    for chunk in chunks:
        context.append(
            f"[{chunk.chunk_id}] ({chunk.ticker}, {chunk.form_type}, "
            f"{chunk.item}, {chunk.filing_date.isoformat()})\n{chunk.text}"
        )
    context_text = "\n\n".join(context)
    user_content = f"Context:\n{context_text}\n\nQuestion: {question}"
    return _SYSTEM_PROMPT, [{"role": "user", "content": user_content}]


def _strip_code_fence(output: str) -> str:
    match = _CODE_FENCE.fullmatch(output.strip())
    return match.group(1).strip() if match else output.strip()


def _normalize_quote(text: str) -> str:
    return " ".join(text.casefold().split())


def _verify_citations(
    citations: list[Citation], chunks: list[ScoredChunk]
) -> list[Citation]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    verified_citations = []
    for citation in citations:
        chunk = chunks_by_id.get(citation.chunk_id)
        if chunk is None:
            continue
        normalized_quote = _normalize_quote(citation.quote)
        normalized_text = _normalize_quote(chunk.text)
        verified_citations.append(
            citation.model_copy(
                update={
                    "verified": bool(
                        normalized_quote and normalized_quote in normalized_text
                    )
                }
            )
        )
    return verified_citations


def answer_question(
    question: str,
    chunks: list[ScoredChunk],
    llm: LLMClient,
    max_retries: int = 2,
) -> AskResult:
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")

    system, messages = build_prompt(question, chunks)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 2):
        output = llm.complete(system, messages)
        try:
            parsed: Any = json.loads(_strip_code_fence(output))
            answer = LLMAnswer.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "llm_answer_parse_failed", extra={"attempt": attempt, "error": str(exc)}
            )
            if attempt > max_retries:
                break
            messages.extend(
                [
                    {"role": "assistant", "content": output},
                    {
                        "role": "user",
                        "content": (
                            f"The previous response was invalid. Error: {exc}\n"
                            "Return only valid JSON matching the schema"
                        ),
                    },
                ]
            )
            continue

        logger.info(
            "llm_answer_parsed", extra={"attempts": attempt, "model": llm.model}
        )
        return AskResult(
            answer=answer.answer,
            citations=_verify_citations(answer.citations, chunks),
            insufficient_context=answer.insufficient_context,
            attempts=attempt,
            model=llm.model,
        )

    raise AnswerParseError(
        f"Could not parse a valid LLM answer after {max_retries + 1} attempts: {last_error}"
    ) from last_error
