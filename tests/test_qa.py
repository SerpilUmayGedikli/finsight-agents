import json
from datetime import date

import pytest

from apps.api.qa import AnswerParseError, answer_question, build_prompt
from gateway.llm import FakeLLMClient
from memory.vector_store import ScoredChunk


def make_chunk(
    chunk_id="chunk-1",
    text="Revenue grew 12 percent year over year due to services.",
):
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        ticker="ACME",
        cik="0000000001",
        accession_number="accession-1",
        form_type="10-K",
        filing_date=date(2025, 2, 1),
        item="Item 7",
        chunk_index=0,
        source_url="https://example.test/filing",
        score=0.9,
    )


def payload(**updates):
    answer = {
        "answer": "Revenue grew 12 percent.",
        "citations": [{"chunk_id": "chunk-1", "quote": "Revenue grew 12 percent"}],
        "insufficient_context": False,
    }
    answer.update(updates)
    return answer


def test_valid_json_succeeds_on_first_attempt():
    llm = FakeLLMClient([json.dumps(payload())])

    result = answer_question("What happened to revenue?", [make_chunk()], llm)

    assert result.attempts == 1
    assert result.answer == "Revenue grew 12 percent."
    assert result.model == "fake"


def test_malformed_json_retries_and_feeds_back_error():
    llm = FakeLLMClient(["{bad json", json.dumps(payload())])

    result = answer_question("What happened to revenue?", [make_chunk()], llm)

    assert result.attempts == 2
    retry_messages = llm.calls[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": "{bad json"}
    feedback = retry_messages[-1]["content"]
    assert "Expecting property name" in feedback
    assert "Return only valid JSON matching the schema" in feedback


def test_all_invalid_attempts_raise_answer_parse_error():
    llm = FakeLLMClient(["not json", "still not json", "also not json"])

    with pytest.raises(AnswerParseError, match="after 3 attempts"):
        answer_question("question", [make_chunk()], llm, max_retries=2)

    assert len(llm.calls) == 3


def test_json_in_markdown_fence_parses():
    response = "```json\n" + json.dumps(payload()) + "\n```"
    result = answer_question("question", [make_chunk()], FakeLLMClient([response]))

    assert result.answer == "Revenue grew 12 percent."
    assert result.attempts == 1


def test_verbatim_quote_is_verified_with_whitespace_and_case_normalization():
    answer = payload(
        citations=[{"chunk_id": "chunk-1", "quote": "  REVENUE   grew 12 percent "}]
    )

    result = answer_question(
        "question", [make_chunk()], FakeLLMClient([json.dumps(answer)])
    )

    assert result.citations[0].verified is True


def test_fabricated_quote_is_kept_but_unverified():
    answer = payload(
        citations=[{"chunk_id": "chunk-1", "quote": "Revenue grew 90 percent"}]
    )

    result = answer_question(
        "question", [make_chunk()], FakeLLMClient([json.dumps(answer)])
    )

    assert len(result.citations) == 1
    assert result.citations[0].verified is False


def test_unknown_chunk_citation_is_dropped():
    answer = payload(citations=[{"chunk_id": "missing", "quote": "some quote"}])

    result = answer_question(
        "question", [make_chunk()], FakeLLMClient([json.dumps(answer)])
    )

    assert result.citations == []


def test_insufficient_context_is_preserved():
    answer = payload(
        answer="The context does not provide this information.",
        insufficient_context=True,
    )

    result = answer_question(
        "question", [make_chunk()], FakeLLMClient([json.dumps(answer)])
    )

    assert result.insufficient_context is True
    assert result.answer == "The context does not provide this information."


def test_build_prompt_formats_context_and_json_instructions():
    system, messages = build_prompt("What happened?", [make_chunk()])

    assert "ONLY from the provided context" in system
    assert "JSON object" in system
    assert "[chunk-1] (ACME, 10-K, Item 7, 2025-02-01)" in messages[0]["content"]
    assert make_chunk().text in messages[0]["content"]
