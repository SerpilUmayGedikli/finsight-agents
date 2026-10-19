import re

import tiktoken

from memory.models import Chunk, FilingSection

_TOKENIZER = tiktoken.get_encoding("cl100k_base")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _token_count(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _prefix_length(text: str, max_tokens: int) -> int:
    low = 1
    high = len(text)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        if _token_count(text[:middle]) <= max_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1

    if best == 0:
        return min(1, len(text))

    boundary = text.rfind(" ", 0, best + 1)
    if boundary > 0 and _token_count(text[:boundary]) >= max_tokens - 20:
        return boundary
    return best


def _hard_split(text: str, max_tokens: int) -> list[str]:
    pieces = []
    remaining = text
    while remaining:
        end = _prefix_length(remaining, max_tokens)
        pieces.append(remaining[:end])
        remaining = remaining[end:]
    return pieces


def _section_pieces(text: str, max_tokens: int) -> list[tuple[str, str]]:
    pieces = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if _token_count(paragraph) <= max_tokens:
            pieces.append((paragraph, "\n\n"))
            continue

        sentences = _SENTENCE_BOUNDARY.split(paragraph)
        for sentence in sentences:
            if not sentence:
                continue
            sentence_pieces = (
                [sentence]
                if _token_count(sentence) <= max_tokens
                else _hard_split(sentence, max_tokens)
            )
            pieces.extend((piece, " ") for piece in sentence_pieces)
    return pieces


def _overlap_suffix(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""

    boundaries = [match.end() for match in _SENTENCE_BOUNDARY.finditer(text)]
    boundaries.extend(match.start() for match in re.finditer(r"\n\n+", text))
    valid = [
        text[index:]
        for index in boundaries
        if _token_count(text[index:]) <= overlap_tokens
    ]
    if valid:
        return min(valid, key=lambda suffix: overlap_tokens - _token_count(suffix))

    start = len(text) - _prefix_length(text[::-1], overlap_tokens)
    return text[start:]


def _rebalance_short_chunks(
    chunks: list[str], min_tokens: int, max_tokens: int
) -> list[str]:
    index = 0
    while index < len(chunks) - 1:
        current = chunks[index]
        following = chunks[index + 1]
        current_count = _token_count(current)
        combined = current + "\n\n" + following
        if current_count >= min_tokens:
            index += 1
            continue
        if _token_count(combined) <= max_tokens:
            chunks[index : index + 2] = [combined]
            continue

        needed = min_tokens - current_count
        separator = "\n\n"
        needed = max(1, needed - _token_count(separator))
        split_at = _prefix_length(following, needed)
        if split_at <= 0 or split_at >= len(following):
            index += 1
            continue
        chunks[index] = current + separator + following[:split_at]
        chunks[index + 1] = following[split_at:]
        index += 1
    return chunks


def chunk_section(
    section: FilingSection,
    max_tokens: int = 700,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative")
    if not section.text.strip():
        return []

    overlap = min(overlap_tokens, max_tokens - 1)
    content_limit = max_tokens - overlap
    pieces = _section_pieces(section.text, content_limit)
    content_chunks = []
    current = ""
    for piece, separator in pieces:
        candidate = piece if not current else current + separator + piece
        if current and _token_count(candidate) > content_limit:
            content_chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        content_chunks.append(current)

    min_tokens = min(500, content_limit)
    content_chunks = _rebalance_short_chunks(content_chunks, min_tokens, content_limit)

    chunks = []
    previous = ""
    for chunk_index, content in enumerate(content_chunks):
        prefix = _overlap_suffix(previous, overlap) if previous else ""
        chunk_text = prefix + content
        chunks.append(
            Chunk(
                chunk_id=f"{section.accession_number}:{section.item}:{chunk_index}",
                text=chunk_text,
                ticker=section.ticker,
                cik=section.cik,
                accession_number=section.accession_number,
                form_type=section.form_type,
                filing_date=section.filing_date,
                item=section.item,
                chunk_index=chunk_index,
                source_url=section.source_url,
            )
        )
        previous = content
    return chunks
