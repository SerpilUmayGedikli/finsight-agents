from datetime import date

import tiktoken

from memory.chunking import chunk_section
from memory.models import FilingSection

ENCODING = tiktoken.get_encoding("cl100k_base")


def make_section(text: str) -> FilingSection:
    return FilingSection(
        text=text,
        ticker="ACME",
        cik="0000123456",
        accession_number="0000123456-26-000001",
        form_type="10-K",
        filing_date=date(2026, 3, 1),
        item="Item 1A",
        source_url="https://example.test/filing",
    )


def test_chunk_sizes_stay_within_limits():
    section = make_section(
        ("Revenue increased across all business units. " * 120) + "Final sentence."
    )
    chunks = chunk_section(section)
    counts = [len(ENCODING.encode(chunk.text)) for chunk in chunks]

    assert len(chunks) > 1
    assert all(count <= 700 for count in counts)
    assert all(count >= 500 for count in counts[:-1])


def test_overlap_is_present_between_chunks():
    section = make_section("A detailed operational update follows. " * 200)
    chunks = chunk_section(section)

    assert len(chunks) > 1
    assert chunks[0].text[-80:] in chunks[1].text


def test_no_empty_chunks():
    chunks = chunk_section(make_section("Useful filing text."))

    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)


def test_metadata_is_copied():
    section = make_section("Useful filing text.")
    chunk = chunk_section(section)[0]

    assert chunk.ticker == section.ticker
    assert chunk.cik == section.cik
    assert chunk.accession_number == section.accession_number
    assert chunk.form_type == section.form_type
    assert chunk.filing_date == section.filing_date
    assert chunk.item == section.item
    assert chunk.source_url == section.source_url


def test_repeated_calls_are_deterministic():
    section = make_section("A deterministic sentence. " * 80)

    assert chunk_section(section) == chunk_section(section)


def test_short_text_produces_one_chunk():
    chunks = chunk_section(make_section("A short section."))

    assert len(chunks) == 1
    assert chunks[0].text == "A short section."


def test_very_long_single_paragraph_is_split():
    section = make_section("Unusually detailed financial disclosures continue. " * 250)
    chunks = chunk_section(section)

    assert len(chunks) > 1
    assert all(len(ENCODING.encode(chunk.text)) <= 700 for chunk in chunks)


def test_chunk_id_uses_accession_item_and_index():
    chunks = chunk_section(make_section("First. Second."))

    assert chunks[0].chunk_id == "0000123456-26-000001:Item 1A:0"
