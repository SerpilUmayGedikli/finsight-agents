from datetime import date

from pydantic import BaseModel


class FilingSection(BaseModel):
    text: str
    ticker: str
    cik: str
    accession_number: str
    form_type: str
    filing_date: date
    item: str
    source_url: str


class Chunk(BaseModel):
    chunk_id: str
    text: str
    ticker: str
    cik: str
    accession_number: str
    form_type: str
    filing_date: date
    item: str
    chunk_index: int
    source_url: str
