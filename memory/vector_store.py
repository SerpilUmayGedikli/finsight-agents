from datetime import UTC, date, datetime, time
from typing import Any, cast

import weaviate
from pydantic import BaseModel
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery

from memory.models import Chunk


class SearchFilters(BaseModel):
    ticker: str | None = None
    form_type: str | None = None
    item: str | None = None
    filing_date_from: date | None = None
    filing_date_to: date | None = None


class ScoredChunk(Chunk):
    score: float


class VectorStore:
    collection_name = "FilingChunk"

    def __init__(
        self, host: str = "localhost", port: int = 8080, grpc_port: int = 50051
    ):
        self._client = weaviate.connect_to_local(
            host=host, port=port, grpc_port=grpc_port
        )

    def ensure_collection(self) -> None:
        if self._client.collections.exists(self.collection_name):
            return

        self._client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="ticker", data_type=DataType.TEXT),
                Property(name="cik", data_type=DataType.TEXT),
                Property(name="accession_number", data_type=DataType.TEXT),
                Property(name="form_type", data_type=DataType.TEXT),
                Property(name="filing_date", data_type=DataType.DATE),
                Property(name="item", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="source_url", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.TEXT),
            ],
        )

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.ensure_collection()
        collection = self._client.collections.get(self.collection_name)

        for chunk, vector in zip(chunks, vectors):
            properties = chunk.model_dump()
            properties["filing_date"] = datetime.combine(
                chunk.filing_date,
                time.min,
                tzinfo=UTC,
            )
            object_uuid = weaviate.util.generate_uuid5(chunk.chunk_id)
            if collection.data.exists(object_uuid):
                collection.data.replace(
                    uuid=object_uuid,
                    properties=properties,
                    vector=vector,
                )
            else:
                collection.data.insert(
                    uuid=object_uuid,
                    properties=properties,
                    vector=vector,
                )
        return len(chunks)

    def _build_filter(self, filters: SearchFilters | None):
        if filters is None:
            return None

        conditions = []
        for field in ("ticker", "form_type", "item"):
            value = getattr(filters, field)
            if value is not None:
                conditions.append(Filter.by_property(field).equal(value))
        if filters.filing_date_from is not None:
            start = datetime.combine(filters.filing_date_from, time.min, tzinfo=UTC)
            conditions.append(Filter.by_property("filing_date").greater_or_equal(start))
        if filters.filing_date_to is not None:
            end = datetime.combine(filters.filing_date_to, time.max, tzinfo=UTC)
            conditions.append(Filter.by_property("filing_date").less_or_equal(end))

        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else Filter.all_of(conditions)

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        k: int = 20,
        filters: SearchFilters | None = None,
    ) -> list[ScoredChunk]:
        if k <= 0:
            return []
        collection = self._client.collections.get(self.collection_name)
        results = collection.query.hybrid(
            query=query,
            vector=query_vector,
            limit=k,
            alpha=0.5,
            return_metadata=MetadataQuery(score=True),
            filters=self._build_filter(filters),
        )
        return [
            ScoredChunk.model_validate(
                {
                    **cast(dict[str, Any], result.properties),
                    "score": float(result.metadata.score or 0.0),
                }
            )
            for result in results.objects
        ]

    def count(self) -> int:
        if not self._client.collections.exists(self.collection_name):
            return 0
        collection = self._client.collections.get(self.collection_name)
        return collection.aggregate.over_all(total_count=True).total_count or 0

    def close(self) -> None:
        if self._client.is_connected():
            self._client.close()
