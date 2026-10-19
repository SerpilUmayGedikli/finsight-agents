from memory.embeddings import Embedder
from memory.reranker import Reranker
from memory.vector_store import ScoredChunk, SearchFilters, VectorStore


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: SearchFilters | None = None,
        rerank: bool = False,
    ) -> list[ScoredChunk]:
        if k <= 0:
            return []

        query_vector = self.embedder.embed_query(query)
        candidates = self.store.hybrid_search(
            query=query,
            query_vector=query_vector,
            k=20 if rerank else k,
            filters=filters,
        )
        if not rerank:
            return candidates[:k]
        if self.reranker is None:
            raise ValueError("rerank=True requires a Reranker")
        return self.reranker.rerank(query, candidates, top_k=k)
