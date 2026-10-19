from memory.vector_store import ScoredChunk


class Reranker:
    model_name = "BAAI/bge-reranker-base"

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, chunks: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if top_k <= 0 or not chunks:
            return []

        scores = self._get_model().predict([(query, chunk.text) for chunk in chunks])
        ranked = [
            chunk.model_copy(update={"score": float(score)})
            for chunk, score in zip(chunks, scores)
        ]
        return sorted(ranked, key=lambda chunk: chunk.score, reverse=True)[:top_k]
