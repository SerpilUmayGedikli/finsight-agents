from typing import Protocol

from apps.api.settings import Settings


class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    dimension = 384
    model_name = "BAAI/bge-small-en-v1.5"
    query_instruction = "Represent this sentence for searching relevant passages: "

    def __init__(self, batch_size: int = 32):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )
        return [
            vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for vector in vectors
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_instruction + text])[0]


class OpenAIEmbedder:
    dimension = 1536
    model_name = "text-embedding-3-small"
    batch_limit = 100

    def __init__(self, client=None):
        api_key = Settings().openai_api_key
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required to use OpenAIEmbedder; set it in the environment."
            )
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self._client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_limit):
            batch = texts[start : start + self.batch_limit]
            response = self._client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            data = sorted(response.data, key=lambda item: item.index)
            embeddings.extend(list(item.embedding) for item in data)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or Settings()
    provider = settings.embedder.strip().lower()
    if provider == "local":
        return LocalEmbedder()
    if provider == "openai":
        return OpenAIEmbedder()
    raise ValueError(
        f"Unsupported EMBEDDER value {provider!r}; expected 'local' or 'openai'."
    )
