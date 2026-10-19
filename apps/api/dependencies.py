from collections.abc import Iterator

from fastapi import Depends

from apps.api.settings import Settings, get_settings
from gateway.llm import LLMClient
from gateway.llm import get_llm_client as create_llm_client
from memory.embeddings import Embedder, LocalEmbedder, OpenAIEmbedder
from memory.reranker import Reranker
from memory.retriever import Retriever
from memory.vector_store import VectorStore


def get_vector_store(
    settings: Settings = Depends(get_settings),
) -> Iterator[VectorStore]:
    store = VectorStore(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
        grpc_port=settings.weaviate_grpc_port,
    )
    try:
        yield store
    finally:
        store.close()


def get_embedder(settings: Settings = Depends(get_settings)) -> Embedder:
    provider = settings.embedder.strip().lower()
    if provider == "local":
        return LocalEmbedder()
    if provider == "openai":
        return OpenAIEmbedder()
    raise ValueError(
        f"Unsupported EMBEDDER value {provider!r}; expected 'local' or 'openai'."
    )


def get_retriever(
    store: VectorStore = Depends(get_vector_store),
    embedder: Embedder = Depends(get_embedder),
) -> Retriever:
    return Retriever(store, embedder, reranker=Reranker())


def get_llm_client(settings: Settings = Depends(get_settings)) -> LLMClient:
    return create_llm_client(settings)
