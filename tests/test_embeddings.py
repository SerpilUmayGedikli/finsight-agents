from types import SimpleNamespace

import pytest

from memory import embeddings
from memory.embeddings import LocalEmbedder, OpenAIEmbedder, get_embedder


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return [[float(index)] * 384 for index, _ in enumerate(texts)]


def test_local_model_loads_lazily_and_prefixes_only_queries(monkeypatch):
    instances = []

    def make_model(model_name):
        model = FakeSentenceTransformer(model_name)
        instances.append(model)
        return model

    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=make_model),
    )
    embedder = LocalEmbedder(batch_size=7)

    assert instances == []
    embedder.embed_documents(["first document", "second document"])
    query_vector = embedder.embed_query("a search query")

    assert len(instances) == 1
    assert instances[0].model_name == "BAAI/bge-small-en-v1.5"
    assert instances[0].calls == [
        (
            ["first document", "second document"],
            {"batch_size": 7, "normalize_embeddings": True},
        ),
        (
            ["Represent this sentence for searching relevant passages: a search query"],
            {"batch_size": 7, "normalize_embeddings": True},
        ),
    ]
    assert len(query_vector) == 384


def test_openai_batches_at_most_100_texts_and_preserves_order(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requested_batches = []

    class FakeEmbeddings:
        def create(self, model, input):
            requested_batches.append((model, input))
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=index, embedding=[float(int(text))])
                    for index, text in reversed(list(enumerate(input)))
                ]
            )

    client = SimpleNamespace(embeddings=FakeEmbeddings())
    vectors = OpenAIEmbedder(client=client).embed_documents(
        [str(index) for index in range(205)]
    )

    assert [len(batch) for _, batch in requested_batches] == [100, 100, 5]
    assert all(model == "text-embedding-3-small" for model, _ in requested_batches)
    assert [vector[0] for vector in vectors] == [float(index) for index in range(205)]


def test_factory_defaults_to_local_and_selects_openai(monkeypatch):
    assert isinstance(get_embedder(), LocalEmbedder)

    class FakeOpenAIEmbedder:
        pass

    monkeypatch.setenv("EMBEDDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(embeddings, "OpenAIEmbedder", FakeOpenAIEmbedder)

    assert isinstance(get_embedder(), FakeOpenAIEmbedder)


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDER", "remote")

    with pytest.raises(ValueError, match="expected 'local' or 'openai'"):
        get_embedder()


def test_openai_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        OpenAIEmbedder()


@pytest.mark.slow
def test_real_local_model_has_expected_dimension():
    pytest.importorskip("sentence_transformers")
    vector = LocalEmbedder().embed_query("A financial filing query")

    assert len(vector) == 384
