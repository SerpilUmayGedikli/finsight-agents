import sys
from types import SimpleNamespace

import pytest

from apps.api.settings import Settings
from gateway import llm
from gateway.llm import AnthropicLLMClient, FakeLLMClient, LLMError, get_llm_client


def test_fake_client_returns_responses_in_order_and_records_calls():
    client = FakeLLMClient(["first", "second"])
    messages = [{"role": "user", "content": "hello"}]

    assert client.complete("system", messages, max_tokens=20) == "first"
    assert client.complete("system-2", messages) == "second"
    assert client.calls == [
        {"system": "system", "messages": messages, "max_tokens": 20},
        {"system": "system-2", "messages": messages, "max_tokens": 1024},
    ]


def test_fake_client_raises_when_responses_are_exhausted():
    client = FakeLLMClient(["only response"])
    client.complete("system", [])

    with pytest.raises(LLMError, match="no canned responses remaining"):
        client.complete("system", [])


def test_factory_selects_fake_and_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    fake = get_llm_client(fake_responses=["hello"])
    assert isinstance(fake, FakeLLMClient)

    class FakeAnthropicClient:
        def __init__(self, settings=None):
            self.settings = settings

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(llm, "AnthropicLLMClient", FakeAnthropicClient)
    settings = Settings(anthropic_api_key="test-key")

    assert isinstance(get_llm_client(settings), FakeAnthropicClient)


def test_anthropic_client_uses_sdk_and_concatenates_text(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                content=[SimpleNamespace(text="Hello"), SimpleNamespace(text=" world")]
            )

    class FakeAnthropic:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    monkeypatch.setitem(
        sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic)
    )
    settings = Settings(anthropic_api_key="test-key", anthropic_model="test-model")
    client = AnthropicLLMClient(settings)
    messages = [{"role": "user", "content": "Hi"}]

    result = client.complete("system prompt", messages, max_tokens=30)

    assert result == "Hello world"
    assert captured["api_key"] == "test-key"
    assert captured["request"] == {
        "model": "test-model",
        "system": "system prompt",
        "messages": messages,
        "max_tokens": 30,
    }


def test_anthropic_sdk_errors_are_wrapped(monkeypatch):
    class FakeMessages:
        def create(self, **kwargs):
            raise RuntimeError("provider unavailable")

    class FakeAnthropic:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    monkeypatch.setitem(
        sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic)
    )
    client = AnthropicLLMClient(Settings(anthropic_api_key="test-key"))

    with pytest.raises(
        LLMError, match="Anthropic request failed: provider unavailable"
    ):
        client.complete("system", [])
