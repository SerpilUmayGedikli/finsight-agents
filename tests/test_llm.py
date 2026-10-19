import sys
from types import SimpleNamespace

import pytest

from apps.api.settings import Settings
from gateway import llm
from gateway.llm import (
    AnthropicLLMClient,
    FakeLLMClient,
    GeminiLLMClient,
    LLMError,
    LLMResponse,
    get_llm_client,
)


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
                content=[SimpleNamespace(text="Hello"), SimpleNamespace(text=" world")],
                usage=SimpleNamespace(input_tokens=17, output_tokens=9),
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

    detailed = client.complete_detailed("system prompt", messages, max_tokens=30)

    assert detailed == LLMResponse(
        text="Hello world",
        provider="anthropic",
        model="test-model",
        input_tokens=17,
        output_tokens=9,
        latency_ms=detailed.latency_ms,
    )


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


def test_fake_complete_detailed_reports_provider_and_zero_token_usage():
    client = FakeLLMClient(["fake output"])

    result = client.complete_detailed("system", [{"role": "user", "content": "Hi"}])

    assert result.text == "fake output"
    assert result.provider == "fake"
    assert result.model == "fake"
    assert result.input_tokens == result.output_tokens == 0
    assert result.latency_ms >= 0


def _mock_genai(monkeypatch, generate_content):
    captured = {}

    class FakeGeminiModels:
        def generate_content(self, **kwargs):
            captured["request"] = kwargs
            return generate_content()

    class FakeGeminiClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.models = FakeGeminiModels()

    def make_config(**kwargs):
        captured["config"] = kwargs
        return SimpleNamespace(**kwargs)

    fake_genai = SimpleNamespace(
        Client=FakeGeminiClient,
        types=SimpleNamespace(GenerateContentConfig=make_config),
    )
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))
    return captured


def test_gemini_role_mapping_config_and_usage(monkeypatch):
    captured = _mock_genai(
        monkeypatch,
        lambda: SimpleNamespace(
            text='{"ok": true}',
            usage_metadata=SimpleNamespace(
                prompt_token_count=21,
                candidates_token_count=8,
            ),
        ),
    )
    client = GeminiLLMClient(
        Settings(gemini_api_key="test-key", gemini_model="test-gemini")
    )

    result = client.complete_detailed(
        "system prompt",
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Prior response"},
        ],
        max_tokens=73,
    )

    assert captured["api_key"] == "test-key"
    assert captured["request"] == {
        "model": "test-gemini",
        "contents": [
            {"role": "user", "parts": [{"text": "Question"}]},
            {"role": "model", "parts": [{"text": "Prior response"}]},
        ],
        "config": captured["request"]["config"],
    }
    assert captured["config"] == {
        "system_instruction": "system prompt",
        "max_output_tokens": 73,
        "response_mime_type": "application/json",
    }
    assert result.text == '{"ok": true}'
    assert result.provider == "gemini"
    assert result.model == "test-gemini"
    assert result.input_tokens == 21
    assert result.output_tokens == 8
    assert result.latency_ms >= 0


def test_gemini_missing_key_raises_clear_error():
    with pytest.raises(LLMError, match="GEMINI_API_KEY is required"):
        GeminiLLMClient(Settings(gemini_api_key=""))


def test_gemini_sdk_error_is_wrapped(monkeypatch):
    _mock_genai(
        monkeypatch,
        lambda: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
    )
    client = GeminiLLMClient(Settings(gemini_api_key="test-key"))

    with pytest.raises(LLMError, match="Gemini request failed: provider unavailable"):
        client.complete("system", [{"role": "user", "content": "Hi"}])


def test_gemini_complete_returns_empty_string_for_none_text(monkeypatch):
    _mock_genai(
        monkeypatch,
        lambda: SimpleNamespace(text=None, usage_metadata=None),
    )
    client = GeminiLLMClient(Settings(gemini_api_key="test-key"))

    assert client.complete("system", [{"role": "user", "content": "Hi"}]) == ""
