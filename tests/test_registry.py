from apps.api.settings import Settings
from gateway import registry
from gateway.llm import FakeLLMClient
from gateway.registry import available_providers, build_client


def test_build_client_selects_provider(monkeypatch):
    anthropic_client = object()
    gemini_client = object()
    monkeypatch.setattr(registry, "get_settings", lambda: Settings())
    monkeypatch.setattr(
        registry, "AnthropicLLMClient", lambda settings: anthropic_client
    )
    monkeypatch.setattr(registry, "GeminiLLMClient", lambda settings: gemini_client)

    assert build_client(" ANTHROPIC ") is anthropic_client
    assert build_client("gemini") is gemini_client
    assert isinstance(build_client("fake"), FakeLLMClient)


def test_available_providers_requires_keys_except_explicit_fake():
    assert available_providers(Settings()) == []
    assert available_providers(
        Settings(anthropic_api_key="anthropic-key", gemini_api_key=" ")
    ) == ["anthropic"]
    assert available_providers(
        Settings(
            anthropic_api_key=" ",
            gemini_api_key="gemini-key",
            llm_provider="fake",
        )
    ) == ["gemini", "fake"]
