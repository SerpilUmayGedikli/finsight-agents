from apps.api.settings import Settings, get_settings
from gateway.llm import (
    AnthropicLLMClient,
    FakeLLMClient,
    GeminiLLMClient,
    LLMClient,
    LLMError,
)


def build_client(provider: str) -> LLMClient:
    settings = get_settings()
    normalized_provider = provider.strip().lower()
    if normalized_provider == "anthropic":
        return AnthropicLLMClient(settings)
    if normalized_provider == "gemini":
        return GeminiLLMClient(settings)
    if normalized_provider == "fake":
        return FakeLLMClient(["Fake response"])
    raise LLMError(
        f"Unsupported LLM provider {provider!r}; expected 'anthropic', 'gemini', or 'fake'."
    )


def available_providers(settings: Settings) -> list[str]:
    providers = []
    if settings.anthropic_api_key and settings.anthropic_api_key.strip():
        providers.append("anthropic")
    if settings.gemini_api_key.strip():
        providers.append("gemini")
    if settings.llm_provider.strip().lower() == "fake":
        providers.append("fake")
    return providers
