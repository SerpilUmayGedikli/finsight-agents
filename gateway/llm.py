from typing import Any, Protocol, cast

from apps.api.settings import Settings, get_settings


class LLMError(Exception):
    """Raised when an LLM provider cannot complete a request."""


class LLMClient(Protocol):
    model: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
    ) -> str: ...


class AnthropicLLMClient:
    def __init__(self, settings: Settings | None = None):
        settings = settings or get_settings()
        self.model = settings.anthropic_model
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is required to use AnthropicLLMClient")

        try:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=settings.anthropic_api_key)
        except Exception as exc:
            raise LLMError(f"Could not initialize Anthropic client: {exc}") from exc

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
    ) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                system=system,
                messages=cast(Any, messages),
                max_tokens=max_tokens,
            )
            return "".join(
                text
                for block in response.content
                if isinstance((text := getattr(block, "text", None)), str)
            )
        except Exception as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc


class FakeLLMClient:
    model = "fake"

    def __init__(self, responses: list[str], *, cycle: bool = False):
        if not responses:
            raise ValueError("responses must contain at least one canned response")
        self._responses = list(responses)
        self._cycle = cycle
        self._next_response = 0
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "messages": [dict(message) for message in messages],
                "max_tokens": max_tokens,
            }
        )
        if self._next_response >= len(self._responses):
            if not self._cycle:
                raise LLMError("FakeLLMClient has no canned responses remaining")
            self._next_response = 0

        response = self._responses[self._next_response]
        self._next_response += 1
        return response


def get_llm_client(
    settings: Settings | None = None,
    *,
    fake_responses: list[str] | None = None,
) -> LLMClient:
    settings = settings or Settings()
    provider = settings.llm_provider.strip().lower()
    if provider == "fake":
        responses = fake_responses if fake_responses is not None else ["Fake response"]
        return FakeLLMClient(responses)
    if provider == "anthropic":
        return AnthropicLLMClient(settings)
    raise LLMError(
        f"Unsupported LLM_PROVIDER {provider!r}; expected 'anthropic' or 'fake'."
    )
