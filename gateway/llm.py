from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol, cast

from apps.api.settings import Settings, get_settings


class LLMError(Exception):
    """Raised when an LLM provider cannot complete a request."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient(Protocol):
    model: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
    ) -> str: ...

    def complete_detailed(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


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
        return self.complete_detailed(system, messages, max_tokens).text

    def complete_detailed(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
    ) -> LLMResponse:
        started_at = perf_counter()
        try:
            response = self._client.messages.create(
                model=self.model,
                system=system,
                messages=cast(Any, messages),
                max_tokens=max_tokens,
            )
            text = "".join(
                text
                for block in response.content
                if isinstance((text := getattr(block, "text", None)), str)
            )
            usage = getattr(response, "usage", None)
            return LLMResponse(
                text=text,
                provider="anthropic",
                model=self.model,
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                latency_ms=(perf_counter() - started_at) * 1000,
            )
        except Exception as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc


class GeminiLLMClient:
    def __init__(self, settings: Settings | None = None):
        settings = settings or get_settings()
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is required to use GeminiLLMClient")

        self.model = settings.gemini_model
        try:
            from google import genai

            self._genai = genai
            self._client = genai.Client(api_key=settings.gemini_api_key)
        except Exception as exc:
            raise LLMError(f"Could not initialize Gemini client: {exc}") from exc

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
    ) -> str:
        return self.complete_detailed(system, messages, max_tokens).text

    def complete_detailed(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
    ) -> LLMResponse:
        started_at = perf_counter()
        contents = [
            {
                "role": {"user": "user", "assistant": "model"}[message["role"]],
                "parts": [{"text": message["content"]}],
            }
            for message in messages
        ]
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=cast(Any, contents),
                config=self._genai.types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
            )
            usage = getattr(response, "usage_metadata", None)
            return LLMResponse(
                text=response.text or "",
                provider="gemini",
                model=self.model,
                input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
                latency_ms=(perf_counter() - started_at) * 1000,
            )
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc


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
        return self.complete_detailed(system, messages, max_tokens).text

    def complete_detailed(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
    ) -> LLMResponse:
        started_at = perf_counter()
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
        return LLMResponse(
            text=response,
            provider="fake",
            model=self.model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=(perf_counter() - started_at) * 1000,
        )


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
    if provider == "gemini":
        return GeminiLLMClient(settings)
    raise LLMError(
        f"Unsupported LLM_PROVIDER {provider!r}; expected 'anthropic', 'gemini', or 'fake'."
    )
