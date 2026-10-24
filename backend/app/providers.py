from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.schemas import ChatMessage, Usage


@dataclass
class ProviderResult:
    """What every provider returns, regardless of vendor."""

    content: str
    finish_reason: str
    usage: Usage | None  # None = unknown (never a fake zero)


class Provider(ABC):
    """Common interface. Routes and routing logic only use this."""

    @abstractmethod
    async def generate(
        self,
        model_id: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResult:
        """Return one complete (non-streaming) answer."""


class FakeProvider(Provider):
    """Deterministic, offline provider for development and tests."""

    async def generate(
        self,
        model_id: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResult:
        last = messages[-1].content
        return ProviderResult(
            content=f"[FAKE:{model_id}] Echo: {last}",
            finish_reason="stop",
            usage=None,
        )


# Milestone 2: every provider name maps to the fake. Milestone 3 swaps in
# real Groq/Gemini adapters here.
_PROVIDERS: dict[str, Provider] = {
    "gemini": FakeProvider(),
    "groq": FakeProvider(),
}


def get_provider(name: str) -> Provider:
    return _PROVIDERS[name]
