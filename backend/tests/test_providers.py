import asyncio
import pytest

from app.providers import FakeProvider, Provider, ProviderChunk, get_provider
from app.schemas import ChatMessage


def _generate(messages):
    return asyncio.run(FakeProvider().generate("m1", messages))


def test_fake_provider_is_a_provider():
    assert isinstance(FakeProvider(), Provider)


def test_fake_provider_is_deterministic():
    msgs = [ChatMessage(role="user", content="hi")]
    assert _generate(msgs) == _generate(msgs)


def test_fake_provider_echoes_last_message_and_model():
    msgs = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="user", content="second"),
    ]
    result = _generate(msgs)
    assert "second" in result.content
    assert "m1" in result.content
    assert result.finish_reason == "stop"


def test_fake_provider_reports_unknown_usage():
    result = _generate([ChatMessage(role="user", content="hi")])
    assert result.usage is None


def test_get_provider_for_each_configured_name():
    assert isinstance(get_provider("groq"), Provider)
    assert isinstance(get_provider("gemini"), Provider)


@pytest.mark.anyio
async def test_fake_provider_generate_stream():
    provider = FakeProvider()
    msgs = [ChatMessage(role="user", content="hello world")]
    chunks = []
    async for chunk in provider.generate_stream("test-model", msgs):
        assert isinstance(chunk, ProviderChunk)
        chunks.append(chunk)

    # Check words are emitted as deltas
    deltas = [c.delta for c in chunks if c.delta]
    full_text = "".join(deltas)
    assert full_text == "[FAKE:test-model] Echo: hello world"

    # Last chunk has stop finish reason and None usage
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is None
