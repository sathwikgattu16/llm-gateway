import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.providers import (
    AuthenticationError,
    ModelNotFoundError,
    ProviderChunk,
    ProviderResult,
    ProviderTimeoutError,
    RateLimitError,
    UpstreamProviderError,
)
from app.schemas import Usage

client = TestClient(app)

VALID_BODY = {"messages": [{"role": "user", "content": "Hello"}]}


def test_valid_request_returns_expected_structure():
    fake_result = ProviderResult(content="Mocked Gemini response", finish_reason="stop", usage=None)
    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = fake_result
        response = client.post("/v1/chat/completions", json=VALID_BODY)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl-")
        assert data["model"] == "gemini-3.8-flash"  # "auto" placeholder: first model
        assert isinstance(data["created"], int)
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert choice["finish_reason"] == "stop"


def test_explicit_model_is_used(monkeypatch):
    body = {**VALID_BODY, "model": "openai/gpt-oss-20b"}
    fake_result = ProviderResult(content="Mocked response", finish_reason="stop", usage=None)

    with patch("app.providers.GroqProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = fake_result
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        assert response.json()["model"] == "openai/gpt-oss-20b"


def test_unknown_model_returns_404():
    body = {**VALID_BODY, "model": "does-not-exist"}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


def test_usage_is_unknown_not_fake():
    fake_result = ProviderResult(content="Mocked response", finish_reason="stop", usage=None)
    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = fake_result
        data = client.post("/v1/chat/completions", json=VALID_BODY).json()
        assert data["usage"] is None


def test_empty_messages_rejected():
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 422


def test_invalid_role_rejected():
    body = {"messages": [{"role": "tool", "content": "Hello"}]}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 422


def test_invalid_temperature_rejected():
    body = {**VALID_BODY, "temperature": 2.5}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 422


def test_health_still_works():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_groq_route_authentication_error_401():
    body = {**VALID_BODY, "model": "openai/gpt-oss-20b"}
    with patch("app.providers.GroqProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = AuthenticationError("Groq authentication failed.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 401
        assert "authentication failed" in response.json()["detail"].lower()


def test_groq_route_model_not_found_404():
    body = {**VALID_BODY, "model": "openai/gpt-oss-20b"}
    with patch("app.providers.GroqProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = ModelNotFoundError("Groq model not found.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


def test_groq_route_rate_limit_429():
    body = {**VALID_BODY, "model": "openai/gpt-oss-20b"}
    with patch("app.providers.GroqProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = RateLimitError("Groq rate limit exceeded.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()


def test_groq_route_upstream_server_error_502():
    body = {**VALID_BODY, "model": "openai/gpt-oss-20b"}
    with patch("app.providers.GroqProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = UpstreamProviderError("Groq upstream server error.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 502
        assert "upstream" in response.json()["detail"].lower()


def test_groq_route_timeout_error_504():
    body = {**VALID_BODY, "model": "openai/gpt-oss-20b"}
    with patch("app.providers.GroqProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = ProviderTimeoutError("Groq request timed out.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()


def test_groq_route_redacts_api_key_in_error_detail():
    secret_key = "gsk_route_secret_998877"
    body = {**VALID_BODY, "model": "openai/gpt-oss-20b"}
    with patch("app.providers.GroqProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = AuthenticationError("Groq authentication failed: [REDACTED]")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 401
        detail = response.json()["detail"]
        assert secret_key not in detail
        assert "[REDACTED]" in detail


def test_gemini_explicit_model_is_used():
    body = {**VALID_BODY, "model": "gemini-3.7-flash"}
    fake_result = ProviderResult(content="Mocked Gemini response", finish_reason="stop", usage=None)

    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = fake_result
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        assert response.json()["model"] == "gemini-3.7-flash"


def test_gemini_route_authentication_error_401():
    body = {**VALID_BODY, "model": "gemini-3.8-flash"}
    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = AuthenticationError("Gemini authentication failed: API key not valid.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 401
        assert "authentication failed" in response.json()["detail"].lower()


def test_gemini_route_model_not_found_404():
    body = {**VALID_BODY, "model": "gemini-3.8-flash"}
    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = ModelNotFoundError("Gemini model not found.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


def test_gemini_route_rate_limit_429():
    body = {**VALID_BODY, "model": "gemini-3.8-flash"}
    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = RateLimitError("Gemini rate limit exceeded.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()


def test_gemini_route_upstream_server_error_502():
    body = {**VALID_BODY, "model": "gemini-3.8-flash"}
    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = UpstreamProviderError("Gemini upstream server error.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 502
        assert "upstream" in response.json()["detail"].lower()


def test_gemini_route_timeout_error_504():
    body = {**VALID_BODY, "model": "gemini-3.8-flash"}
    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = ProviderTimeoutError("Gemini request timed out.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()


def test_gemini_route_redacts_api_key_in_error_detail():
    secret_key = "AIzaSy_gemini_secret_123456"
    body = {**VALID_BODY, "model": "gemini-3.8-flash"}
    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = AuthenticationError("Gemini authentication failed: [REDACTED]")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 401
        detail = response.json()["detail"]
        assert secret_key not in detail
        assert "[REDACTED]" in detail


# --- Milestone 5: Streaming Route Tests ---


def test_streaming_route_success_gemini():
    async def mock_stream(*args, **kwargs):
        yield ProviderChunk(delta="Hello")
        yield ProviderChunk(delta=" world")
        yield ProviderChunk(finish_reason="stop", usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7))

    with patch("app.providers.GeminiProvider.generate_stream", side_effect=mock_stream):
        body = {**VALID_BODY, "stream": True}
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = [line.strip() for line in response.text.split("\n") if line.strip()]
        assert lines[-1] == "data: [DONE]"

        data_events = [json.loads(line[len("data:"):].strip()) for line in lines if line.startswith("data:") and line != "data: [DONE]"]
        assert len(data_events) >= 3

        # First chunk has role and content
        first = data_events[0]
        assert first["object"] == "chat.completion.chunk"
        assert first["model"] == "gemini-3.8-flash"
        assert first["choices"][0]["delta"]["role"] == "assistant"
        assert first["choices"][0]["delta"]["content"] == "Hello"
        assert first["choices"][0]["finish_reason"] is None

        # Second chunk has content
        second = data_events[1]
        assert second["choices"][0]["delta"]["content"] == " world"
        assert second["choices"][0]["finish_reason"] is None

        # Final chunk has finish_reason and usage
        final = data_events[2]
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final["choices"][0]["delta"] == {}
        assert final["usage"]["prompt_tokens"] == 5
        assert final["usage"]["completion_tokens"] == 2
        assert final["usage"]["total_tokens"] == 7


def test_streaming_route_success_groq():
    async def mock_stream(*args, **kwargs):
        yield ProviderChunk(delta="Groq")
        yield ProviderChunk(delta=" response", finish_reason="stop", usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12))

    with patch("app.providers.GroqProvider.generate_stream", side_effect=mock_stream):
        body = {**VALID_BODY, "model": "openai/gpt-oss-20b", "stream": True}
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = [line.strip() for line in response.text.split("\n") if line.strip()]
        assert lines[-1] == "data: [DONE]"

        data_events = [json.loads(line[len("data:"):].strip()) for line in lines if line.startswith("data:") and line != "data: [DONE]"]
        assert len(data_events) == 3  # delta 1, delta 2, finish chunk
        assert data_events[0]["choices"][0]["delta"]["content"] == "Groq"
        assert data_events[1]["choices"][0]["delta"]["content"] == " response"
        assert data_events[2]["choices"][0]["finish_reason"] == "stop"
        assert data_events[2]["usage"]["total_tokens"] == 12


def test_streaming_route_auth_error_401():
    async def mock_failing_stream(*args, **kwargs):
        raise AuthenticationError("Authentication failed: invalid key.")
        yield  # makes it an async generator

    with patch("app.providers.GeminiProvider.generate_stream", side_effect=mock_failing_stream):
        body = {**VALID_BODY, "stream": True}
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 401
        assert "authentication failed" in response.json()["detail"].lower()


def test_streaming_route_model_not_found_404():
    async def mock_failing_stream(*args, **kwargs):
        raise ModelNotFoundError("Model not found.")
        yield

    with patch("app.providers.GeminiProvider.generate_stream", side_effect=mock_failing_stream):
        body = {**VALID_BODY, "stream": True}
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


def test_streaming_route_rate_limit_429():
    async def mock_failing_stream(*args, **kwargs):
        raise RateLimitError("Rate limit exceeded.")
        yield

    with patch("app.providers.GeminiProvider.generate_stream", side_effect=mock_failing_stream):
        body = {**VALID_BODY, "stream": True}
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()


def test_streaming_route_upstream_error_502():
    async def mock_failing_stream(*args, **kwargs):
        raise UpstreamProviderError("Upstream server error.")
        yield

    with patch("app.providers.GeminiProvider.generate_stream", side_effect=mock_failing_stream):
        body = {**VALID_BODY, "stream": True}
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 502
        assert "upstream" in response.json()["detail"].lower()


def test_streaming_route_timeout_error_504():
    async def mock_failing_stream(*args, **kwargs):
        raise ProviderTimeoutError("Provider request timed out.")
        yield

    with patch("app.providers.GeminiProvider.generate_stream", side_effect=mock_failing_stream):
        body = {**VALID_BODY, "stream": True}
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()


def test_streaming_route_unknown_model_rejected_404():
    body = {**VALID_BODY, "model": "non-configured-model", "stream": True}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 404
    assert "non-configured-model" in response.json()["detail"]
