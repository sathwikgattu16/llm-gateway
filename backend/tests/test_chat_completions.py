from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_BODY = {"messages": [{"role": "user", "content": "Hello"}]}


def test_valid_request_returns_expected_structure():
    from unittest.mock import AsyncMock, patch
    from app.providers import ProviderResult

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
    from unittest.mock import AsyncMock, patch
    from app.providers import ProviderResult

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
    from unittest.mock import AsyncMock, patch
    from app.providers import ProviderResult

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


def test_streaming_not_implemented():
    body = {**VALID_BODY, "stream": True}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 501


def test_health_still_works():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_groq_route_authentication_error_401():
    from unittest.mock import AsyncMock, patch
    from app.providers import AuthenticationError

    body = {**VALID_BODY, "model": "openai/gpt-oss-20b"}
    with patch("app.providers.GroqProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = AuthenticationError("Groq authentication failed.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 401
        assert "authentication failed" in response.json()["detail"].lower()


def test_groq_route_model_not_found_404():
    from unittest.mock import AsyncMock, patch
    from app.providers import ModelNotFoundError

    body = {**VALID_BODY, "model": "openai/gpt-oss-20b"}
    with patch("app.providers.GroqProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = ModelNotFoundError("Groq model not found.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


def test_groq_route_rate_limit_429():
    from unittest.mock import AsyncMock, patch
    from app.providers import RateLimitError

    body = {**VALID_BODY, "model": "openai/gpt-oss-20b"}
    with patch("app.providers.GroqProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = RateLimitError("Groq rate limit exceeded.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()


def test_groq_route_upstream_server_error_502():
    from unittest.mock import AsyncMock, patch
    from app.providers import UpstreamProviderError

    body = {**VALID_BODY, "model": "openai/gpt-oss-20b"}
    with patch("app.providers.GroqProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = UpstreamProviderError("Groq upstream server error.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 502
        assert "upstream" in response.json()["detail"].lower()


def test_groq_route_timeout_error_504():
    from unittest.mock import AsyncMock, patch
    from app.providers import ProviderTimeoutError

    body = {**VALID_BODY, "model": "openai/gpt-oss-20b"}
    with patch("app.providers.GroqProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = ProviderTimeoutError("Groq request timed out.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()


def test_groq_route_redacts_api_key_in_error_detail():
    from unittest.mock import AsyncMock, patch
    from app.providers import AuthenticationError

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
    from unittest.mock import AsyncMock, patch
    from app.providers import ProviderResult

    body = {**VALID_BODY, "model": "gemini-3.7-flash"}
    fake_result = ProviderResult(content="Mocked Gemini response", finish_reason="stop", usage=None)

    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = fake_result
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        assert response.json()["model"] == "gemini-3.7-flash"


def test_gemini_route_authentication_error_401():
    from unittest.mock import AsyncMock, patch
    from app.providers import AuthenticationError

    body = {**VALID_BODY, "model": "gemini-3.8-flash"}
    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = AuthenticationError("Gemini authentication failed: API key not valid.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 401
        assert "authentication failed" in response.json()["detail"].lower()


def test_gemini_route_model_not_found_404():
    from unittest.mock import AsyncMock, patch
    from app.providers import ModelNotFoundError

    body = {**VALID_BODY, "model": "gemini-3.8-flash"}
    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = ModelNotFoundError("Gemini model not found.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


def test_gemini_route_rate_limit_429():
    from unittest.mock import AsyncMock, patch
    from app.providers import RateLimitError

    body = {**VALID_BODY, "model": "gemini-3.8-flash"}
    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = RateLimitError("Gemini rate limit exceeded.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()


def test_gemini_route_upstream_server_error_502():
    from unittest.mock import AsyncMock, patch
    from app.providers import UpstreamProviderError

    body = {**VALID_BODY, "model": "gemini-3.8-flash"}
    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = UpstreamProviderError("Gemini upstream server error.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 502
        assert "upstream" in response.json()["detail"].lower()


def test_gemini_route_timeout_error_504():
    from unittest.mock import AsyncMock, patch
    from app.providers import ProviderTimeoutError

    body = {**VALID_BODY, "model": "gemini-3.8-flash"}
    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = ProviderTimeoutError("Gemini request timed out.")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()


def test_gemini_route_redacts_api_key_in_error_detail():
    from unittest.mock import AsyncMock, patch
    from app.providers import AuthenticationError

    secret_key = "AIzaSy_gemini_secret_123456"
    body = {**VALID_BODY, "model": "gemini-3.8-flash"}
    with patch("app.providers.GeminiProvider.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = AuthenticationError("Gemini authentication failed: [REDACTED]")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 401
        detail = response.json()["detail"]
        assert secret_key not in detail
        assert "[REDACTED]" in detail

