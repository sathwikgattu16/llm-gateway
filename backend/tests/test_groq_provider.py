import json
import pytest
import httpx

from app.providers import (
    AuthenticationError,
    GroqProvider,
    ModelNotFoundError,
    Provider,
    ProviderError,
    ProviderResult,
    ProviderTimeoutError,
    RateLimitError,
    UpstreamProviderError,
    get_provider,
)
from app.schemas import ChatMessage, Usage


MOCK_GROQ_SUCCESS_JSON = {
    "id": "chatcmpl-test-123",
    "object": "chat.completion",
    "created": 1729864200,
    "model": "openai/gpt-oss-20b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Paris is the capital of France.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    },
}


def test_groq_provider_implements_provider_interface():
    provider = GroqProvider(api_key="test-key")
    assert isinstance(provider, Provider)


def test_groq_registered_in_get_provider():
    provider = get_provider("groq")
    assert isinstance(provider, GroqProvider)


@pytest.mark.anyio
async def test_groq_success_normalization():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.url.path == "/openai/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-oss-20b"
        assert len(body["messages"]) == 1
        return httpx.Response(200, json=MOCK_GROQ_SUCCESS_JSON)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    result = await provider.generate(
        model_id="openai/gpt-oss-20b",
        messages=[ChatMessage(role="user", content="What is the capital of France?")],
    )

    assert isinstance(result, ProviderResult)
    assert result.content == "Paris is the capital of France."
    assert result.finish_reason == "stop"
    assert isinstance(result.usage, Usage)
    assert result.usage.prompt_tokens == 12
    assert result.usage.completion_tokens == 8
    assert result.usage.total_tokens == 20


@pytest.mark.anyio
async def test_groq_system_message_passed_in_messages():
    captured_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        captured_body = json.loads(request.content)
        return httpx.Response(200, json=MOCK_GROQ_SUCCESS_JSON)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello!"),
    ]

    await provider.generate(
        model_id="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.7,
        max_tokens=100,
    )

    assert captured_body["model"] == "openai/gpt-oss-20b"
    assert captured_body["temperature"] == 0.7
    assert captured_body["max_tokens"] == 100
    assert len(captured_body["messages"]) == 2
    assert captured_body["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert captured_body["messages"][1] == {"role": "user", "content": "Hello!"}


@pytest.mark.anyio
async def test_groq_missing_api_key_raises_authentication_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    provider = GroqProvider(api_key=None)

    with pytest.raises(AuthenticationError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 401
    assert "GROQ_API_KEY" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_authentication_error_401():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "Invalid API Key provided.", "type": "invalid_request_error"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="bad-key", http_client=mock_client)

    with pytest.raises(AuthenticationError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 401
    assert "Invalid API Key" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_invalid_model_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": {"message": "The model `does-not-exist` does not exist", "code": "model_not_found"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(ModelNotFoundError) as exc_info:
        await provider.generate(
            model_id="does-not-exist",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 404
    assert "does-not-exist" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_rate_limit_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "Rate limit exceeded: TPM limit hit.", "type": "tokens"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(RateLimitError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_timeout_raises_provider_timeout_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Connection timed out after 30s")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(ProviderTimeoutError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 504
    assert "timed out" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_groq_upstream_server_error_500():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": {"message": "Internal server error occurred."}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert "Groq upstream error" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_network_connect_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Network unreachable")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert "Failed to communicate with Groq" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_never_falls_back_to_fake_provider():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "Down"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(ProviderError):
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Test")],
        )


@pytest.mark.anyio
async def test_groq_reads_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-secret-key-12345")
    captured_auth = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_auth
        captured_auth = request.headers.get("Authorization")
        return httpx.Response(200, json=MOCK_GROQ_SUCCESS_JSON)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key=None, http_client=mock_client)

    result = await provider.generate(
        model_id="openai/gpt-oss-20b",
        messages=[ChatMessage(role="user", content="Hello")],
    )

    assert result.content == "Paris is the capital of France."
    assert captured_auth == "Bearer env-secret-key-12345"


@pytest.mark.anyio
async def test_groq_explicit_key_takes_precedence_over_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-key-should-be-ignored")
    captured_auth = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_auth
        captured_auth = request.headers.get("Authorization")
        return httpx.Response(200, json=MOCK_GROQ_SUCCESS_JSON)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="explicit-key-chosen", http_client=mock_client)

    await provider.generate(
        model_id="openai/gpt-oss-20b",
        messages=[ChatMessage(role="user", content="Hello")],
    )

    assert captured_auth == "Bearer explicit-key-chosen"


@pytest.mark.anyio
async def test_groq_missing_key_makes_no_http_request(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    http_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal http_called
        http_called = True
        return httpx.Response(200, json=MOCK_GROQ_SUCCESS_JSON)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key=None, http_client=mock_client)

    with pytest.raises(AuthenticationError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hello")],
        )

    assert exc_info.value.status_code == 401
    assert "GROQ_API_KEY" in exc_info.value.detail
    assert not http_called, "No HTTP request should be sent when API key is missing"


@pytest.mark.anyio
@pytest.mark.parametrize("empty_val", ["", "   ", "\t\n"])
async def test_groq_empty_or_whitespace_key_raises_without_http(monkeypatch, empty_val):
    monkeypatch.setenv("GROQ_API_KEY", empty_val)
    http_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal http_called
        http_called = True
        return httpx.Response(200, json=MOCK_GROQ_SUCCESS_JSON)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key=None, http_client=mock_client)

    with pytest.raises(AuthenticationError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hello")],
        )

    assert exc_info.value.status_code == 401
    assert "GROQ_API_KEY" in exc_info.value.detail
    assert not http_called


def test_groq_never_leaks_api_key_in_repr():
    secret_key = "super-secret-key-do-not-leak"
    provider = GroqProvider(api_key=secret_key)

    assert secret_key not in repr(provider)
    assert secret_key not in str(provider)
    assert repr(provider) == "<GroqProvider base_url='https://api.groq.com/openai/v1'>"


@pytest.mark.anyio
async def test_groq_never_leaks_api_key_in_error_response():
    sensitive_key = "gsk_sensitive_key_val_999"

    def handler(request: httpx.Request) -> httpx.Response:
        # Simulate an upstream error message that echoes the sensitive key
        return httpx.Response(
            401,
            json={"error": {"message": f"Invalid API Key provided: {sensitive_key}"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key=sensitive_key, http_client=mock_client)

    with pytest.raises(AuthenticationError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hello")],
        )

    error_detail = exc_info.value.detail
    assert sensitive_key not in error_detail
    assert sensitive_key not in str(exc_info.value)
    assert "[REDACTED]" in error_detail


@pytest.mark.anyio
async def test_groq_never_leaks_api_key_in_network_exceptions():
    sensitive_key = "gsk_network_secret_xyz"

    def handler(request: httpx.Request) -> httpx.Response:
        # Simulate a connection error that echoes the authorization / key
        raise httpx.ConnectError(f"Connection failed for key {sensitive_key}")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key=sensitive_key, http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hello")],
        )

    error_detail = exc_info.value.detail
    assert sensitive_key not in error_detail
    assert sensitive_key not in str(exc_info.value)
    assert "[REDACTED]" in error_detail


def test_gitignore_excludes_secret_files():
    import pathlib
    gitignore_path = pathlib.Path(__file__).resolve().parents[2] / ".gitignore"
    assert gitignore_path.exists(), ".gitignore must exist in project root"
    lines = [line.strip() for line in gitignore_path.read_text().splitlines() if line.strip()]

    assert ".env" in lines
    assert ".env.*" in lines


@pytest.mark.anyio
async def test_groq_invalid_json_response_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>502 Bad Gateway</html>", headers={"content-type": "text/html"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert "invalid JSON" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_non_object_json_response_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="[1, 2, 3]", headers={"content-type": "application/json"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert "expected JSON object" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_missing_choices_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "chatcmpl-123"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert "missing choices" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_empty_choices_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "chatcmpl-123", "choices": []})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert "missing choices" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_non_list_choices_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "chatcmpl-123", "choices": "not-a-list"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert "missing choices" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_malformed_choice_item_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [None]})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert "malformed choice object" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_malformed_message_in_choice_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": None}]})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert "malformed message object" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_none_content_in_message_normalized_to_empty_string():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": None},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    result = await provider.generate(
        model_id="openai/gpt-oss-20b",
        messages=[ChatMessage(role="user", content="Hi")],
    )

    assert result.content == ""
    assert result.finish_reason == "stop"
    assert result.usage is None


@pytest.mark.anyio
async def test_groq_missing_usage_remains_none():
    data_without_usage = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "No usage info"},
                "finish_reason": "stop",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=data_without_usage)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    result = await provider.generate(
        model_id="openai/gpt-oss-20b",
        messages=[ChatMessage(role="user", content="Hi")],
    )

    assert result.usage is None


@pytest.mark.anyio
async def test_groq_null_usage_remains_none():
    data_with_null_usage = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Null usage"},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=data_with_null_usage)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    result = await provider.generate(
        model_id="openai/gpt-oss-20b",
        messages=[ChatMessage(role="user", content="Hi")],
    )

    assert result.usage is None


@pytest.mark.anyio
async def test_groq_empty_usage_dict_remains_none():
    data_with_empty_usage = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Empty usage dict"},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=data_with_empty_usage)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    result = await provider.generate(
        model_id="openai/gpt-oss-20b",
        messages=[ChatMessage(role="user", content="Hi")],
    )

    assert result.usage is None


@pytest.mark.anyio
async def test_groq_partial_usage_never_fakes_zeros():
    data_partial_usage = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Partial usage"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": None,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=data_partial_usage)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    result = await provider.generate(
        model_id="openai/gpt-oss-20b",
        messages=[ChatMessage(role="user", content="Hi")],
    )

    assert isinstance(result.usage, Usage)
    assert result.usage.prompt_tokens == 15
    assert result.usage.completion_tokens is None
    assert result.usage.total_tokens is None


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [502, 503, 504])
async def test_groq_upstream_5xx_server_errors_raise_502(status_code):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"error": {"message": f"Server unavailable ({status_code})"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert f"Groq upstream error ({status_code})" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_connect_timeout_raises_provider_timeout_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("Connect timeout to Groq endpoint")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(ProviderTimeoutError) as exc_info:
        await provider.generate(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 504
    assert "timed out" in exc_info.value.detail.lower()


# --- Milestone 5: Groq Streaming Tests ---

from app.providers import ProviderChunk


GROQ_STREAMING_MOCK_CHUNKS = (
    'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"role":"assistant","content":"Paris"},"finish_reason":null}]}\n\n'
    ': keep-alive\n\n'
    'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}\n\n'
    '\n\n'
    'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":" the capital."},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":6,"total_tokens":18}}\n\n'
    'data: [DONE]\n\n'
    'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"ignored"},"finish_reason":null}]}\n\n'
)


@pytest.mark.anyio
async def test_groq_streaming_success_normalization():
    captured_payload = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.url.path == "/openai/v1/chat/completions"
        captured_payload = json.loads(request.content)
        return httpx.Response(200, text=GROQ_STREAMING_MOCK_CHUNKS)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    chunks = []
    async for chunk in provider.generate_stream(
        model_id="openai/gpt-oss-20b",
        messages=[ChatMessage(role="user", content="Capital of France?")],
        temperature=0.3,
        max_tokens=50,
    ):
        assert isinstance(chunk, ProviderChunk)
        chunks.append(chunk)

    assert captured_payload["stream"] is True
    assert captured_payload["model"] == "openai/gpt-oss-20b"
    assert captured_payload["temperature"] == 0.3
    assert captured_payload["max_tokens"] == 50

    # 3 chunks: "Paris", " is", " the capital."
    assert len(chunks) == 3
    assert chunks[0].delta == "Paris"
    assert chunks[0].finish_reason is None
    assert chunks[0].usage is None

    assert chunks[1].delta == " is"
    assert chunks[1].finish_reason is None
    assert chunks[1].usage is None

    assert chunks[2].delta == " the capital."
    assert chunks[2].finish_reason == "stop"
    assert chunks[2].usage is not None
    assert chunks[2].usage.prompt_tokens == 12
    assert chunks[2].usage.completion_tokens == 6
    assert chunks[2].usage.total_tokens == 18


@pytest.mark.anyio
async def test_groq_streaming_missing_key_raises_auth_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    provider = GroqProvider(api_key=None)

    with pytest.raises(AuthenticationError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 401
    assert "GROQ_API_KEY" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_streaming_auth_error_401():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "Invalid API Key provided.", "type": "invalid_request_error"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="bad-key", http_client=mock_client)

    with pytest.raises(AuthenticationError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 401
    assert "Invalid API Key" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_streaming_model_not_found_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": {"message": "The model `missing-model` does not exist", "code": "model_not_found"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(ModelNotFoundError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="missing-model",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 404
    assert "missing-model" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_streaming_rate_limit_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "Rate limit reached.", "type": "tokens"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(RateLimitError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 429
    assert "rate limit" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_groq_streaming_upstream_500():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": {"message": "Groq internal server crash."}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 502
    assert "upstream error" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_groq_streaming_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Stream read timeout after 30s")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(ProviderTimeoutError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 504
    assert "timed out" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_groq_streaming_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Network connection reset")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 502
    assert "failed to communicate" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_groq_streaming_malformed_json_chunk():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="data: {invalid json content}\n\n")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 502
    assert "invalid JSON" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_streaming_non_dict_chunk():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="data: [\"item1\", \"item2\"]\n\n")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 502
    assert "expected JSON object" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_streaming_redacts_api_key_in_exceptions():
    secret_key = "gsk_stream_secret_key_112233"

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"Read timed out connecting with key {secret_key}")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler))
    provider = GroqProvider(api_key=secret_key, http_client=mock_client)

    with pytest.raises(ProviderTimeoutError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="openai/gpt-oss-20b",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert secret_key not in exc_info.value.detail
    assert "[REDACTED]" in exc_info.value.detail


@pytest.mark.anyio
async def test_groq_streaming_cleanup_on_early_termination():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=GROQ_STREAMING_MOCK_CHUNKS)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(api_key="test-key", http_client=mock_client)

    stream = provider.generate_stream(
        model_id="openai/gpt-oss-20b",
        messages=[ChatMessage(role="user", content="Hi")],
    )

    first = await anext(stream)
    assert first.delta == "Paris"
    # Close early (simulating disconnect or break)
    await stream.aclose()
