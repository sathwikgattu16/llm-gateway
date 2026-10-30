import json
import pytest
import httpx

from app.providers import (
    AuthenticationError,
    GeminiProvider,
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

MOCK_GEMINI_SUCCESS_JSON = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": "Tokyo is the capital of Japan."}
                ],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 14,
        "candidatesTokenCount": 8,
        "totalTokenCount": 22,
    },
    "modelVersion": "gemini-3.8-flash",
}


def test_gemini_provider_implements_provider_interface():
    provider = GeminiProvider(api_key="test-key")
    assert isinstance(provider, Provider)


def test_gemini_registered_in_get_provider():
    provider = get_provider("gemini")
    assert isinstance(provider, GeminiProvider)


def test_gemini_repr():
    provider = GeminiProvider(api_key="test-key")
    assert "<GeminiProvider" in repr(provider)
    assert "googleapis.com" in repr(provider)


@pytest.mark.anyio
async def test_gemini_success_normalization():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "test-key"
        assert request.headers["Content-Type"] == "application/json"
        assert request.url.path == "/v1beta/models/gemini-3.8-flash:generateContent"
        body = json.loads(request.content)
        assert "contents" in body
        assert len(body["contents"]) == 1
        assert body["contents"][0]["role"] == "user"
        assert body["contents"][0]["parts"][0]["text"] == "What is the capital of Japan?"
        return httpx.Response(200, json=MOCK_GEMINI_SUCCESS_JSON)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    result = await provider.generate(
        model_id="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="What is the capital of Japan?")],
    )

    assert isinstance(result, ProviderResult)
    assert result.content == "Tokyo is the capital of Japan."
    assert result.finish_reason == "stop"
    assert isinstance(result.usage, Usage)
    assert result.usage.prompt_tokens == 14
    assert result.usage.completion_tokens == 8
    assert result.usage.total_tokens == 22


@pytest.mark.anyio
async def test_gemini_model_id_prefix_handling():
    captured_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(request.url.path)
        return httpx.Response(200, json=MOCK_GEMINI_SUCCESS_JSON)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    # Without models/ prefix
    await provider.generate(
        model_id="gemini-3.7-flash",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    assert captured_urls[-1] == "/v1beta/models/gemini-3.7-flash:generateContent"

    # With models/ prefix
    await provider.generate(
        model_id="models/gemini-3.7-flash",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    assert captured_urls[-1] == "/v1beta/models/gemini-3.7-flash:generateContent"


@pytest.mark.anyio
async def test_gemini_system_instruction_and_roles_mapping():
    captured_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        captured_body = json.loads(request.content)
        return httpx.Response(200, json=MOCK_GEMINI_SUCCESS_JSON)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    messages = [
        ChatMessage(role="system", content="You are a helpful geography teacher."),
        ChatMessage(role="user", content="Hi!"),
        ChatMessage(role="assistant", content="Hello! How can I help?"),
        ChatMessage(role="user", content="What is Mount Everest?"),
    ]

    await provider.generate(
        model_id="gemini-3.8-flash",
        messages=messages,
        temperature=0.8,
        max_tokens=200,
    )

    # Verify systemInstruction
    assert "systemInstruction" in captured_body
    assert captured_body["systemInstruction"] == {
        "parts": [{"text": "You are a helpful geography teacher."}]
    }

    # Verify contents (roles: user -> user, assistant -> model)
    contents = captured_body["contents"]
    assert len(contents) == 3
    assert contents[0] == {"role": "user", "parts": [{"text": "Hi!"}]}
    assert contents[1] == {"role": "model", "parts": [{"text": "Hello! How can I help?"}]}
    assert contents[2] == {"role": "user", "parts": [{"text": "What is Mount Everest?"}]}

    # Verify generationConfig
    assert captured_body["generationConfig"] == {
        "temperature": 0.8,
        "maxOutputTokens": 200,
    }


@pytest.mark.anyio
async def test_gemini_only_system_message_fallback():
    captured_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        captured_body = json.loads(request.content)
        return httpx.Response(200, json=MOCK_GEMINI_SUCCESS_JSON)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    await provider.generate(
        model_id="gemini-3.8-flash",
        messages=[ChatMessage(role="system", content="Initialize system.")],
    )

    assert "systemInstruction" in captured_body
    assert len(captured_body["contents"]) == 1
    assert captured_body["contents"][0]["role"] == "user"


@pytest.mark.anyio
async def test_gemini_finish_reason_mappings():
    test_cases = [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("RECITATION", "content_filter"),
        ("BLOCKLIST", "content_filter"),
        ("PROHIBITED_CONTENT", "content_filter"),
        ("SPII", "content_filter"),
        ("OTHER", "other"),
        (None, "stop"),
    ]

    for gemini_reason, expected_finish_reason in test_cases:
        def handler(request: httpx.Request, r=gemini_reason) -> httpx.Response:
            resp_data = {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Response text"}]},
                        "finishReason": r,
                    }
                ]
            }
            return httpx.Response(200, json=resp_data)

        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = GeminiProvider(api_key="test-key", http_client=mock_client)

        result = await provider.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert result.finish_reason == expected_finish_reason


@pytest.mark.anyio
async def test_gemini_multiple_content_parts_concatenation():
    def handler(request: httpx.Request) -> httpx.Response:
        resp_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Part 1. "},
                            {"text": "Part 2."},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        return httpx.Response(200, json=resp_data)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    result = await provider.generate(
        model_id="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    assert result.content == "Part 1. Part 2."


@pytest.mark.anyio
async def test_gemini_missing_usage_metadata_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        resp_data = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "No usage info"}]},
                    "finishReason": "STOP",
                }
            ]
        }
        return httpx.Response(200, json=resp_data)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    result = await provider.generate(
        model_id="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    assert result.usage is None


@pytest.mark.anyio
async def test_gemini_partial_usage_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        resp_data = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Partial usage"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
            },
        }
        return httpx.Response(200, json=resp_data)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    result = await provider.generate(
        model_id="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    assert result.usage is not None
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens is None
    assert result.usage.total_tokens is None


@pytest.mark.anyio
async def test_gemini_missing_api_key_raises_authentication_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiProvider(api_key=None)

    with pytest.raises(AuthenticationError) as exc_info:
        await provider.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 401
    assert "GEMINI_API_KEY" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_authentication_error_401():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"code": 401, "message": "API key not valid. Please pass a valid API key.", "status": "UNAUTHENTICATED"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="bad-key", http_client=mock_client)

    with pytest.raises(AuthenticationError) as exc_info:
        await provider.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 401
    assert "API key not valid" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_authentication_error_403():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"error": {"code": 403, "message": "Method not allowed for this project.", "status": "PERMISSION_DENIED"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="bad-key", http_client=mock_client)

    with pytest.raises(AuthenticationError) as exc_info:
        await provider.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 401
    assert "authentication failed" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_gemini_invalid_model_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": {"code": 404, "message": "models/non-existent is not found for API version v1beta", "status": "NOT_FOUND"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(ModelNotFoundError) as exc_info:
        await provider.generate(
            model_id="non-existent",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 404
    assert "non-existent" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_model_not_found_in_400():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"code": 400, "message": "models/bad-model not found", "status": "INVALID_ARGUMENT"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(ModelNotFoundError) as exc_info:
        await provider.generate(
            model_id="bad-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 404
    assert "bad-model" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_generic_400_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"code": 400, "message": "Invalid temperature value", "status": "INVALID_ARGUMENT"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(ProviderError) as exc_info:
        await provider.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 400
    assert "Invalid temperature" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_rate_limit_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota).", "status": "RESOURCE_EXHAUSTED"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(RateLimitError) as exc_info:
        await provider.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 429
    assert "rate limit exceeded" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_gemini_timeout_raises_provider_timeout_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out after 30 seconds")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(ProviderTimeoutError) as exc_info:
        await provider.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 504
    assert "timed out" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_gemini_network_error_raises_upstream_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused by upstream host")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert "failed to communicate" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_gemini_upstream_server_error_500_503():
    for status_code in (500, 503):
        def handler(request: httpx.Request, sc=status_code) -> httpx.Response:
            return httpx.Response(
                sc,
                json={"error": {"code": sc, "message": "An internal error occurred", "status": "INTERNAL"}},
            )

        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = GeminiProvider(api_key="test-key", http_client=mock_client)

        with pytest.raises(UpstreamProviderError) as exc_info:
            await provider.generate(
                model_id="gemini-3.8-flash",
                messages=[ChatMessage(role="user", content="Hi")],
            )

        assert exc_info.value.status_code == 502
        assert "upstream error" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_gemini_prompt_blocked_safety_reason():
    def handler(request: httpx.Request) -> httpx.Response:
        resp_data = {
            "promptFeedback": {
                "blockReason": "SAFETY",
                "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS", "probability": "HIGH"}],
            }
        }
        return httpx.Response(200, json=resp_data)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Blocked content")],
        )

    assert exc_info.value.status_code == 502
    assert "SAFETY" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_malformed_json_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"Not valid JSON{{{")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert "invalid JSON" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_non_dict_json_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"[\"item1\", \"item2\"]")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert "expected JSON object" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_missing_candidates_list():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert "missing candidates" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_malformed_candidate_element():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": ["not-a-dict"]})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        await provider.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert exc_info.value.status_code == 502
    assert "malformed candidate object" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_redacts_api_key_in_exception_messages():
    secret_key = "AIzaSy_SuperSecretGeminiKey_123456"

    # In timeout
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"Read timed out connecting with key {secret_key}")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler))
    provider = GeminiProvider(api_key=secret_key, http_client=mock_client)

    with pytest.raises(ProviderTimeoutError) as exc_info:
        await provider.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert secret_key not in exc_info.value.detail
    assert "[REDACTED]" in exc_info.value.detail

    # In HTTP error response
    def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"code": 401, "message": f"Invalid key: {secret_key}", "status": "UNAUTHENTICATED"}},
        )

    mock_client2 = httpx.AsyncClient(transport=httpx.MockTransport(error_handler))
    provider2 = GeminiProvider(api_key=secret_key, http_client=mock_client2)

    with pytest.raises(AuthenticationError) as exc_info2:
        await provider2.generate(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        )

    assert secret_key not in exc_info2.value.detail
    assert "[REDACTED]" in exc_info2.value.detail


# --- Milestone 5: Gemini Streaming Tests ---

from app.providers import ProviderChunk


GEMINI_STREAMING_MOCK_CHUNKS = (
    'data: {"candidates":[{"content":{"parts":[{"text":"Tokyo"}],"role":"model"}}]}\n\n'
    ': ping\n\n'
    'data: {"candidates":[{"content":{"parts":[{"text":" is the capital of Japan."}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":14,"candidatesTokenCount":8,"totalTokenCount":22}}\n\n'
    '\n\n'
)


@pytest.mark.anyio
async def test_gemini_streaming_success_normalization():
    captured_url = None
    captured_payload = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_url, captured_payload
        assert request.headers["x-goog-api-key"] == "test-key"
        assert request.headers["Content-Type"] == "application/json"
        captured_url = str(request.url)
        captured_payload = json.loads(request.content)
        return httpx.Response(200, text=GEMINI_STREAMING_MOCK_CHUNKS)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    chunks = []
    async for chunk in provider.generate_stream(
        model_id="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="Capital of Japan?")],
        temperature=0.4,
        max_tokens=60,
    ):
        assert isinstance(chunk, ProviderChunk)
        chunks.append(chunk)

    assert "models/gemini-3.8-flash:streamGenerateContent?alt=sse" in captured_url
    assert captured_payload["generationConfig"]["temperature"] == 0.4
    assert captured_payload["generationConfig"]["maxOutputTokens"] == 60
    assert captured_payload["contents"][0]["parts"][0]["text"] == "Capital of Japan?"

    assert len(chunks) == 2
    assert chunks[0].delta == "Tokyo"
    assert chunks[0].finish_reason is None
    assert chunks[0].usage is None

    assert chunks[1].delta == " is the capital of Japan."
    assert chunks[1].finish_reason == "stop"
    assert chunks[1].usage is not None
    assert chunks[1].usage.prompt_tokens == 14
    assert chunks[1].usage.completion_tokens == 8
    assert chunks[1].usage.total_tokens == 22


@pytest.mark.anyio
async def test_gemini_streaming_model_prefix_handling():
    captured_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        return httpx.Response(200, text=GEMINI_STREAMING_MOCK_CHUNKS)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    async for _ in provider.generate_stream(
        model_id="gemini-3.7-flash",
        messages=[ChatMessage(role="user", content="Hi")],
    ):
        pass
    assert captured_paths[-1] == "/v1beta/models/gemini-3.7-flash:streamGenerateContent"

    async for _ in provider.generate_stream(
        model_id="models/gemini-3.7-flash",
        messages=[ChatMessage(role="user", content="Hi")],
    ):
        pass
    assert captured_paths[-1] == "/v1beta/models/gemini-3.7-flash:streamGenerateContent"


@pytest.mark.anyio
async def test_gemini_streaming_system_instruction_and_roles_mapping():
    captured_payload = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json.loads(request.content)
        return httpx.Response(200, text=GEMINI_STREAMING_MOCK_CHUNKS)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    messages = [
        ChatMessage(role="system", content="System instruction text"),
        ChatMessage(role="user", content="User text"),
        ChatMessage(role="assistant", content="Assistant reply"),
    ]

    async for _ in provider.generate_stream(
        model_id="gemini-3.8-flash",
        messages=messages,
    ):
        pass

    assert captured_payload["systemInstruction"] == {"parts": [{"text": "System instruction text"}]}
    contents = captured_payload["contents"]
    assert len(contents) == 2
    assert contents[0] == {"role": "user", "parts": [{"text": "User text"}]}
    assert contents[1] == {"role": "model", "parts": [{"text": "Assistant reply"}]}


@pytest.mark.anyio
async def test_gemini_streaming_finish_reason_mappings():
    test_cases = [
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("RECITATION", "content_filter"),
        ("BLOCKLIST", "content_filter"),
        ("PROHIBITED_CONTENT", "content_filter"),
        ("SPII", "content_filter"),
        ("OTHER", "other"),
    ]

    for gemini_reason, expected in test_cases:
        sse_text = f'data: {{"candidates": [{{"content": {{"parts": [{{"text": "Text"}}]}}, "finishReason": "{gemini_reason}"}}]}}\n\n'

        def handler(request: httpx.Request, payload=sse_text) -> httpx.Response:
            return httpx.Response(200, text=payload)

        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = GeminiProvider(api_key="test-key", http_client=mock_client)

        chunks = []
        async for chunk in provider.generate_stream(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            chunks.append(chunk)

        assert chunks[0].finish_reason == expected


@pytest.mark.anyio
async def test_gemini_streaming_prompt_blocked_safety():
    blocked_sse = (
        'data: {"promptFeedback": {"blockReason": "SAFETY", "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS", "probability": "HIGH"}]}}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=blocked_sse)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Blocked query")],
        ):
            pass

    assert exc_info.value.status_code == 502
    assert "SAFETY" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_streaming_missing_api_key_raises_auth_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiProvider(api_key=None)

    with pytest.raises(AuthenticationError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 401
    assert "GEMINI_API_KEY" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_streaming_auth_error_401():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"code": 401, "message": "API key not valid.", "status": "UNAUTHENTICATED"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="bad-key", http_client=mock_client)

    with pytest.raises(AuthenticationError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 401
    assert "API key not valid" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_streaming_auth_error_403():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"error": {"code": 403, "message": "Permission denied for project.", "status": "PERMISSION_DENIED"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="bad-key", http_client=mock_client)

    with pytest.raises(AuthenticationError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 401
    assert "authentication failed" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_gemini_streaming_invalid_model_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": {"code": 404, "message": "models/non-existent is not found", "status": "NOT_FOUND"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(ModelNotFoundError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="non-existent",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 404
    assert "non-existent" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_streaming_rate_limit_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(RateLimitError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 429
    assert "rate limit" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_gemini_streaming_upstream_500():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": {"code": 500, "message": "Internal error occurred", "status": "INTERNAL"}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 502
    assert "upstream error" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_gemini_streaming_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Stream read timed out")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(ProviderTimeoutError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 504
    assert "timed out" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_gemini_streaming_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Network connection reset")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 502
    assert "failed to communicate" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_gemini_streaming_malformed_json_chunk():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="data: {invalid json\n\n")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 502
    assert "invalid JSON" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_streaming_non_dict_chunk():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="data: [\"chunk1\", \"chunk2\"]\n\n")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    with pytest.raises(UpstreamProviderError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert exc_info.value.status_code == 502
    assert "expected JSON object" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_streaming_redacts_api_key_in_exceptions():
    secret_key = "AIzaSy_stream_gemini_secret_998877"

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"Read timed out connecting with key {secret_key}")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler))
    provider = GeminiProvider(api_key=secret_key, http_client=mock_client)

    with pytest.raises(ProviderTimeoutError) as exc_info:
        async for _ in provider.generate_stream(
            model_id="gemini-3.8-flash",
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            pass

    assert secret_key not in exc_info.value.detail
    assert "[REDACTED]" in exc_info.value.detail


@pytest.mark.anyio
async def test_gemini_streaming_cleanup_on_early_termination():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=GEMINI_STREAMING_MOCK_CHUNKS)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", http_client=mock_client)

    stream = provider.generate_stream(
        model_id="gemini-3.8-flash",
        messages=[ChatMessage(role="user", content="Hi")],
    )

    first = await anext(stream)
    assert first.delta == "Tokyo"
    # Close early (simulating disconnect or break)
    await stream.aclose()
