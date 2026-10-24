from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_BODY = {"messages": [{"role": "user", "content": "Hello"}]}


def test_valid_request_returns_expected_structure():
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


def test_explicit_model_is_used():
    body = {**VALID_BODY, "model": "openai/gpt-oss-20b"}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.json()["model"] == "openai/gpt-oss-20b"


def test_unknown_model_returns_404():
    body = {**VALID_BODY, "model": "does-not-exist"}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


def test_usage_is_unknown_not_fake():
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
