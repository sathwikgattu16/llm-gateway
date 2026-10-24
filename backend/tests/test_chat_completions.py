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
    assert data["model"] == "auto"
    assert isinstance(data["created"], int)
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] == "stop"
    assert set(data["usage"]) == {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }


def test_explicit_model_is_echoed():
    body = {**VALID_BODY, "model": "some-model"}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.json()["model"] == "some-model"


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
