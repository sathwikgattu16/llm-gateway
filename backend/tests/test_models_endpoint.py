from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_list_models_returns_six():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 6


def test_model_card_shape():
    card = client.get("/v1/models").json()["data"][0]
    assert card["object"] == "model"
    assert card["id"] == "gemini-3.8-flash"
    assert card["owned_by"] == "gemini"