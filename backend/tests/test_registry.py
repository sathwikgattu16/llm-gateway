import pytest

from app.registry import ModelConfig, ModelRegistry, get_registry, load_registry

EXPECTED_IDS = {
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
}


def test_default_registry_has_six_models():
    ids = {m.id for m in get_registry().list_models()}
    assert ids == EXPECTED_IDS


def test_get_known_and_unknown():
    reg = load_registry()
    assert reg.get("openai/gpt-oss-20b").provider == "groq"
    assert reg.get("nope") is None


def test_auto_resolves_to_first_model():
    reg = load_registry()
    assert reg.resolve("auto").id == "gemini-3.8-flash"


def test_duplicate_ids_rejected():
    m = ModelConfig(id="x", provider="groq")
    with pytest.raises(ValueError):
        ModelRegistry([m, m])


def test_empty_registry_rejected():
    with pytest.raises(ValueError):
        ModelRegistry([])


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        ModelConfig(id="x", provider="openai")
