from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


class ModelConfig(BaseModel):
    id: str
    provider: Literal["gemini", "groq"]


class ModelRegistry:
    def __init__(self, models: list[ModelConfig]):
        ids = [m.id for m in models]
        if not models:
            raise ValueError("Model registry is empty.")
        if len(ids) != len(set(ids)):
            raise ValueError("Model registry contains duplicate IDs.")
        self._models = {m.id: m for m in models}

    def list_models(self) -> list[ModelConfig]:
        return list(self._models.values())

    def get(self, model_id: str) -> ModelConfig | None:
        return self._models.get(model_id)

    def resolve(self, requested: str) -> ModelConfig | None:
        """Map a requested model name to a configured model.

        PLACEHOLDER: "auto" picks the first configured model. Real
        routing replaces this in a later milestone.
        """
        if requested == "auto":
            return self.list_models()[0]
        return self.get(requested)


def load_registry(path: Path = DEFAULT_PATH) -> ModelRegistry:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    models = [ModelConfig(**item) for item in data["models"]]
    return ModelRegistry(models)


@lru_cache
def get_registry() -> ModelRegistry:
    return load_registry()
