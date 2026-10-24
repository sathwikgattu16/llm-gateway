import time
import uuid

from fastapi import APIRouter, HTTPException

from app.providers import get_provider
from app.registry import get_registry
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ModelCard,
    ModelList,
    ResponseMessage,
)

router = APIRouter()


@router.get("/v1/models", response_model=ModelList)
def list_models():
    models = get_registry().list_models()
    return ModelList(
        data=[ModelCard(id=m.id, owned_by=m.provider) for m in models]
    )


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(
            status_code=501,
            detail="Streaming is not implemented yet.",
        )

    model = get_registry().resolve(request.model)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' is not configured.",
        )

    provider = get_provider(model.provider)
    result = await provider.generate(
        model_id=model.id,
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model.id,
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(content=result.content),
                finish_reason=result.finish_reason,
            )
        ],
        usage=result.usage,
    )