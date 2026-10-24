import time
import uuid

from fastapi import APIRouter, HTTPException

from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ResponseMessage,
    Usage,
)

router = APIRouter()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def create_chat_completion(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(
            status_code=501,
            detail="Streaming is not implemented yet.",
        )

    # PLACEHOLDER: hardcoded response. No provider is called yet.
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,  # PLACEHOLDER: echoes request; no routing yet
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(
                    content="[PLACEHOLDER] This is a hardcoded response."
                ),
                finish_reason="stop",
            )
        ],
        # PLACEHOLDER: fake usage values, not real token counts.
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
