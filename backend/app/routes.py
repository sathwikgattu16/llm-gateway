import json
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.providers import ProviderChunk, get_provider
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
    model = get_registry().resolve(request.model)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' is not configured.",
        )

    provider = get_provider(model.provider)

    if not request.stream:
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

    # Streaming mode
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())

    stream = provider.generate_stream(
        model_id=model.id,
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    # Pre-fetch first chunk so any initial connection/auth/model error raises before 200 headers are sent
    try:
        first_chunk = await anext(stream, None)
    except Exception:
        await stream.aclose()
        raise

    async def sse_generator():
        try:
            first = True
            emitted_finish = False
            last_usage = None

            def format_event(delta_dict: dict, finish_reason: str | None, usage_obj: dict | None) -> str:
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model.id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta_dict,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": usage_obj,
                }
                return f"data: {json.dumps(payload)}\n\n"

            def handle_chunk(chunk: ProviderChunk) -> list[str]:
                nonlocal first, emitted_finish, last_usage
                events = []
                if chunk.usage is not None:
                    last_usage = chunk.usage

                if chunk.delta:
                    delta_dict = (
                        {"role": "assistant", "content": chunk.delta}
                        if first
                        else {"content": chunk.delta}
                    )
                    first = False
                    events.append(format_event(delta_dict, None, None))

                if chunk.finish_reason is not None:
                    usage_dict = last_usage.model_dump() if last_usage else None
                    events.append(format_event({}, chunk.finish_reason, usage_dict))
                    emitted_finish = True

                return events

            if first_chunk is not None:
                for event in handle_chunk(first_chunk):
                    yield event

            async for chunk in stream:
                for event in handle_chunk(chunk):
                    yield event

            if not emitted_finish:
                usage_dict = last_usage.model_dump() if last_usage else None
                yield format_event({}, "stop", usage_dict)

            yield "data: [DONE]\n\n"
        finally:
            await stream.aclose()

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
