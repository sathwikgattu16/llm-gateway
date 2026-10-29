from abc import ABC, abstractmethod
from dataclasses import dataclass
import os

from fastapi import HTTPException
import httpx

from app.schemas import ChatMessage, Usage


@dataclass
class ProviderResult:
    """What every provider returns, regardless of vendor."""

    content: str
    finish_reason: str
    usage: Usage | None  # None = unknown (never a fake zero)


class ProviderError(HTTPException):
    """Base exception for provider failures, mapped to an HTTP status code."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


class AuthenticationError(ProviderError):
    def __init__(self, detail: str = "Groq authentication failed."):
        super().__init__(status_code=401, detail=detail)


class RateLimitError(ProviderError):
    def __init__(self, detail: str = "Groq rate limit exceeded."):
        super().__init__(status_code=429, detail=detail)


class ModelNotFoundError(ProviderError):
    def __init__(self, detail: str = "Groq model not found."):
        super().__init__(status_code=404, detail=detail)


class ProviderTimeoutError(ProviderError):
    def __init__(self, detail: str = "Groq request timed out."):
        super().__init__(status_code=504, detail=detail)


class UpstreamProviderError(ProviderError):
    def __init__(self, detail: str = "Groq upstream server error."):
        super().__init__(status_code=502, detail=detail)


class Provider(ABC):
    """Common interface. Routes and routing logic only use this."""

    @abstractmethod
    async def generate(
        self,
        model_id: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResult:
        """Return one complete (non-streaming) answer."""


class FakeProvider(Provider):
    """Deterministic, offline provider for development and tests."""

    async def generate(
        self,
        model_id: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResult:
        last = messages[-1].content
        return ProviderResult(
            content=f"[FAKE:{model_id}] Echo: {last}",
            finish_reason="stop",
            usage=None,
        )


class GroqProvider(Provider):
    """Real Groq provider using Groq's OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http_client = http_client

    def __repr__(self) -> str:
        return f"<GroqProvider base_url='{self.base_url}'>"

    def _resolve_api_key(self) -> str:
        raw_key = self.api_key if self.api_key is not None else os.environ.get("GROQ_API_KEY")
        if not raw_key or not raw_key.strip():
            raise AuthenticationError("GROQ_API_KEY environment variable is missing.")
        return raw_key.strip()

    def _redact(self, text: str, key: str | None) -> str:
        if key and key.strip() and key in text:
            return text.replace(key, "[REDACTED]")
        return text

    async def generate(
        self,
        model_id: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResult:
        api_key = self._resolve_api_key()

        payload: dict = {
            "model": model_id,
            "messages": [
                {"role": msg.role, "content": msg.content} for msg in messages
            ],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        try:
            if self._http_client is not None:
                response = await self._http_client.post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            clean_exc = self._redact(str(exc), api_key)
            raise ProviderTimeoutError(f"Request to Groq timed out: {clean_exc}") from exc
        except httpx.RequestError as exc:
            clean_exc = self._redact(str(exc), api_key)
            raise UpstreamProviderError(
                f"Failed to communicate with Groq: {clean_exc}"
            ) from exc

        if response.status_code != 200:
            self._handle_error_response(response, model_id, api_key)

        try:
            data = response.json()
        except Exception as exc:
            raise UpstreamProviderError(
                "Malformed response from Groq: invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise UpstreamProviderError(
                "Malformed response from Groq: expected JSON object."
            )

        return self._parse_response(data)

    def _handle_error_response(
        self, response: httpx.Response, model_id: str, api_key: str | None = None
    ) -> None:
        try:
            error_json = response.json()
            error_obj = error_json.get("error", {})
            if isinstance(error_obj, dict):
                message = error_obj.get("message") or response.text
            else:
                message = str(error_obj)
        except Exception:
            message = response.text

        key_to_redact = api_key or self.api_key or os.environ.get("GROQ_API_KEY")
        if key_to_redact and key_to_redact.strip():
            message = self._redact(message, key_to_redact.strip())

        status = response.status_code
        if status == 401:
            raise AuthenticationError(f"Groq authentication failed: {message}")
        elif status == 404:
            raise ModelNotFoundError(
                f"Groq model '{model_id}' was not found: {message}"
            )
        elif status == 429:
            raise RateLimitError(f"Groq rate limit exceeded: {message}")
        elif status >= 500:
            raise UpstreamProviderError(
                f"Groq upstream error ({status}): {message}"
            )
        else:
            raise ProviderError(
                status_code=status, detail=f"Groq API error ({status}): {message}"
            )

    def _parse_response(self, data: dict) -> ProviderResult:
        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise UpstreamProviderError(
                "Malformed response from Groq: missing choices."
            )

        choice = choices[0]
        if not isinstance(choice, dict):
            raise UpstreamProviderError(
                "Malformed response from Groq: malformed choice object."
            )

        message = choice.get("message")
        if not isinstance(message, dict):
            raise UpstreamProviderError(
                "Malformed response from Groq: malformed message object."
            )

        raw_content = message.get("content")
        content = "" if raw_content is None else str(raw_content)

        raw_finish_reason = choice.get("finish_reason")
        finish_reason = "stop" if not raw_finish_reason else str(raw_finish_reason)

        usage = None
        usage_data = data.get("usage")
        if isinstance(usage_data, dict):
            p = usage_data.get("prompt_tokens")
            c = usage_data.get("completion_tokens")
            t = usage_data.get("total_tokens")
            if p is not None or c is not None or t is not None:
                usage = Usage(
                    prompt_tokens=p if isinstance(p, int) else None,
                    completion_tokens=c if isinstance(c, int) else None,
                    total_tokens=t if isinstance(t, int) else None,
                )

        return ProviderResult(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
        )


class GeminiProvider(Provider):
    """Real Gemini provider using Google's native REST API (generateContent)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http_client = http_client

    def __repr__(self) -> str:
        return f"<GeminiProvider base_url='{self.base_url}'>"

    def _resolve_api_key(self) -> str:
        raw_key = (
            self.api_key
            if self.api_key is not None
            else os.environ.get("GEMINI_API_KEY")
        )
        if not raw_key or not raw_key.strip():
            raise AuthenticationError(
                "GEMINI_API_KEY environment variable is missing."
            )
        return raw_key.strip()

    def _redact(self, text: str, key: str | None) -> str:
        if key and key.strip() and key in text:
            return text.replace(key, "[REDACTED]")
        return text

    def _build_payload(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        contents = []
        system_parts = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append({"text": msg.content})
            elif msg.role == "user":
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"text": msg.content}],
                    }
                )
            elif msg.role == "assistant":
                contents.append(
                    {
                        "role": "model",
                        "parts": [{"text": msg.content}],
                    }
                )

        payload: dict = {}
        if not contents and system_parts:
            # Fallback if only a system prompt was provided
            contents.append({"role": "user", "parts": [{"text": ""}]})

        payload["contents"] = contents

        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        generation_config: dict = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        if generation_config:
            payload["generationConfig"] = generation_config

        return payload

    async def generate(
        self,
        model_id: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResult:
        api_key = self._resolve_api_key()

        payload = self._build_payload(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

        clean_model = (
            model_id if model_id.startswith("models/") else f"models/{model_id}"
        )
        url = f"{self.base_url}/{clean_model}:generateContent"

        try:
            if self._http_client is not None:
                response = await self._http_client.post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        url, json=payload, headers=headers
                    )
        except httpx.TimeoutException as exc:
            clean_exc = self._redact(str(exc), api_key)
            raise ProviderTimeoutError(
                f"Request to Gemini timed out: {clean_exc}"
            ) from exc
        except httpx.RequestError as exc:
            clean_exc = self._redact(str(exc), api_key)
            raise UpstreamProviderError(
                f"Failed to communicate with Gemini: {clean_exc}"
            ) from exc

        if response.status_code != 200:
            self._handle_error_response(response, model_id, api_key)

        try:
            data = response.json()
        except Exception as exc:
            raise UpstreamProviderError(
                "Malformed response from Gemini: invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise UpstreamProviderError(
                "Malformed response from Gemini: expected JSON object."
            )

        return self._parse_response(data)

    def _handle_error_response(
        self,
        response: httpx.Response,
        model_id: str,
        api_key: str | None = None,
    ) -> None:
        try:
            error_json = response.json()
            error_obj = error_json.get("error", {})
            if isinstance(error_obj, dict):
                message = error_obj.get("message") or response.text
            else:
                message = str(error_obj)
        except Exception:
            message = response.text

        key_to_redact = (
            api_key or self.api_key or os.environ.get("GEMINI_API_KEY")
        )
        if key_to_redact and key_to_redact.strip():
            message = self._redact(message, key_to_redact.strip())

        status = response.status_code
        if status in (401, 403):
            raise AuthenticationError(
                f"Gemini authentication failed: {message}"
            )
        elif status == 404 or (status == 400 and "not found" in message.lower()):
            raise ModelNotFoundError(
                f"Gemini model '{model_id}' was not found: {message}"
            )
        elif status == 429:
            raise RateLimitError(f"Gemini rate limit exceeded: {message}")
        elif status >= 500:
            raise UpstreamProviderError(
                f"Gemini upstream error ({status}): {message}"
            )
        else:
            raise ProviderError(
                status_code=status,
                detail=f"Gemini API error ({status}): {message}",
            )

    def _parse_response(self, data: dict) -> ProviderResult:
        candidates = data.get("candidates")
        if not candidates or not isinstance(candidates, list):
            prompt_feedback = data.get("promptFeedback")
            if isinstance(prompt_feedback, dict):
                block_reason = prompt_feedback.get("blockReason")
                if block_reason:
                    raise UpstreamProviderError(
                        f"Gemini blocked prompt: {block_reason}."
                    )
            raise UpstreamProviderError(
                "Malformed response from Gemini: missing candidates."
            )

        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise UpstreamProviderError(
                "Malformed response from Gemini: malformed candidate object."
            )

        content_obj = candidate.get("content")
        content = ""
        if isinstance(content_obj, dict):
            parts = content_obj.get("parts")
            if isinstance(parts, list):
                content = "".join(
                    str(part.get("text", ""))
                    for part in parts
                    if isinstance(part, dict) and "text" in part
                )

        raw_finish_reason = candidate.get("finishReason")
        if raw_finish_reason == "STOP":
            finish_reason = "stop"
        elif raw_finish_reason == "MAX_TOKENS":
            finish_reason = "length"
        elif raw_finish_reason in (
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
        ):
            finish_reason = "content_filter"
        elif raw_finish_reason:
            finish_reason = str(raw_finish_reason).lower()
        else:
            finish_reason = "stop"

        usage = None
        usage_data = data.get("usageMetadata")
        if isinstance(usage_data, dict):
            p = usage_data.get("promptTokenCount")
            c = usage_data.get("candidatesTokenCount")
            t = usage_data.get("totalTokenCount")
            if p is not None or c is not None or t is not None:
                usage = Usage(
                    prompt_tokens=p if isinstance(p, int) else None,
                    completion_tokens=c if isinstance(c, int) else None,
                    total_tokens=t if isinstance(t, int) else None,
                )

        return ProviderResult(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
        )


# Milestone 4: Both Gemini and Groq are real providers.
# FakeProvider remains available for deterministic offline testing.
_PROVIDERS: dict[str, Provider] = {
    "gemini": GeminiProvider(),
    "groq": GroqProvider(),
}


def get_provider(name: str) -> Provider:
    return _PROVIDERS[name]

