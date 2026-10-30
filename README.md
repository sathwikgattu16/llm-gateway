# LLM Gateway

A modular LLM gateway built with Python and FastAPI that provides a unified API for interacting with multiple LLM providers. The project focuses on provider abstraction, normalized responses, reliable error handling, and eventually intelligent routing and mid-stream recovery.

## Current Status

**Implemented**
- FastAPI backend with an OpenAI-compatible chat completions endpoint (`/v1/chat/completions`).
- Model registry backed by YAML configuration (`backend/configs/models.yaml`).
- Shared asynchronous provider interface (`Provider` ABC returning `ProviderResult` and async generator `ProviderChunk`).
- Groq provider adapter supporting non-streaming and native SSE streaming.
- Gemini provider adapter using Google's native `generateContent` and `streamGenerateContent` API with SSE streaming.
- Normalized completion responses, streaming chunks, and token usage across providers.
- Provider error handling for authentication failures, rate limits, timeouts, network errors, and upstream failures.
- API-key protection and secret redaction in exception messages and representations.
- Automated tests for provider adapters, API routes, model registry, and error handling (137 tests passing).

**Planned**
- Intelligent rule-based model routing.
- Retries, timeouts, cancellation, and circuit breakers.
- Provider fallback and mid-stream response continuation.
- Recovery event tracking and response-splicing strategies.
- Request logging, metrics, and observability.
- Gateway monitoring dashboard.
- Demo chat application.
- Routing and recovery experiments.
- ML-based routing.

## Architecture

```text
Client Application
       |
       v
FastAPI Gateway
       |
       v
Model Registry
       |
       v
Provider Interface
       |
       +------> Groq Adapter ------> Groq API (REST & SSE Stream)
       |
       +------> Gemini Adapter ----> Gemini API (REST & streamGenerateContent SSE)
       |
       v
Normalized Completion Response / SSE Event Stream
```

The gateway exposes a common interface to clients while provider adapters translate requests and responses into a shared internal format. This separation allows routing and recovery logic to evolve independently of individual provider APIs.

## Tech Stack

- Python
- FastAPI
- Pydantic
- HTTPX
- PyYAML
- python-dotenv
- Uvicorn
- pytest

## Project Structure

```text
llm-gateway/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── providers.py
│   │   ├── registry.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── configs/
│   │   └── models.yaml
│   ├── tests/
│   │   ├── test_chat_completions.py
│   │   ├── test_gemini_provider.py
│   │   ├── test_groq_provider.py
│   │   ├── test_models_endpoint.py
│   │   ├── test_providers.py
│   │   └── test_registry.py
│   ├── pytest.ini
│   └── requirements.txt
├── demo-app/                   # Placeholder (planned chat demo application)
├── docs/                       # Placeholder (planned architecture documentation)
├── experiments/                # Placeholder (planned routing and recovery evaluations)
├── frontend/                   # Placeholder (planned monitoring dashboard)
├── scripts/                    # Provider investigation spikes and reports
│   ├── provider_investigation_report.md
│   ├── spike_gemini.py
│   └── spike_groq.py
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python installed (Python 3.10+ recommended)
- Valid API keys for the providers you want to test (Groq and/or Google Gemini)

### Setup

From the project root, activate the existing virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies if needed:

```powershell
pip install -r backend\requirements.txt
```

Configure your provider keys in the root `.env` file (create it if not present):

```dotenv
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Never commit `.env` or expose API keys in source code.

### Run the Backend

From the project root:

```powershell
uvicorn app.main:app --app-dir backend --reload --port 8000
```

- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Available models: http://localhost:8000/v1/models

## Example API Request

Endpoint: `POST /v1/chat/completions`

### Non-Streaming Request

```json
{
  "model": "gemini-3.5-flash-lite",
  "messages": [
    {
      "role": "user",
      "content": "Explain what an LLM gateway does in two sentences."
    }
  ],
  "temperature": 0.2,
  "max_tokens": 150,
  "stream": false
}
```

### Streaming Request (`stream: true`)

```json
{
  "model": "gemini-3.5-flash-lite",
  "messages": [
    {
      "role": "user",
      "content": "Explain what an LLM gateway does in two sentences."
    }
  ],
  "temperature": 0.2,
  "max_tokens": 150,
  "stream": true
}
```

Streaming responses emit Server-Sent Events (`text/event-stream`) with standard OpenAI-compatible chunks:

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1729864200,"model":"gemini-3.5-flash-lite","choices":[{"index":0,"delta":{"role":"assistant","content":"An LLM"},"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1729864200,"model":"gemini-3.5-flash-lite","choices":[{"index":0,"delta":{"content":" gateway acts as"},"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1729864200,"model":"gemini-3.5-flash-lite","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":25,"total_tokens":40}}

data: [DONE]
```

Model IDs must match the configured model registry (`backend/configs/models.yaml`) and be available through the relevant provider.

## Testing

Run the full backend test suite from the project root:

```powershell
python -m pytest backend -v
```

The test suite completes with **137 passed, 1 warning** (Starlette test client deprecation).

Provider adapter tests use mocked HTTP responses (`httpx.MockTransport`) to verify request construction, response normalization, and error handling without depending on live provider calls. Live API requests should be tested separately.

## Roadmap

1. Provider abstraction and model registry — completed.
2. Groq adapter — implemented and tested.
3. Native Gemini adapter — implemented and tested.
4. Normalized SSE streaming — completed.
5. Authentication and rate limiting — planned.
6. Rule-based routing — planned.
7. Retries, timeouts, cancellation, and circuit breakers — planned.
8. Fallback and mid-stream recovery — planned.
9. Recovery events and response-splicing strategies — planned.
10. Request logging and observability — planned.
11. Monitoring dashboard and demo chat application — planned.
12. Routing and recovery experiments — planned.
13. ML-based routing — planned.
14. Hardening, documentation, and deployment preparation — planned.

## Design Principles

- Keep provider-specific logic inside adapters.
- Maintain a consistent internal provider contract.
- Keep credentials outside source code.
- Test failure cases as well as successful requests.
- Separate implemented capabilities from planned work.
- Design experiments to measure gateway behaviour independently of provider-specific differences.
