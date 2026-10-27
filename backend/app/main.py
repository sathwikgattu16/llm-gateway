from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load environment variables from project root .env if present
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from app.routes import router

app = FastAPI(title="LLM Gateway")

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}

