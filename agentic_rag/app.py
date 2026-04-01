"""FastAPI application — Agentic RAG REST API with frontend UI."""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from .rag.config import Settings
from .rag.pipeline import Pipeline
import json
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"


# ── Request / Response Models ─────────────────────────────────────

class QueryIn(BaseModel):
    question: str = Field(..., min_length=2, description="The question to ask")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of documents to retrieve")


class QueryOut(BaseModel):
    plan: str
    draft: str
    final: str
    observation: Optional[str] = None


class LoadIn(BaseModel):
    documents: List[str] = Field(..., min_length=1, description="Documents to add to the knowledge base")


class LoadOut(BaseModel):
    added: int
    total: int


class HealthOut(BaseModel):
    status: str
    documents_loaded: int
    model: str


# ── App Setup ─────────────────────────────────────────────────────

def load_seed() -> list:
    path = Path(__file__).parent / "data" / "seed_documents.json"
    return json.loads(path.read_text(encoding="utf-8"))


settings = Settings()
pipe = Pipeline(settings, seed_docs=load_seed())

app = FastAPI(
    title="Agentic RAG API",
    description="An agentic Retrieval-Augmented Generation system with LLM-driven planning, "
                "tool use, conversation memory, and self-reflection.",
    version="2.0.0",
)

# CORS — allow the frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/", tags=["info"], include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthOut, tags=["info"])
def health():
    return HealthOut(
        status="ok",
        documents_loaded=pipe.doc_count,
        model=settings.generator_model_name,
    )


@app.post("/query", response_model=QueryOut, tags=["rag"])
def query(payload: QueryIn):
    try:
        out = pipe.answer(payload.question, top_k=payload.top_k)
        return QueryOut(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer: {e}")


@app.post("/load", response_model=LoadOut, tags=["rag"])
def load(payload: LoadIn):
    try:
        n = pipe.add_documents(payload.documents)
        return LoadOut(added=n, total=pipe.doc_count)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load docs: {e}")


@app.post("/memory/clear", tags=["memory"])
def clear_memory():
    pipe.clear_memory()
    return {"status": "memory cleared"}
