from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from .rag.config import Settings
from .rag.pipeline import Pipeline
import json
from pathlib import Path

class QueryIn(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)

class LoadIn(BaseModel):
    documents: List[str]

def load_seed() -> list:
    path = Path(__file__).parent / "data" / "seed_documents.json"
    return json.loads(path.read_text(encoding="utf-8"))

settings = Settings()
pipe = Pipeline(settings, seed_docs=load_seed())

app = FastAPI(title="Agentic RAG API", version="1.0.0")

@app.post("/query")
def query(payload: QueryIn):
    try:
        out = pipe.answer(payload.question, top_k=payload.top_k)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer: {e}")

@app.post("/load")
def load(payload: LoadIn):
    try:
        n = pipe.add_documents(payload.documents)
        return {"added": n}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load docs: {e}")
