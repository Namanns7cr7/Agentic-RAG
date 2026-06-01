import os
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Central configuration for the Agentic RAG pipeline."""

    embedding_model_name: str = Field(default_factory=lambda: os.getenv("VERTEX_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    generator_model_name: str = Field(default_factory=lambda: os.getenv("VERTEX_MODEL", os.getenv("VERTEX_GEMINI_MODEL", "google/flan-t5-base")))
    top_k_default: int = Field(default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHUNKS", "5")), ge=1, le=20)
    max_new_tokens: int = Field(default_factory=lambda: int(os.getenv("MAX_OUTPUT_TOKENS", "700")), ge=16, le=2048)
    temperature: float = Field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.2")), ge=0.0, le=1.5)
    seed: int = 42
    max_memory_turns: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Max Q&A pairs to keep in conversation memory",
    )
    google_cloud_project: str = Field(default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    google_cloud_location: str = Field(default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", os.getenv("RAG_PROVIDER", "local")))
    max_chars_per_chunk: int = Field(default_factory=lambda: int(os.getenv("MAX_CHARS_PER_CHUNK", "2500")))

