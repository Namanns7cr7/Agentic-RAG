from pydantic import BaseModel, Field
from typing import Optional

class Settings(BaseModel):
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    generator_model_name: str = Field(default="google/flan-t5-small")
    top_k_default: int = Field(default=3, ge=1, le=20)
    max_new_tokens: int = Field(default=128, ge=16, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    seed: int = 42
    # Future: add auth keys, device, etc.
