from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
import os
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True).astype("float32")[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        arr = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        if len(arr.shape) == 1:
            arr = np.expand_dims(arr, axis=0)
        return arr.tolist()

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dim: int = 8):
        self._dim = dim

    def embed_text(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            vec = np.zeros(self._dim, dtype="float32")
            for i, ch in enumerate(text.encode("utf-8")):
                vec[i % self._dim] += float(ch)
            norm = np.linalg.norm(vec) or 1.0
            vectors.append((vec / norm).tolist())
        return vectors

    @property
    def dim(self) -> int:
        return self._dim


class VertexEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str, project: str, location: str) -> None:
        self._model_name = model_name
        self._project = project
        self._location = location
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel

            vertexai.init(project=project, location=location)
            self._model = TextEmbeddingModel.from_pretrained(model_name)
        except Exception:
            self._model = None
        self._dim: Optional[int] = None

    def embed_text(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            return [[0.0] * (self._dim or 8) for _ in texts]
        embeddings = self._model.get_embeddings(texts)
        vectors = [e.values for e in embeddings]
        if vectors and self._dim is None:
            self._dim = len(vectors[0])
        return vectors

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Default dimension until a real embedding call occurs
            return 768
        return self._dim

def get_embedding_provider(model_name: str) -> EmbeddingProvider:
    if os.getenv("RAG_MOCK_MODE", "").lower() in ("1", "true", "yes") or os.getenv("TESTING", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return MockEmbeddingProvider()

    rag_provider = os.getenv("RAG_PROVIDER", "").lower()
    if rag_provider == "vertex":
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        model = os.getenv("VERTEX_EMBEDDING_MODEL", "text-embedding-004")
        return VertexEmbeddingProvider(model, project, location)

    return LocalEmbeddingProvider(model_name)


class Embeddings:
    def __init__(self, model_name: str, provider: Optional[EmbeddingProvider] = None) -> None:
        self.provider = provider or get_embedding_provider(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.provider.embed_documents(texts)
        arr = np.array(vectors, dtype="float32")
        if len(arr.shape) == 1:
            arr = np.expand_dims(arr, axis=0)
        return arr

    @property
    def dim(self) -> int:
        return self.provider.dim
