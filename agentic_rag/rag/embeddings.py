from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .utils_logger import get_logger

logger = get_logger(__name__)

# Task-type constants for Vertex text-embedding-004
_VERTEX_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
_VERTEX_TASK_QUERY = "RETRIEVAL_QUERY"


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


# ---------------------------------------------------------------------------
# Local (sentence-transformers)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Mock (for tests / CI — no model downloads)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Vertex AI text-embedding-004
# ---------------------------------------------------------------------------

class VertexEmbeddingProvider(EmbeddingProvider):
    """Google Vertex AI text embeddings (text-embedding-004 or similar).

    Authentication:
      - Local dev:  ``gcloud auth application-default login``
      - Production: GOOGLE_APPLICATION_CREDENTIALS or attached service account.

    Environment variables:
      GOOGLE_CLOUD_PROJECT       — GCP project ID (required)
      GOOGLE_CLOUD_LOCATION      — region, default ``us-central1``
      VERTEX_EMBEDDING_MODEL     — model name, default ``text-embedding-004``

    Notes:
      - Indexing uses task_type=RETRIEVAL_DOCUMENT.
      - Query-time embedding uses task_type=RETRIEVAL_QUERY.
      - A 1-token probe embedding is performed on init to determine the real
        vector dimension so the FAISS index is correctly sized from the start.
    """

    def __init__(self, model_name: str, project: str, location: str) -> None:
        self._model_name = model_name
        self._project = project
        self._location = location
        self._model = None
        self._dim: Optional[int] = None

        if not project:
            logger.warning(
                "VertexEmbeddingProvider: GOOGLE_CLOUD_PROJECT is not set. "
                "Will return zero vectors."
            )
            return

        try:
            import vertexai  # type: ignore
            from vertexai.language_models import TextEmbeddingModel  # type: ignore

            vertexai.init(project=project, location=location)
            self._model = TextEmbeddingModel.from_pretrained(model_name)
            logger.info(
                "VertexEmbeddingProvider initialised — model=%s  project=%s  location=%s",
                model_name, project, location,
            )
            # Probe a single token to fix the real dimension at startup.
            self._probe_dim()
        except ImportError:
            logger.error(
                "google-cloud-aiplatform is not installed. "
                "Run: pip install 'google-cloud-aiplatform>=1.60.0'"
            )
        except Exception as exc:
            logger.error("Failed to initialise VertexEmbeddingProvider: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_dim(self) -> None:
        """Embed a single word to discover the real output dimension."""
        try:
            from vertexai.language_models import TextEmbeddingInput  # type: ignore
            sample = TextEmbeddingInput("hello", task_type=_VERTEX_TASK_DOCUMENT)
            result = self._model.get_embeddings([sample])
            if result:
                self._dim = len(result[0].values)
                logger.info("VertexEmbeddingProvider dim probed: %d", self._dim)
        except Exception as exc:
            logger.warning("Vertex dim probe failed, using default 768: %s", exc)
            self._dim = 768

    def _get_embeddings_with_task(
        self, texts: List[str], task_type: str
    ) -> List[List[float]]:
        """Call Vertex API with a specific task_type for better quality."""
        if self._model is None:
            return [[0.0] * (self._dim or 768) for _ in texts]
        try:
            logger.info(
                "Making real Vertex AI Embedding request (model=%s, texts_count=%d, task_type=%s)...",
                self._model_name, len(texts), task_type
            )
            from vertexai.language_models import TextEmbeddingInput  # type: ignore
            inputs = [TextEmbeddingInput(t, task_type=task_type) for t in texts]
            results = self._model.get_embeddings(inputs)
            vectors = [r.values for r in results]
            if vectors and self._dim is None:
                self._dim = len(vectors[0])
            return vectors
        except Exception as exc:
            logger.error("Vertex embedding call failed: %s", exc)
            return [[0.0] * (self._dim or 768) for _ in texts]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> List[float]:
        """Embed a single query string (RETRIEVAL_QUERY task type)."""
        return self._get_embeddings_with_task([text], _VERTEX_TASK_QUERY)[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document chunks (RETRIEVAL_DOCUMENT task type)."""
        return self._get_embeddings_with_task(texts, _VERTEX_TASK_DOCUMENT)

    @property
    def dim(self) -> int:
        return self._dim if self._dim is not None else 768


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedding_provider(model_name: str) -> EmbeddingProvider:
    """Select an embedding provider from environment variables with safe fallback.

    Priority:
      1. Mock mode          (RAG_MOCK_MODE or TESTING)
      2. EMBEDDING_PROVIDER=vertex   (explicit Vertex embeddings)
      3. RAG_PROVIDER=vertex         (legacy: use Vertex when RAG_PROVIDER is vertex)
      4. Local sentence-transformers (default / fallback if Vertex fails)
    """
    # 1. Mock / test mode
    if os.getenv("RAG_MOCK_MODE", "").lower() in ("1", "true", "yes") or os.getenv(
        "TESTING", ""
    ).lower() in ("1", "true", "yes"):
        return MockEmbeddingProvider()

    project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    vertex_model = os.getenv("VERTEX_EMBEDDING_MODEL", "text-embedding-004")

    # 2. Explicit embedding provider override
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
    if embedding_provider == "vertex":
        logger.info("Embedding backend: Vertex AI  model=%s  project=%s", vertex_model, project)
        try:
            prov = VertexEmbeddingProvider(vertex_model, project, location)
            if getattr(prov, "_model", None) is not None:
                return prov
            logger.warning("Vertex AI embeddings failed to initialize. Falling back to local.")
        except Exception as exc:
            logger.warning("Vertex AI embeddings failed to initialize: %s. Falling back to local.", exc)

    # 3. Legacy: if RAG_PROVIDER=vertex, also use Vertex embeddings
    rag_provider = os.getenv("RAG_PROVIDER", "").lower()
    if rag_provider == "vertex":
        logger.info(
            "Embedding backend: Vertex AI (via RAG_PROVIDER)  model=%s  project=%s",
            vertex_model, project,
        )
        try:
            prov = VertexEmbeddingProvider(vertex_model, project, location)
            if getattr(prov, "_model", None) is not None:
                return prov
            logger.warning("Vertex AI embeddings failed to initialize. Falling back to local.")
        except Exception as exc:
            logger.warning("Vertex AI embeddings failed to initialize: %s. Falling back to local.", exc)

    # 4. Default: local sentence-transformers
    logger.info("Embedding backend: local  model=%s", model_name)
    return LocalEmbeddingProvider(model_name)



# ---------------------------------------------------------------------------
# High-level Embeddings wrapper (used by Pipeline / Retriever)
# ---------------------------------------------------------------------------

class Embeddings:
    def __init__(self, model_name: str, provider: Optional[EmbeddingProvider] = None) -> None:
        self.provider = provider or get_embedding_provider(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.provider.embed_documents(texts)
        arr = np.array(vectors, dtype="float32")
        if len(arr.shape) == 1:
            arr = np.expand_dims(arr, axis=0)
        return arr

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query using the query task type (Vertex-aware)."""
        vec = self.provider.embed_text(text)
        return np.array([vec], dtype="float32")

    @property
    def dim(self) -> int:
        return self.provider.dim
