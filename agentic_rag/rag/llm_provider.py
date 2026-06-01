"""Configurable LLM provider abstraction.

Supports:
  - Ollama (POST /api/chat)
  - OpenAI-compatible local APIs (POST /chat/completions)
  - Vertex AI Gemini (via google-cloud-aiplatform SDK)
  - Fallback to the existing HuggingFace LocalGenerator

Environment variables:
  LLM_PROVIDER                  = ollama | openai_compatible | local  (default: local)
  LLM_MODEL                     = qwen3.5:9b                          (default)
  OLLAMA_BASE_URL               = http://localhost:11434
  OPENAI_COMPATIBLE_BASE_URL    = http://localhost:8000/v1
  OPENAI_COMPATIBLE_API_KEY     = dummy

  RAG_PROVIDER                  = vertex   (overrides LLM_PROVIDER for Gemini)
  GOOGLE_CLOUD_PROJECT          = <your-gcp-project-id>
  GOOGLE_CLOUD_LOCATION         = us-central1
  VERTEX_GEMINI_MODEL           = gemini-2.5-flash
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .utils_logger import get_logger

logger = get_logger(__name__)

SAFE_LLM_ERROR = "The AI model is temporarily unavailable. Please try again later."

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove ``<think>…</think>`` blocks emitted by some models (e.g. Qwen 3.5)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _is_transient_error(exc: Exception) -> bool:
    """Return True for Google API errors that are worth retrying once."""
    try:
        from google.api_core import exceptions as gexc  # type: ignore
        return isinstance(exc, (gexc.ServiceUnavailable, gexc.InternalServerError, gexc.DeadlineExceeded))
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Common interface every LLM backend must implement."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Send a chat-style request and return the assistant reply as text."""

    # Convenience wrapper for simple single-prompt calls
    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.7) -> str:
        return self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_json(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate JSON output, returning {} if parsing fails."""
        raw = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """Calls Ollama's ``POST /api/chat`` endpoint."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        logger.info("OllamaProvider initialised — model=%s  base_url=%s", model, self.base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
            content = body.get("message", {}).get("content", "")
            return _strip_thinking(content).strip()
        except Exception as exc:
            logger.error("Ollama request failed: %s", exc)
            return SAFE_LLM_ERROR


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (vLLM, LM-Studio, etc.)
# ---------------------------------------------------------------------------

class OpenAICompatibleProvider(LLMProvider):
    """Calls an OpenAI-compatible ``POST /chat/completions`` endpoint."""

    def __init__(self, model: str, base_url: str, api_key: str = "dummy"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        logger.info("OpenAICompatibleProvider — model=%s  base_url=%s", model, self.base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
            content = body["choices"][0]["message"]["content"]
            return _strip_thinking(content).strip()
        except Exception as exc:
            logger.error("OpenAI-compatible request failed: %s", exc)
            return SAFE_LLM_ERROR


# ---------------------------------------------------------------------------
# Local HuggingFace fallback (wraps existing LocalGenerator)
# ---------------------------------------------------------------------------

class LocalHFProvider(LLMProvider):
    """Falls back to the existing ``LocalGenerator`` from ``llm.py``."""

    def __init__(self, model_name: str):
        from .llm import LocalGenerator
        self._gen = LocalGenerator(model_name)
        logger.info("LocalHFProvider — model=%s", model_name)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        # Flatten messages into a single prompt string
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        prompt = "\n".join(parts) + "\nAssistant:"
        return self._gen.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)


# ---------------------------------------------------------------------------
# Vertex AI Gemini provider
# ---------------------------------------------------------------------------

class VertexGeminiProvider(LLMProvider):
    """Calls Vertex AI Gemini via the Vertex AI SDK.

    Authentication:
      - Local dev:  ``gcloud auth application-default login``
      - Production: set GOOGLE_APPLICATION_CREDENTIALS or use a service account
                    attached to the compute instance.

    Environment variables:
      GOOGLE_CLOUD_PROJECT   — GCP project ID (required)
      GOOGLE_CLOUD_LOCATION  — region, default ``us-central1``
      VERTEX_MODEL           — model name, default ``gemini-2.0-flash``
    """

    def __init__(self, model: str, project: str, location: str):
        self.model = model
        self.project = project
        self.location = location
        self._genai_model = None
        self._setup_error = None

        if not project:
            self._setup_error = (
                "VertexGeminiProvider: GOOGLE_CLOUD_PROJECT is not set in .env. "
                "To use Vertex AI, specify GOOGLE_CLOUD_PROJECT and run: "
                "'gcloud auth application-default login'"
            )
            logger.error(self._setup_error)
            return

        try:
            import vertexai  # type: ignore
            from vertexai.generative_models import GenerativeModel  # type: ignore

            vertexai.init(project=project, location=location)
            self._genai_model = GenerativeModel(model)
            logger.info(
                "VertexGeminiProvider initialised — model=%s  project=%s  location=%s",
                model, project, location,
            )
        except ImportError:
            self._setup_error = (
                "google-cloud-aiplatform is not installed. "
                "Run: pip install google-cloud-aiplatform>=1.60.0"
            )
            logger.error(self._setup_error)
        except Exception as exc:
            self._setup_error = f"Failed to initialise Vertex AI Gemini: {exc}"
            logger.error(self._setup_error)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_parts(messages: List[Dict[str, str]]) -> tuple[Optional[str], List[str]]:
        """Split a chat message list into a system instruction + user/model turns.

        The Vertex AI GenerativeModel supports an optional system_instruction
        at model-init time, but we use a simpler flat-prompt approach here so
        we don't need to re-initialise for every system prompt.
        """
        system_parts: List[str] = []
        user_parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)

        combined_system = "\n".join(system_parts) if system_parts else None
        return combined_system, user_parts

    def _generate_with_retry(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate with a single silent retry on transient errors."""
        from vertexai.generative_models import GenerationConfig  # type: ignore

        config = GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        for attempt in range(2):
            try:
                logger.info(
                    "Making real Vertex AI Gemini request (model=%s, attempt=%d, max_tokens=%d, temp=%.2f)...",
                    self.model, attempt + 1, max_tokens, temperature
                )
                response = self._genai_model.generate_content(
                    prompt,
                    generation_config=config,
                )
                text = ""
                # Handle both simple text and multi-candidate responses
                if hasattr(response, "text"):
                    text = response.text or ""
                elif hasattr(response, "candidates") and response.candidates:
                    parts = response.candidates[0].content.parts
                    text = "".join(getattr(p, "text", "") for p in parts)
                return _strip_thinking(text).strip()
            except Exception as exc:
                if attempt == 0 and _is_transient_error(exc):
                    logger.warning("Vertex transient error (attempt 1), retrying: %s", exc)
                    continue
                logger.error("Vertex Gemini request failed: %s", exc)
                return SAFE_LLM_ERROR

        return SAFE_LLM_ERROR

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        if self._setup_error:
            return f"Vertex AI Configuration Error: {self._setup_error}"

        if self._genai_model is None:
            return SAFE_LLM_ERROR

        try:
            system_instruction, user_parts = self._messages_to_parts(messages)

            # Build a single coherent prompt string.
            # Vertex GenerativeModel.generate_content() works best with a plain
            # string prompt when we don't need history tracking.
            prompt_parts: List[str] = []
            if system_instruction:
                prompt_parts.append(f"[System]\n{system_instruction}\n")
            if user_parts:
                prompt_parts.append("\n".join(user_parts))

            prompt = "\n".join(prompt_parts)
            return self._generate_with_retry(prompt, max_tokens, temperature)

        except Exception as exc:
            logger.error("VertexGeminiProvider.chat failed: %s", exc)
            return SAFE_LLM_ERROR


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_provider() -> LLMProvider:
    """Build a provider from environment variables with automatic fallback.

    Priority:
      1. Mock mode  (RAG_MOCK_MODE or TESTING env var)
      2. Vertex AI  (LLM_PROVIDER=vertex or RAG_PROVIDER=vertex)
      3. Ollama     (LLM_PROVIDER=ollama)
      4. OpenAI-compatible  (LLM_PROVIDER=openai_compatible)
      5. Local HuggingFace  (default)
    """

    # --- 1. Mock / test mode ---
    if os.getenv("RAG_MOCK_MODE", "").lower() in ("1", "true", "yes") or os.getenv(
        "TESTING", ""
    ).lower() in ("1", "true", "yes"):
        from .mock_components import MockLLMProvider
        return MockLLMProvider()

    # --- 2. Vertex AI ---
    llm_provider = os.getenv("LLM_PROVIDER", "").lower()
    rag_provider = os.getenv("RAG_PROVIDER", "").lower()
    if llm_provider == "vertex" or rag_provider == "vertex":
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        model = os.getenv("VERTEX_MODEL", os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.0-flash"))
        logger.info("LLM backend: Vertex AI Gemini  model=%s  project=%s", model, project)
        return VertexGeminiProvider(model=model, project=project, location=location)

    # --- 3 & 4. Ollama / OpenAI-compatible ---
    provider_name = os.getenv("LLM_PROVIDER", "local").lower()
    model = os.getenv("LLM_MODEL", "qwen3.5:9b")

    if provider_name == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        prov = OllamaProvider(model=model, base_url=base)
        try:
            test_url = f"{base.rstrip('/')}/api/tags"
            with urllib.request.urlopen(test_url, timeout=5):
                pass
            logger.info("Ollama is reachable — using OllamaProvider")
            return prov
        except Exception:
            logger.warning("Ollama unreachable at %s — falling back to local HF", base)
            return LocalHFProvider(os.getenv("FALLBACK_MODEL", "google/flan-t5-base"))

    if provider_name == "openai_compatible":
        base = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1")
        key = os.getenv("OPENAI_COMPATIBLE_API_KEY", "dummy")
        prov = OpenAICompatibleProvider(model=model, base_url=base, api_key=key)
        try:
            test_url = f"{base.rstrip('/')}/models"
            req = urllib.request.Request(test_url, headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=5):
                pass
            logger.info("OpenAI-compatible endpoint reachable — using OpenAICompatibleProvider")
            return prov
        except Exception:
            logger.warning("OpenAI-compatible endpoint unreachable — falling back to local HF")
            return LocalHFProvider(os.getenv("FALLBACK_MODEL", "google/flan-t5-base"))

    # --- 5. Default: local HuggingFace ---
    return LocalHFProvider(model_name=os.getenv("FALLBACK_MODEL", "google/flan-t5-base"))

