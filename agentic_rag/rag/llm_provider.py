"""Configurable LLM provider abstraction.

Supports:
  - Ollama (POST /api/chat)
  - OpenAI-compatible local APIs (POST /chat/completions)
  - Fallback to the existing HuggingFace LocalGenerator

Environment variables:
  LLM_PROVIDER           = ollama | openai_compatible | local   (default: local)
  LLM_MODEL              = qwen3.5:9b                          (default)
  OLLAMA_BASE_URL         = http://localhost:11434
  OPENAI_COMPATIBLE_BASE_URL = http://localhost:8000/v1
  OPENAI_COMPATIBLE_API_KEY  = dummy
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .utils_logger import get_logger

logger = get_logger(__name__)

SAFE_LLM_ERROR = "The AI model is temporarily unavailable. Please try again later."

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
            # Strip <think>…</think> blocks that Qwen 3.5 emits
            content = _strip_thinking(content)
            return content.strip()
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
            content = _strip_thinking(content)
            return content.strip()
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
    """Calls Vertex AI Gemini via the Vertex AI SDK."""

    def __init__(self, model: str, project: str, location: str):
        self.model = model
        self.project = project
        self.location = location
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(project=project, location=location)
            self._model = GenerativeModel(model)
        except Exception as exc:
            logger.error("Failed to initialize Vertex AI: %s", exc)
            self._model = None

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        if self._model is None:
            return SAFE_LLM_ERROR
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        try:
            response = self._model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            text = getattr(response, "text", "") or ""
            return text.strip()
        except Exception as exc:
            logger.error("Vertex Gemini request failed: %s", exc)
            return SAFE_LLM_ERROR


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove ``<think>…</think>`` blocks emitted by Qwen 3.5."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_llm_provider() -> LLMProvider:
    """Build a provider from environment variables with automatic fallback."""

    if os.getenv("RAG_MOCK_MODE", "").lower() in ("1", "true", "yes") or os.getenv("TESTING", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        from .mock_components import MockLLMProvider
        return MockLLMProvider()

    rag_provider = os.getenv("RAG_PROVIDER", "").lower()
    if rag_provider == "vertex":
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        model = os.getenv("VERTEX_GEMINI_MODEL", "gemini-1.5-flash")
        return VertexGeminiProvider(model=model, project=project, location=location)

    provider_name = os.getenv("LLM_PROVIDER", "local").lower()
    model = os.getenv("LLM_MODEL", "qwen3.5:9b")

    if provider_name == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        prov = OllamaProvider(model=model, base_url=base)
        # Quick connectivity check
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
        # Quick connectivity check
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

    # Default: local HuggingFace
    return LocalHFProvider(model_name=os.getenv("FALLBACK_MODEL", "google/flan-t5-base"))
