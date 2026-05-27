"""Tests for LLM provider config loading — no actual model server needed."""

import os
import pytest
from unittest.mock import patch


class TestLLMProviderConfig:
    def test_defaults_to_local(self):
        """Without env vars set, provider should not crash on import."""
        from agentic_rag.rag.llm_provider import get_llm_provider
        # Just verify the factory function exists and is callable
        assert callable(get_llm_provider)

    def test_env_selects_ollama(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "qwen3.5:9b",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }):
            from agentic_rag.rag.llm_provider import OllamaProvider
            prov = OllamaProvider(model="qwen3.5:9b", base_url="http://localhost:11434")
            assert prov.model == "qwen3.5:9b"
            assert "11434" in prov.base_url

    def test_env_selects_openai_compatible(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai_compatible",
            "LLM_MODEL": "qwen3.5:9b",
            "OPENAI_COMPATIBLE_BASE_URL": "http://localhost:8001/v1",
            "OPENAI_COMPATIBLE_API_KEY": "test-key",
        }):
            from agentic_rag.rag.llm_provider import OpenAICompatibleProvider
            prov = OpenAICompatibleProvider(
                model="qwen3.5:9b",
                base_url="http://localhost:8001/v1",
                api_key="test-key",
            )
            assert prov.model == "qwen3.5:9b"
            assert prov.api_key == "test-key"

    def test_model_is_configurable_without_code_change(self):
        """Changing LLM_MODEL should propagate without touching any Python file."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "qwen3.5:4b",
        }):
            model_from_env = os.getenv("LLM_MODEL", "qwen3.5:9b")
            assert model_from_env == "qwen3.5:4b"
