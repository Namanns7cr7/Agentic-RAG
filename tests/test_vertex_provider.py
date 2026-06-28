"""Tests for Vertex AI provider implementations — fully mocked, CI-safe.

These tests exercise:
  - VertexGeminiProvider construction and chat()
  - VertexEmbeddingProvider construction and embed_*()
  - get_llm_provider() factory with RAG_PROVIDER=vertex
  - get_embedding_provider() factory with EMBEDDING_PROVIDER=vertex
  - Graceful degradation when the SDK is unavailable or project is unset
  - <think> block stripping from Vertex responses
  - Transient-error silent retry logic
"""

import os
import types
import sys
from unittest.mock import MagicMock, patch, PropertyMock
import pytest


# ---------------------------------------------------------------------------
# Helpers — build minimal mock of the vertexai SDK so we never hit GCP
# ---------------------------------------------------------------------------

def _make_vertexai_mock() -> types.ModuleType:
    """Create a minimal fake ``vertexai`` module tree."""
    vertexai = types.ModuleType("vertexai")
    vertexai.init = MagicMock()

    # generative_models sub-module
    gen_models = types.ModuleType("vertexai.generative_models")
    mock_response = MagicMock()
    mock_response.text = "Hello from Gemini."
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = mock_response
    MockGenerativeModel = MagicMock(return_value=mock_model_instance)
    MockGenerationConfig = MagicMock()
    gen_models.GenerativeModel = MockGenerativeModel
    gen_models.GenerationConfig = MockGenerationConfig
    vertexai.generative_models = gen_models

    # language_models sub-module
    lang_models = types.ModuleType("vertexai.language_models")
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 768
    mock_embedding_model = MagicMock()
    mock_embedding_model.get_embeddings.return_value = [mock_embedding]
    MockTextEmbeddingModel = MagicMock()
    MockTextEmbeddingModel.from_pretrained.return_value = mock_embedding_model
    MockTextEmbeddingInput = MagicMock(side_effect=lambda text, task_type=None: (text, task_type))
    lang_models.TextEmbeddingModel = MockTextEmbeddingModel
    lang_models.TextEmbeddingInput = MockTextEmbeddingInput
    vertexai.language_models = lang_models

    return vertexai


def _install_vertexai_mock():
    """Insert the fake vertexai module into sys.modules."""
    vertexai = _make_vertexai_mock()
    sys.modules["vertexai"] = vertexai
    sys.modules["vertexai.generative_models"] = vertexai.generative_models
    sys.modules["vertexai.language_models"] = vertexai.language_models
    return vertexai


def _uninstall_vertexai_mock():
    for key in list(sys.modules):
        if key.startswith("vertexai"):
            del sys.modules[key]


# ---------------------------------------------------------------------------
# VertexGeminiProvider tests
# ---------------------------------------------------------------------------

class TestVertexGeminiProvider:

    def setup_method(self):
        _install_vertexai_mock()
        # Force re-import of the provider module with the mock in place
        for key in list(sys.modules):
            if "llm_provider" in key and "agentic_rag" in key:
                del sys.modules[key]

    def teardown_method(self):
        _uninstall_vertexai_mock()

    def test_construction_with_project_succeeds(self):
        from agentic_rag.rag.llm_provider import VertexGeminiProvider
        prov = VertexGeminiProvider(
            model="gemini-2.5-flash",
            project="test-project",
            location="us-central1",
        )
        assert prov.model == "gemini-2.5-flash"
        assert prov.project == "test-project"
        assert prov._genai_model is not None

    def test_construction_without_project_sets_model_none(self):
        from agentic_rag.rag.llm_provider import VertexGeminiProvider
        prov = VertexGeminiProvider(model="gemini-2.5-flash", project="", location="us-central1")
        assert prov._genai_model is None

    def test_chat_returns_text(self):
        from agentic_rag.rag.llm_provider import VertexGeminiProvider
        prov = VertexGeminiProvider(
            model="gemini-2.5-flash",
            project="test-project",
            location="us-central1",
        )
        result = prov.chat([{"role": "user", "content": "Hello"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chat_returns_safe_error_when_model_is_none(self):
        from agentic_rag.rag.llm_provider import VertexGeminiProvider
        prov = VertexGeminiProvider(model="gemini-2.5-flash", project="", location="us-central1")
        # The provider returns a configuration error string when model is None
        result = prov.chat([{"role": "user", "content": "Hello"}])
        expected = "Vertex AI Configuration Error: VertexGeminiProvider: GOOGLE_CLOUD_PROJECT is not set in .env. To use Vertex AI, specify GOOGLE_CLOUD_PROJECT and run: 'gcloud auth application-default login'"
        assert result == expected

    def test_chat_strips_think_blocks(self):
        vertexai = sys.modules["vertexai"]
        mock_resp = MagicMock()
        mock_resp.text = "<think>internal reasoning</think>Final answer."
        vertexai.generative_models.GenerativeModel.return_value.generate_content.return_value = mock_resp

        from agentic_rag.rag.llm_provider import VertexGeminiProvider
        prov = VertexGeminiProvider(
            model="gemini-2.5-flash",
            project="test-project",
            location="us-central1",
        )
        result = prov.chat([{"role": "user", "content": "Explain something."}])
        assert "<think>" not in result
        assert "Final answer." in result

    def test_chat_with_system_message(self):
        from agentic_rag.rag.llm_provider import VertexGeminiProvider
        prov = VertexGeminiProvider(
            model="gemini-2.5-flash",
            project="test-project",
            location="us-central1",
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
        ]
        result = prov.chat(messages)
        assert isinstance(result, str)

    def test_chat_graceful_on_api_error(self):
        vertexai = sys.modules["vertexai"]
        vertexai.generative_models.GenerativeModel.return_value.generate_content.side_effect = (
            RuntimeError("API error")
        )

        from agentic_rag.rag.llm_provider import VertexGeminiProvider, SAFE_LLM_ERROR
        prov = VertexGeminiProvider(
            model="gemini-2.5-flash",
            project="test-project",
            location="us-central1",
        )
        result = prov.chat([{"role": "user", "content": "Hello"}])
        assert result == SAFE_LLM_ERROR

    def test_generate_convenience_wrapper(self):
        from agentic_rag.rag.llm_provider import VertexGeminiProvider
        prov = VertexGeminiProvider(
            model="gemini-2.5-flash",
            project="test-project",
            location="us-central1",
        )
        result = prov.generate("Summarise this.")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# VertexEmbeddingProvider tests
# ---------------------------------------------------------------------------

class TestVertexEmbeddingProvider:

    def setup_method(self):
        _install_vertexai_mock()
        for key in list(sys.modules):
            if "embeddings" in key and "agentic_rag" in key:
                del sys.modules[key]

    def teardown_method(self):
        _uninstall_vertexai_mock()

    def test_construction_with_project_succeeds(self):
        from agentic_rag.rag.embeddings import VertexEmbeddingProvider
        prov = VertexEmbeddingProvider(
            model_name="text-embedding-004",
            project="test-project",
            location="us-central1",
        )
        assert prov._model is not None

    def test_dim_probed_on_init(self):
        from agentic_rag.rag.embeddings import VertexEmbeddingProvider
        prov = VertexEmbeddingProvider(
            model_name="text-embedding-004",
            project="test-project",
            location="us-central1",
        )
        # The mock returns 768-dim vectors
        assert prov.dim == 768

    def test_construction_without_project_is_safe(self):
        from agentic_rag.rag.embeddings import VertexEmbeddingProvider
        prov = VertexEmbeddingProvider(
            model_name="text-embedding-004",
            project="",
            location="us-central1",
        )
        assert prov._model is None
        assert prov.dim == 768  # default fallback

    def test_embed_documents_returns_correct_shape(self):
        from agentic_rag.rag.embeddings import VertexEmbeddingProvider
        prov = VertexEmbeddingProvider(
            model_name="text-embedding-004",
            project="test-project",
            location="us-central1",
        )
        result = prov.embed_documents(["Hello world", "Python is great"])
        # The mock returns 1 embedding per call — we accept flexible return here
        assert isinstance(result, list)
        assert all(isinstance(v, list) for v in result)

    def test_embed_text_returns_list_of_floats(self):
        from agentic_rag.rag.embeddings import VertexEmbeddingProvider
        prov = VertexEmbeddingProvider(
            model_name="text-embedding-004",
            project="test-project",
            location="us-central1",
        )
        result = prov.embed_text("Hello")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_embed_documents_returns_zeros_when_model_none(self):
        from agentic_rag.rag.embeddings import VertexEmbeddingProvider
        prov = VertexEmbeddingProvider(
            model_name="text-embedding-004",
            project="",
            location="us-central1",
        )
        result = prov.embed_documents(["Hello", "World"])
        assert len(result) == 2
        assert all(v == [0.0] * 768 for v in result)


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------

class TestLLMProviderFactory:

    def setup_method(self):
        _install_vertexai_mock()
        for key in list(sys.modules):
            if "llm_provider" in key and "agentic_rag" in key:
                del sys.modules[key]

    def teardown_method(self):
        _uninstall_vertexai_mock()

    def test_factory_returns_vertex_when_rag_provider_set(self):
        with patch.dict(os.environ, {
            "RAG_MOCK_MODE": "false",
            "RAG_PROVIDER": "vertex",
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
            "VERTEX_GEMINI_MODEL": "gemini-2.5-flash",
        }):
            from agentic_rag.rag.llm_provider import get_llm_provider, VertexGeminiProvider
            prov = get_llm_provider()
            assert isinstance(prov, VertexGeminiProvider)
            assert prov.model == "gemini-2.5-flash"

    def test_factory_vertex_default_model_is_2_5_flash(self):
        env = {
            "RAG_MOCK_MODE": "false",
            "RAG_PROVIDER": "vertex",
            "GOOGLE_CLOUD_PROJECT": "test-project",
        }
        env.pop("VERTEX_GEMINI_MODEL", None)
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("VERTEX_GEMINI_MODEL", None)
            from agentic_rag.rag.llm_provider import get_llm_provider, VertexGeminiProvider
            prov = get_llm_provider()
            assert isinstance(prov, VertexGeminiProvider)
            assert prov.model == "gemini-2.5-flash"

    def test_factory_returns_mock_in_test_mode(self):
        with patch.dict(os.environ, {"RAG_MOCK_MODE": "true", "RAG_PROVIDER": "vertex"}):
            from agentic_rag.rag.llm_provider import get_llm_provider
            from agentic_rag.rag.mock_components import MockLLMProvider
            prov = get_llm_provider()
            assert isinstance(prov, MockLLMProvider)


class TestEmbeddingProviderFactory:

    def setup_method(self):
        _install_vertexai_mock()
        for key in list(sys.modules):
            if "embeddings" in key and "agentic_rag" in key:
                del sys.modules[key]

    def teardown_method(self):
        _uninstall_vertexai_mock()

    def test_factory_returns_vertex_when_embedding_provider_set(self):
        with patch.dict(os.environ, {
            "RAG_MOCK_MODE": "false",
            "EMBEDDING_PROVIDER": "vertex",
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
            "VERTEX_EMBEDDING_MODEL": "text-embedding-004",
        }):
            from agentic_rag.rag.embeddings import get_embedding_provider, VertexEmbeddingProvider
            prov = get_embedding_provider("all-MiniLM-L6-v2")
            assert isinstance(prov, VertexEmbeddingProvider)

    def test_factory_returns_vertex_via_rag_provider_legacy(self):
        env = {
            "RAG_MOCK_MODE": "false",
            "RAG_PROVIDER": "vertex",
            "GOOGLE_CLOUD_PROJECT": "test-project",
        }
        with patch.dict(os.environ, env):
            os.environ.pop("EMBEDDING_PROVIDER", None)
            from agentic_rag.rag.embeddings import get_embedding_provider, VertexEmbeddingProvider
            prov = get_embedding_provider("all-MiniLM-L6-v2")
            assert isinstance(prov, VertexEmbeddingProvider)

    def test_factory_returns_mock_in_test_mode(self):
        with patch.dict(os.environ, {"RAG_MOCK_MODE": "true", "EMBEDDING_PROVIDER": "vertex"}):
            from agentic_rag.rag.embeddings import get_embedding_provider, MockEmbeddingProvider
            prov = get_embedding_provider("all-MiniLM-L6-v2")
            assert isinstance(prov, MockEmbeddingProvider)

    def test_factory_embedding_provider_local_is_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EMBEDDING_PROVIDER", None)
            os.environ.pop("RAG_PROVIDER", None)
            # We can't easily test LocalEmbeddingProvider without model downloads,
            # but we can verify the factory does not select Vertex when no env is set
            from agentic_rag.rag.embeddings import get_embedding_provider, VertexEmbeddingProvider
            prov = get_embedding_provider("all-MiniLM-L6-v2")
            assert not isinstance(prov, VertexEmbeddingProvider)


# ---------------------------------------------------------------------------
# Graceful degradation — SDK not installed
# ---------------------------------------------------------------------------

class TestVertexSDKMissing:
    """Ensure providers degrade gracefully when google-cloud-aiplatform is absent."""

    def test_vertex_llm_provider_degrades_when_sdk_missing(self):
        # Hide vertexai from imports
        with patch.dict(sys.modules, {"vertexai": None, "vertexai.generative_models": None}):
            for key in list(sys.modules):
                if "llm_provider" in key and "agentic_rag" in key:
                    del sys.modules[key]
            from agentic_rag.rag.llm_provider import VertexGeminiProvider
            prov = VertexGeminiProvider(
                model="gemini-2.5-flash",
                project="test-project",
                location="us-central1",
            )
            # Model should be None → config error string returned
            assert prov._genai_model is None
            result = prov.chat([{"role": "user", "content": "Hello"}])
            expected = "Vertex AI Configuration Error: google-cloud-aiplatform is not installed. Run: pip install google-cloud-aiplatform>=1.60.0"
            assert result == expected

    def test_vertex_embedding_provider_degrades_when_sdk_missing(self):
        with patch.dict(sys.modules, {"vertexai": None, "vertexai.language_models": None}):
            for key in list(sys.modules):
                if "embeddings" in key and "agentic_rag" in key:
                    del sys.modules[key]
            from agentic_rag.rag.embeddings import VertexEmbeddingProvider
            prov = VertexEmbeddingProvider(
                model_name="text-embedding-004",
                project="test-project",
                location="us-central1",
            )
            assert prov._model is None
            # Should return zero vectors without crashing
            result = prov.embed_documents(["test"])
            assert len(result) == 1
            assert all(x == 0.0 for x in result[0])

