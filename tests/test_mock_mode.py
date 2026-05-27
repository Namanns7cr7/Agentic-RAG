from agentic_rag import app as app_module
from agentic_rag.rag.mock_components import MockEmbeddings, MockGenerator


def test_app_uses_mock_components_in_test_mode():
    pipe = app_module._services.pipeline()
    assert isinstance(pipe.embedder, MockEmbeddings)
    assert isinstance(pipe.generator, MockGenerator)
