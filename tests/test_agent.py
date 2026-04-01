"""Agent and pipeline tests."""

import pytest
from agentic_rag.rag.config import Settings
from agentic_rag.rag.pipeline import Pipeline


@pytest.fixture(scope="module")
def pipe():
    """Shared pipeline instance for the test module (avoids reloading the model)."""
    return Pipeline(
        Settings(),
        seed_docs=[
            "RAG stands for Retrieval Augmented Generation.",
            "Python is a popular programming language.",
            "The sky is blue on a clear day.",
            "FAISS enables fast vector similarity search.",
            "Transformers use self-attention mechanisms.",
        ],
    )


def test_retriever_returns_results(pipe):
    res = pipe.answer("What does RAG stand for?", top_k=2)
    assert "final" in res
    assert res["plan"] == "retrieve"


def test_calculator_tool(pipe):
    out = pipe.answer("calculate 2*(3+4)")
    assert out["plan"] == "use_tool:calculator"
    assert "14" in out.get("observation", "") or "14" in out.get("draft", "") or "14" in out.get("final", "")


def test_time_tool(pipe):
    out = pipe.answer("What time is it right now?")
    assert out["plan"] == "use_tool:time"
    assert out["observation"] is not None
    # Observation should look like an ISO timestamp
    assert "T" in out["observation"]


def test_conversation_memory(pipe):
    pipe.clear_memory()
    pipe.answer("What is Python?")
    assert len(pipe.agent.memory) == 1
    pipe.answer("Tell me about FAISS.")
    assert len(pipe.agent.memory) == 2
    pipe.clear_memory()
    assert len(pipe.agent.memory) == 0


def test_add_documents(pipe):
    before = pipe.doc_count
    pipe.add_documents(["New document for testing."])
    assert pipe.doc_count == before + 1


def test_add_empty_documents(pipe):
    before = pipe.doc_count
    added = pipe.add_documents([])
    assert added == 0
    assert pipe.doc_count == before
