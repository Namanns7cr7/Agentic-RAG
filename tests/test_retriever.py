from agentic_rag.rag.config import Settings
from agentic_rag.rag.pipeline import Pipeline

def test_retriever_basic():
    pipe = Pipeline(Settings(), seed_docs=[
        "RAG stands for Retrieval Augmented Generation.",
        "The sky is blue."
    ])
    res = pipe.answer("What does RAG stand for?", top_k=2)
    assert "RAG" in res["final"]
