from agentic_rag.learning.doc_learning_agent import DocLearningAgent
from agentic_rag.rag.llm_provider import SAFE_LLM_ERROR
from agentic_rag.rag.mock_components import MockEmbeddings
from agentic_rag.rag.vectorstore import VectorStore
from agentic_rag.rag.retriever import Retriever


class FailingLLM:
    def chat(self, messages, *, max_tokens=512, temperature=0.7):
        return SAFE_LLM_ERROR


def test_llm_failure_returns_clean_error():
    embedder = MockEmbeddings()
    store = VectorStore(embedder.dim)
    store.add(["RAG stands for Retrieval Augmented Generation."], embedder.encode(["RAG stands for Retrieval Augmented Generation."]), metadatas=[{"doc_id": "doc"}])
    retriever = Retriever(embedder, store)
    agent = DocLearningAgent(llm=FailingLLM(), retriever=retriever)

    res = agent.ask(
        user_id="u1",
        question="What does RAG stand for?",
        mode="general_qa",
        coding_level="beginner",
        preferred_language="python",
    )
    assert res["status"] == "error"
    assert res["answer"] == SAFE_LLM_ERROR
