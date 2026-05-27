"""High-level RAG orchestrator — wires together all components."""

from typing import Dict, Any, List, Optional
from .config import Settings
from .embeddings import Embeddings
from .vectorstore import VectorStore
from .retriever import Retriever
from .llm import LocalGenerator
from .agent import Agent
from .tools import CalculatorTool, TimeTool, SummarizeTool, WebScraperTool
from .utils_logger import get_logger

logger = get_logger(__name__)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        seed_docs: List[str],
        *,
        embedder: Optional[Embeddings] = None,
        store: Optional[VectorStore] = None,
        generator: Optional[LocalGenerator] = None,
    ):
        self.settings = settings
        self.embedder = embedder or Embeddings(settings.embedding_model_name)
        self.store = store or VectorStore(self.embedder.dim)
        self.generator = generator or LocalGenerator(settings.generator_model_name)
        self.retriever = Retriever(self.embedder, self.store)
        tools = {
            "calculator": CalculatorTool(),
            "time": TimeTool(),
            "summarize": SummarizeTool(),
            "scrape": WebScraperTool(),
        }
        self.agent = Agent(
            self.retriever,
            self.generator,
            tools,
            max_memory=settings.max_memory_turns,
        )
        if seed_docs:
            self.add_documents(seed_docs, doc_id="seed")

    def add_documents(self, docs: List[str], *, doc_id: Optional[str] = None) -> int:
        if not docs:
            return 0
        embs = self.embedder.encode(docs)
        metadatas = [{"doc_id": doc_id or ""} for _ in docs]
        self.store.add(docs, embs, metadatas=metadatas)
        logger.info("Added %d documents. Total=%d", len(docs), self.store.size)
        return len(docs)

    def answer(self, question: str, top_k: int = None) -> Dict[str, Any]:
        top_k = top_k or self.settings.top_k_default
        plan = self.agent.plan(question)
        logger.info("Plan: %s", plan)

        step = self.agent.act(plan, question, top_k)
        draft = step.get("draft", "")
        final = self.agent.reflect(question, draft)

        # Store in conversation memory
        self.agent.remember(question, final)

        return {
            "plan": plan,
            "draft": draft,
            "final": final,
            "observation": step.get("observation"),
        }

    def clear_memory(self) -> None:
        """Reset conversation memory."""
        self.agent.clear_memory()

    @property
    def doc_count(self) -> int:
        return self.store.size
