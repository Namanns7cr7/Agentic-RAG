from typing import Dict, Any, List
from .config import Settings
from .embeddings import Embeddings
from .vectorstore import VectorStore
from .retriever import Retriever
from .llm import LocalGenerator
from .agent import Agent
from .tools import CalculatorTool, TimeTool
from .utils_logger import get_logger
import numpy as np

logger = get_logger(__name__)

class Pipeline:
    def __init__(self, settings: Settings, seed_docs: List[str]):
        self.settings = settings
        self.embedder = Embeddings(settings.embedding_model_name)
        self.store = VectorStore(self.embedder.dim)
        self.generator = LocalGenerator(settings.generator_model_name)
        self.retriever = Retriever(self.embedder, self.store)
        tools = {
            "calculator": CalculatorTool(),
            "time": TimeTool(),
        }
        self.agent = Agent(self.retriever, self.generator, tools)
        self.add_documents(seed_docs)

    def add_documents(self, docs: List[str]) -> int:
        if not docs:
            return 0
        embs = self.embedder.encode(docs)
        self.store.add(docs, embs)
        logger.info("Added %d documents. Total=%d", len(docs), len(self.store.docs))
        return len(docs)

    def answer(self, question: str, top_k: int = None) -> Dict[str, Any]:
        top_k = top_k or self.settings.top_k_default
        plan = self.agent.plan(question)
        logger.info("Plan: %s", plan)
        step = self.agent.act(plan, question, top_k)
        draft = step.get("draft", "")
        final = self.agent.reflect(question, draft)
        return {
            "plan": plan,
            "draft": draft,
            "final": final,
            "observation": step.get("observation"),
        }
