"""Agentic loop: Plan → Act → Reflect with LLM-driven planning and conversation memory."""

from typing import List, Dict, Optional, Tuple
from .retriever import Retriever
from .prompts import rag_prompt, reflection_prompt, planner_prompt
from .llm import LocalGenerator
from .utils_logger import get_logger

logger = get_logger(__name__)


class Agent:
    def __init__(
        self,
        retriever: Retriever,
        generator: LocalGenerator,
        tools: Dict[str, object],
        max_memory: int = 5,
    ):
        self.retriever = retriever
        self.generator = generator
        self.tools = tools
        self.max_memory = max_memory
        self.memory: List[Tuple[str, str]] = []

    # ── Planning ──────────────────────────────────────────────────────

    def plan(self, question: str) -> str:
        """Use the LLM to decide the action, with keyword fallback for reliability."""

        # Fast-path heuristics for obvious cases (keeps things snappy)
        q = question.lower()
        if any(tok in q for tok in ["calculate", "compute", "what is", "how much"]) and any(
            c in question for c in "+-*/^%"
        ):
            return "use_tool:calculator"
        if any(tok in q for tok in ["what time", "current time", "date today", "today's date"]):
            return "use_tool:time"
        if any(tok in q for tok in ["summarize", "summarise", "key points", "bullet points"]):
            return "use_tool:summarize"
        if any(tok in q for tok in ["scrape", "read website", "fetch url", "read url", "download page"]):
            return "use_tool:scrape"

        # LLM-driven planning for ambiguous queries
        try:
            prompt = planner_prompt(question, list(self.tools.keys()))
            raw = self.generator.generate(prompt, max_new_tokens=20, temperature=0.1)
            action = raw.strip().split("\n")[0].strip().lower()
            # Validate the action
            if action == "retrieve":
                return "retrieve"
            if action.startswith("use_tool:"):
                tool_name = action.split(":", 1)[1]
                if tool_name in self.tools:
                    return action
            logger.warning("LLM planner returned invalid action '%s', falling back to retrieve", action)
        except Exception as e:
            logger.warning("LLM planner failed (%s), falling back to retrieve", e)

        return "retrieve"

    # ── Acting ────────────────────────────────────────────────────────

    def act(self, action: str, question: str, top_k: int) -> Dict[str, Optional[str]]:
        try:
            if action == "retrieve":
                contexts = self.retriever.retrieve(question, top_k=top_k)
                prompt = rag_prompt(contexts, question, memory=self.memory)
                draft = self.generator.generate(prompt)
                return {"draft": draft, "observation": None, "contexts": contexts}

            if action.startswith("use_tool:"):
                _, tool_name = action.split(":", 1)
                tool = self.tools.get(tool_name)
                if tool is None:
                    return {"draft": f"Tool '{tool_name}' not available.", "observation": None}
                obs = tool.run(question)
                prompt = (
                    f"Observation from {tool_name}: {obs}\n\n"
                    f"Question: {question}\n"
                    f"Provide a clear, helpful answer based on the observation.\nAnswer:"
                )
                draft = self.generator.generate(prompt)
                return {"draft": draft, "observation": obs}

            return {"draft": "Unknown action.", "observation": None}

        except Exception as e:
            logger.error("Error during action '%s': %s", action, e)
            return {"draft": f"Error during action '{action}': {e}", "observation": None}

    # ── Reflection ────────────────────────────────────────────────────

    def reflect(self, question: str, draft: str) -> str:
        try:
            prompt = reflection_prompt(question, draft)
            return self.generator.generate(prompt)
        except Exception:
            return draft  # Fail-safe: return draft unchanged

    # ── Memory ────────────────────────────────────────────────────────

    def remember(self, question: str, answer: str) -> None:
        """Store a Q&A pair in conversation memory."""
        self.memory.append((question, answer))
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory :]

    def clear_memory(self) -> None:
        self.memory.clear()
