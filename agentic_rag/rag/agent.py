from typing import List, Dict, Optional
from .retriever import Retriever
from .prompts import rag_prompt, reflection_prompt
from .llm import LocalGenerator

class Agent:
    def __init__(self, retriever: Retriever, generator: LocalGenerator, tools: Dict[str, object]):
        self.retriever = retriever
        self.generator = generator
        self.tools = tools

    def plan(self, question: str) -> str:
        # Very small planner: choose retrieval by default; detect calculator intent
        if any(tok in question.lower() for tok in ["calculate", "compute", "+", "-", "*", "/", "^"]):
            return "use_tool:calculator"
        return "retrieve"

    def act(self, action: str, question: str, top_k: int) -> Dict[str, Optional[str]]:
        obs = None
        contexts: List[str] = []
        try:
            if action == "retrieve":
                contexts = self.retriever.retrieve(question, top_k=top_k)
                prompt = rag_prompt(contexts, question)
                draft = self.generator.generate(prompt)
                return {"draft": draft, "observation": None}
            elif action.startswith("use_tool:"):
                _, tool_name = action.split(":", 1)
                tool = self.tools.get(tool_name)
                if tool is None:
                    return {"draft": f"Tool '{tool_name}' not available.", "observation": None}
                obs = tool.run(question)
                prompt = f"Observation from {tool_name}: {obs}\n\nQuestion: {question}\nAnswer:"
                draft = self.generator.generate(prompt)
                return {"draft": draft, "observation": obs}
            else:
                return {"draft": "Unknown action.", "observation": None}
        except Exception as e:
            return {"draft": f"Error during action '{action}': {e}", "observation": None}

    def reflect(self, question: str, draft: str) -> str:
        try:
            prompt = reflection_prompt(question, draft)
            return self.generator.generate(prompt)
        except Exception:
            return draft  # Fail-safe
