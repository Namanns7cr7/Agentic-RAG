"""Prompt templates for the Agentic RAG pipeline."""

from typing import List, Tuple


def rag_prompt(contexts: List[str], question: str, memory: List[Tuple[str, str]] | None = None) -> str:
    context = "\n".join(f"- {c}" for c in contexts) if contexts else "No context available."
    mem_block = ""
    if memory:
        lines = []
        for q, a in memory:
            lines.append(f"User: {q}")
            lines.append(f"Assistant: {a}")
        mem_block = "Conversation history:\n" + "\n".join(lines) + "\n\n"

    return f"""You are a helpful, knowledgeable assistant that answers using the provided context.
If you don't know, say so honestly and briefly.

{mem_block}Context:
{context}

Question: {question}
Answer:
"""


def reflection_prompt(question: str, draft: str) -> str:
    return f"""You wrote the following draft answer:
---
{draft}
---
Double-check for hallucinations, factual errors, or unsupported claims.
If something lacks support from the context, tone it down or say you don't know.
Provide a concise, corrected final answer.

Question: {question}
Final Answer:
"""


def planner_prompt(question: str, tool_names: List[str]) -> str:
    tools_str = ", ".join(tool_names) if tool_names else "none"
    return f"""You are a planning agent. Decide the best action for this question.

Available tools: {tools_str}
Available actions:
- "retrieve" — search the knowledge base for relevant documents
- "use_tool:<tool_name>" — use a specific tool

Reply with ONLY the action string, nothing else.

Question: {question}
Action:
"""
