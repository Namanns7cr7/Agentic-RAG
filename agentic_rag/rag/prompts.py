from typing import List

def rag_prompt(contexts: List[str], question: str) -> str:
    context = "\n".join(f"- {c}" for c in contexts) if contexts else "No context."
    return f"""You are a helpful assistant that answers using the provided context.
If you don't know, say so briefly.

Context:
{context}

Question: {question}
Answer:
"""

def reflection_prompt(question: str, draft: str) -> str:
    return f"""You wrote the following draft answer:
---
{draft}
---
Double-check for hallucinations or unsupported claims. If something lacks support
from the context, tone it down or say you don't know. Provide a concise, corrected answer.

Question: {question}
Final Answer:
"""
