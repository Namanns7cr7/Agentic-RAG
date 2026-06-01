"""Lightweight mock components for test mode (no heavy model downloads)."""

from __future__ import annotations

import json
import re
from typing import Dict, List

import numpy as np

from .llm_provider import LLMProvider


class MockEmbeddings:
    """Deterministic embedding mock using simple hashing."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self._dim, dtype="float32")
            for i, ch in enumerate(text.encode("utf-8")):
                vec[i % self._dim] += float(ch)
            norm = np.linalg.norm(vec) or 1.0
            vectors.append(vec / norm)
        return np.vstack(vectors)

    @property
    def dim(self) -> int:
        return self._dim


class MockGenerator:
    """Simple generator that echoes a deterministic response."""

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        return "Mock response."


class MockLLMProvider(LLMProvider):
    """Mock LLM that returns valid JSON based on the prompt."""

    def chat(self, messages: List[Dict[str, str]], *, max_tokens: int = 512, temperature: float = 0.7) -> str:
        prompt = messages[-1]["content"] if messages else ""

        doc_line = ""
        if "Retrieved docs:" in prompt:
            for line in prompt.splitlines():
                stripped = line.strip()
                if stripped.startswith("[") or (stripped.startswith("-") and not stripped.startswith("---")):
                    doc_line = line
                    break
        if doc_line:
            doc_text = doc_line.split("]", 1)[-1].strip() if "]" in doc_line else doc_line.lstrip("-").strip()
        else:
            doc_text = ""

        if "\"questions\"" in prompt and "multiple-choice" in prompt:
            count = 3
            match = re.search(r"Create (\d+) multiple-choice", prompt)
            if match:
                count = int(match.group(1))
            questions = []
            for i in range(count):
                options = [f"Option {j}" for j in range(1, 5)]
                questions.append(
                    {
                        "question": f"Mock question {i + 1}?",
                        "options": options,
                        "correct_answer": options[0],
                        "explanation": "Mock explanation.",
                        "source_indices": [1],
                    }
                )
            return json.dumps({"questions": questions})

        answer = doc_text or "Mock grounded answer."
        return json.dumps(
            {
                "answer": answer,
                "code_example": "print('mock')",
                "mini_challenge": "Try a small task.",
                "quick_check": "What did you learn?",
                "next_step": "Read the next section.",
            }
        )
