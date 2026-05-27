"""Generates adaptive quizzes and flashcards."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..rag.llm_provider import LLMProvider
from ..rag.utils_logger import get_logger
from .learning_prompts import LEARNMATE_SYSTEM_PROMPT, flashcard_prompt

logger = get_logger(__name__)


def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}


class QuizGenerator:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def generate_flashcards(self, topic: str, count: int = 5, level: str = "beginner") -> List[Dict[str, str]]:
        prompt = flashcard_prompt(topic, count, level)
        raw = self.llm.chat(
            [{"role": "system", "content": LEARNMATE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=1024, temperature=0.7,
        )
        parsed = _parse_json(raw)
        cards = parsed.get("flashcards", [])
        if not cards:
            cards = [{"front": f"What is {topic}?", "back": f"A concept related to {topic}."}]
        valid: List[Dict[str, str]] = []
        for c in cards:
            if isinstance(c, dict) and "front" in c and "back" in c:
                valid.append({"front": str(c["front"]), "back": str(c["back"])})
        return valid or [{"front": f"What is {topic}?", "back": f"A concept related to {topic}."}]
