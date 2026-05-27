"""Detects common misconceptions in learner answers."""

from __future__ import annotations

from ..rag.llm_provider import LLMProvider
from ..rag.utils_logger import get_logger
from .learning_prompts import LEARNMATE_SYSTEM_PROMPT, misconception_prompt

logger = get_logger(__name__)


class MisconceptionDetector:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def detect(self, topic: str, question: str, answer: str) -> str:
        prompt = misconception_prompt(topic, question, answer)
        raw = self.llm.chat(
            [{"role": "system", "content": LEARNMATE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=128, temperature=0.2,
        )
        result = raw.strip().strip('"').strip("'")
        if result.lower() in ("none", "no misconception", "n/a", ""):
            return ""
        return result
