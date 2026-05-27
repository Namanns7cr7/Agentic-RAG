"""Generates a structured learning path for a topic."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..rag.llm_provider import LLMProvider
from ..rag.utils_logger import get_logger
from .learning_prompts import LEARNMATE_SYSTEM_PROMPT, learning_path_prompt

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
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except json.JSONDecodeError:
                pass
    return {}


class LearningPathGenerator:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def generate(self, topic: str, goal: str = "conceptual") -> List[Dict[str, Any]]:
        prompt = learning_path_prompt(topic, goal)
        raw = self.llm.chat(
            [{"role": "system", "content": LEARNMATE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=1024, temperature=0.7,
        )
        parsed = _parse_json(raw)
        path = parsed.get("learning_path", [])
        if not path:
            logger.warning("Learning path generation returned empty — building fallback")
            path = [
                {"step": 1, "subtopic": topic, "why_it_matters": "Core concept", "estimated_difficulty": "medium"},
            ]
        valid: List[Dict[str, Any]] = []
        for i, item in enumerate(path, 1):
            if isinstance(item, dict):
                valid.append({
                    "step": item.get("step", i),
                    "subtopic": item.get("subtopic", f"Step {i}"),
                    "why_it_matters": item.get("why_it_matters", ""),
                    "estimated_difficulty": item.get("estimated_difficulty", "medium"),
                })
        return valid or [{"step": 1, "subtopic": topic, "why_it_matters": "Core concept", "estimated_difficulty": "medium"}]
