"""Evaluates learner answers and provides feedback."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..rag.llm_provider import LLMProvider
from ..rag.utils_logger import get_logger
from .learning_prompts import LEARNMATE_SYSTEM_PROMPT, evaluation_prompt

logger = get_logger(__name__)

_FALLBACK: Dict[str, Any] = {
    "score": 0,
    "correctness": "incorrect",
    "feedback": "Could not evaluate the answer at this time.",
    "missing_points": [],
    "misconception": "",
    "pace_decision": "continue",
    "next_explanation_style": "simple",
    "next_question": "",
}


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


class AnswerEvaluator:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def evaluate(self, topic: str, question: str, answer: str) -> Dict[str, Any]:
        prompt = evaluation_prompt(topic, question, answer)
        raw = self.llm.chat(
            [{"role": "system", "content": LEARNMATE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=512, temperature=0.3,
        )
        result = _parse_json(raw)
        if not result:
            logger.warning("Evaluation JSON parse failed — using fallback")
            result = {**_FALLBACK, "feedback": raw}
        for k, v in _FALLBACK.items():
            result.setdefault(k, v)
        # Clamp score
        try:
            result["score"] = max(0, min(5, int(result["score"])))
        except (ValueError, TypeError):
            result["score"] = 0
        return result
