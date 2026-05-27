"""Generates personalised concept explanations via the LLM."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..rag.llm_provider import LLMProvider
from ..rag.utils_logger import get_logger
from .learning_prompts import (
    LEARNMATE_SYSTEM_PROMPT,
    concept_explanation_prompt,
    reflection_prompt,
)

logger = get_logger(__name__)

# Default fallback when the LLM returns unparseable output
_FALLBACK_EXPLANATION: Dict[str, Any] = {
    "explanation": "",
    "analogy": "",
    "real_life_example": "",
    "key_points": [],
    "quick_check_question": "",
    "next_step": "",
}


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Try to extract a JSON object from potentially messy LLM output."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first { … } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}


class ConceptExplainer:
    """Uses the LLM to produce a structured, personalised explanation."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def explain(
        self,
        topic: str,
        level: str,
        goal: str,
        strategy_instruction: str,
        retrieved_context: Optional[List[str]] = None,
        preferred_style: str = "simple",
        learner_weak_topics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        user_prompt = concept_explanation_prompt(
            topic=topic,
            level=level,
            goal=goal,
            strategy_instruction=strategy_instruction,
            retrieved_context=retrieved_context,
            preferred_style=preferred_style,
            learner_weak_topics=learner_weak_topics,
        )

        raw = self.llm.chat(
            [
                {"role": "system", "content": LEARNMATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        logger.debug("ConceptExplainer raw output: %s", raw[:300])

        result = _parse_json_response(raw)
        if not result:
            # If parsing failed entirely, wrap the raw text
            logger.warning("Failed to parse concept explanation JSON — using raw text")
            result = {**_FALLBACK_EXPLANATION, "explanation": raw}

        # Ensure all keys exist
        for key, default in _FALLBACK_EXPLANATION.items():
            result.setdefault(key, default)

        return result

    def reflect_and_improve(
        self,
        draft_json: str,
        level: str,
        retrieved_context: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the reflection/self-check step on the draft."""
        context = "\n".join(f"- {c}" for c in (retrieved_context or [])) or "No retrieved context."
        prompt = reflection_prompt(draft_json, level, context)
        raw = self.llm.chat(
            [
                {"role": "system", "content": LEARNMATE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        result = _parse_json_response(raw)
        if not result:
            logger.warning("Reflection produced unparseable output — keeping draft")
            return _parse_json_response(draft_json) or {**_FALLBACK_EXPLANATION}
        for key, default in _FALLBACK_EXPLANATION.items():
            result.setdefault(key, default)
        return result
