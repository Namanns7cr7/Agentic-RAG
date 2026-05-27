"""Main orchestrator for the LearnMate AI learning flow.

Flow:
  Understand Learner → Plan Learning Strategy → Retrieve/Teach/Quiz/Evaluate
  → Reflect → Remember → Adapt Next Step
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..rag.llm_provider import LLMProvider, get_llm_provider
from ..rag.utils_logger import get_logger
from .answer_evaluator import AnswerEvaluator
from .concept_explainer import ConceptExplainer
from .learner_profile import LearnerProfile
from .learning_memory import LearningMemory
from .learning_path import LearningPathGenerator
from .learning_prompts import LEARNMATE_SYSTEM_PROMPT, teach_from_docs_prompt
from .misconception_detector import MisconceptionDetector
from .quiz_generator import QuizGenerator
from .teaching_strategy import choose_strategy, get_strategy_instruction

logger = get_logger(__name__)


class LearningAgent:
    """Orchestrates the personalised learning pipeline."""

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        memory: Optional[LearningMemory] = None,
        retriever=None,
    ):
        self.llm = llm or get_llm_provider()
        self.memory = memory or LearningMemory()
        self.retriever = retriever  # existing RAG retriever (optional)

        # Sub-components
        self.explainer = ConceptExplainer(self.llm)
        self.evaluator = AnswerEvaluator(self.llm)
        self.quiz_gen = QuizGenerator(self.llm)
        self.misconception = MisconceptionDetector(self.llm)
        self.path_gen = LearningPathGenerator(self.llm)

    # ── /learn ────────────────────────────────────────────────────

    def learn(
        self,
        user_id: str,
        topic: str,
        level: str = "auto",
        goal: str = "conceptual",
        context: Optional[str] = None,
        preferred_style: str = "simple",
    ) -> Dict[str, Any]:
        # 1. Understand learner
        profile = self.memory.get(user_id)
        profile.touch()
        profile.set_current_topic(topic)
        profile.preferred_style = preferred_style

        # 2. Detect level
        detected_level = self._detect_level(level, profile)
        profile.level = detected_level

        # 3. Choose strategy
        strategy = choose_strategy(detected_level, goal, preferred_style, profile)
        strategy_instruction = get_strategy_instruction(strategy)

        # 4. Optionally retrieve context
        retrieved: Optional[List[str]] = None
        used_retrieval = False
        if self.retriever:
            try:
                raw = self.retriever.retrieve(topic, top_k=3)
                retrieved = [r.get("text", "") for r in raw]
                if retrieved:
                    used_retrieval = True
            except Exception as exc:
                logger.warning("Retrieval failed: %s", exc)

        # 5. Generate explanation
        explanation = self.explainer.explain(
            topic=topic,
            level=detected_level,
            goal=goal,
            strategy_instruction=strategy_instruction,
            retrieved_context=retrieved,
            preferred_style=preferred_style,
            learner_weak_topics=profile.weak_topics,
        )

        # 6. Reflect
        draft_json = json.dumps(explanation, ensure_ascii=False)
        improved = self.explainer.reflect_and_improve(draft_json, detected_level, retrieved_context=retrieved)

        # 7. Remember
        self.memory.save(profile)

        return {
            "topic": topic,
            "detected_level": detected_level,
            "teaching_strategy": strategy,
            "explanation": improved.get("explanation", ""),
            "analogy": improved.get("analogy", ""),
            "real_life_example": improved.get("real_life_example", ""),
            "key_points": improved.get("key_points", []),
            "quick_check_question": improved.get("quick_check_question", ""),
            "next_step": improved.get("next_step", ""),
            "retrieved_sources_used": used_retrieval,
        }

    # ── /evaluate ─────────────────────────────────────────────────

    def evaluate(
        self,
        user_id: str,
        topic: str,
        question: str,
        answer: str,
    ) -> Dict[str, Any]:
        profile = self.memory.get(user_id)
        profile.touch()

        result = self.evaluator.evaluate(topic, question, answer)

        # Misconception detection
        if result.get("correctness") != "correct":
            misconception = self.misconception.detect(topic, question, answer)
            if misconception:
                result["misconception"] = misconception
                profile.add_misconception(misconception)

        # Update profile
        score = result.get("score", 0)
        profile.add_score(score)

        pace = result.get("pace_decision", "continue")
        profile.update_pace(pace)

        if score <= 2:
            profile.mark_weak(topic)

        self.memory.save(profile)
        return result

    # ── /profile ──────────────────────────────────────────────────

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        profile = self.memory.get(user_id)
        return profile.to_dict()

    # ── /flashcards ───────────────────────────────────────────────

    def flashcards(self, user_id: str, topic: str, count: int = 5, level: str = "beginner") -> List[Dict[str, str]]:
        profile = self.memory.get(user_id)
        profile.touch()
        profile.set_current_topic(topic)
        self.memory.save(profile)
        return self.quiz_gen.generate_flashcards(topic, count, level)

    # ── /learning-path ────────────────────────────────────────────

    def learning_path(self, user_id: str, topic: str, goal: str = "conceptual") -> Dict[str, Any]:
        profile = self.memory.get(user_id)
        profile.touch()
        self.memory.save(profile)
        steps = self.path_gen.generate(topic, goal)
        return {"topic": topic, "learning_path": steps}

    # ── /teach-from-docs ──────────────────────────────────────────

    def teach_from_docs(self, user_id: str, query: str, level: str = "beginner") -> Dict[str, Any]:
        profile = self.memory.get(user_id)
        profile.touch()

        docs: List[str] = []
        if self.retriever:
            try:
                raw = self.retriever.retrieve(query, top_k=5)
                docs = [r.get("text", "") for r in raw]
            except Exception as exc:
                logger.warning("Retrieval failed in teach-from-docs: %s", exc)

        prompt = teach_from_docs_prompt(docs, query, level)
        raw = self.llm.chat(
            [{"role": "system", "content": LEARNMATE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=1024, temperature=0.7,
        )

        # Parse response
        result = self._parse_json(raw)
        if not result:
            result = {
                "explanation": raw,
                "analogy": "",
                "real_life_example": "",
                "key_points": [],
                "quick_check_question": "",
                "next_step": "",
            }
        for key in ("explanation", "analogy", "real_life_example", "quick_check_question", "next_step"):
            result.setdefault(key, "")
        result.setdefault("key_points", [])
        result["retrieved_sources_used"] = bool(docs)

        self.memory.save(profile)
        return result

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _detect_level(requested: str, profile: LearnerProfile) -> str:
        if requested != "auto":
            return requested
        avg = profile.average_score
        if avg == 0 and not profile.scores:
            return "beginner"
        if avg < 2:
            return "beginner"
        if avg < 3.5:
            return "intermediate"
        return "advanced"

    @staticmethod
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
