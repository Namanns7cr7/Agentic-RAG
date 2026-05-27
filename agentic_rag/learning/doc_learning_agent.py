"""DocuMentor AI: Agentic developer-learning orchestrator.

Extends the existing learning pipeline — does NOT duplicate it.

Flow:
    Understand → Classify Mode → Retrieve → Generate → Reflect → Remember → Adapt
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from ..rag.llm_provider import LLMProvider, get_llm_provider, SAFE_LLM_ERROR
from ..rag.grounding import verify_grounding
from ..rag.utils_logger import get_logger
from .learner_profile import LearnerProfile
from .learning_memory import LearningMemory
from .query_classifier import classify

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are DocuMentor AI, an Agentic-RAG assistant for developer learning.
Your goal is to make coding documentation easier, practical, and fun.

Rules:
- Ground all answers in the retrieved documentation chunks provided.
- If docs do not clearly mention something, say:
  "The uploaded docs do not clearly mention this."
- Do NOT invent API names, parameters, or behaviors.
- Tailor complexity to the learner's coding level.
- Always include a practical next step.
- Return responses as valid JSON (no markdown fences).
"""

# ---------------------------------------------------------------------------
# Per-mode prompt builders
# ---------------------------------------------------------------------------

def _explain_prompt(question: str, docs: List[str], level: str, language: str) -> str:
    ctx = _fmt_docs(docs)
    return f"""Using the retrieved documentation below, explain the concept to a {level} developer.

Retrieved docs:
{ctx}

Question: {question}
Preferred language: {language}

Return JSON:
{{
  "answer": "<clear, step-by-step explanation>",
  "code_example": "<small illustrative code snippet or empty string>",
  "mini_challenge": "<one small hands-on task>",
    "quick_check": "<one check question>",
    "next_step": "<what to study next>"
}}"""


def _code_example_prompt(question: str, docs: List[str], level: str, language: str) -> str:
    ctx = _fmt_docs(docs)
    return f"""Using the retrieved documentation, generate a runnable {language} code example for a {level} developer.

Retrieved docs:
{ctx}

Request: {question}

Return JSON:
{{
  "answer": "<brief explanation>",
  "code_example": "<complete runnable code with comments>",
  "mini_challenge": "<extend the example task>",
  "quick_check": "<one check question>",
  "next_step": "<next experiment>"
}}"""


def _challenge_prompt(question: str, docs: List[str], level: str, language: str) -> str:
    ctx = _fmt_docs(docs)
    return f"""Create a {level} coding challenge in {language} based on the retrieved documentation.

Retrieved docs:
{ctx}

Topic context: {question}

Return JSON:
{{
  "answer": "<challenge description>",
  "code_example": "<starter template or empty string>",
  "mini_challenge": "<the actual challenge task>",
  "quick_check": "<success criteria>",
  "next_step": "<harder follow-up task>"
}}"""


def _quiz_prompt(question: str, docs: List[str], level: str, num_questions: int) -> str:
    ctx = _fmt_docs(docs)
    return f"""Create {num_questions} multiple-choice quiz questions for a {level} developer based only on the retrieved docs.

Retrieved docs:
{ctx}

Topic: {question}

Return JSON:
{{
    "questions": [
        {{
            "question": "...",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "...",
            "source_indices": [1]
        }}
    ]
}}"""


def _cheat_sheet_prompt(question: str, docs: List[str]) -> str:
    ctx = _fmt_docs(docs)
    return f"""Create a concise developer cheat sheet from the retrieved documentation.

Retrieved docs:
{ctx}

Topic/focus: {question}

Return JSON:
{{
  "answer": "<cheat sheet with key concepts, syntax, and patterns>",
  "code_example": "<most important code pattern>",
  "mini_challenge": "<quick review task>",
  "quick_check": "<one self-test question>",
  "next_step": "<what to learn next>"
}}"""


def _debug_prompt(question: str, code_context: str, docs: List[str], language: str) -> str:
    ctx = _fmt_docs(docs)
    return f"""Use the retrieved documentation and the user's code to debug the issue.

Retrieved docs:
{ctx}

User's code / error:
{code_context or question}

Language: {language}

Return JSON:
{{
  "answer": "<likely issue and why it happens>",
  "code_example": "<corrected code>",
  "mini_challenge": "<refactoring suggestion>",
  "quick_check": "<prevention tip>",
  "next_step": "<what to read in docs to prevent this>"
}}"""


def _interview_prompt(question: str, docs: List[str], level: str) -> str:
    ctx = _fmt_docs(docs)
    return f"""Generate 5 interview-style questions for a {level} developer based only on the retrieved docs.

Retrieved docs:
{ctx}

Topic: {question}

Return JSON:
{{
  "answer": "<5 questions with model answers>",
  "code_example": "<code-based interview question>",
  "mini_challenge": "<take-home task>",
  "quick_check": "<hardest concept to check>",
  "next_step": "<topics to study before interviews>"
}}"""


def _general_qa_prompt(question: str, docs: List[str], level: str) -> str:
    ctx = _fmt_docs(docs)
    return f"""Answer the developer's question using the retrieved documentation.

Retrieved docs:
{ctx}

Question: {question}
Level: {level}

Return JSON:
{{
  "answer": "<grounded answer>",
  "code_example": "<relevant code snippet or empty string>",
  "mini_challenge": "<optional small task>",
  "quick_check": "<one check question>",
  "next_step": "<next learning step>"
}}"""


def _fmt_docs(docs: List[str]) -> str:
    if not docs:
        return "No documentation chunks retrieved."
    return "\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))


# ---------------------------------------------------------------------------
# Reflection prompt
# ---------------------------------------------------------------------------

_REFLECTION_PROMPT = """\
Review the following draft developer-learning response.

Draft:
---
{draft}
---

Retrieved docs:
{docs}

Check:
1. Is every claim grounded in the retrieved docs? If not, remove unsupported claims.
2. Are any API names, parameters, or behaviors invented? If yes, remove them.
3. Is the code syntactically reasonable?
4. Is the difficulty appropriate for a {level} developer?
5. Does the response include a practical next step?

Return the corrected JSON (same keys), with no markdown fences.
"""


# ---------------------------------------------------------------------------
# DocLearningAgent
# ---------------------------------------------------------------------------

class DocLearningAgent:
    """Orchestrates DocuMentor AI's developer-learning flow.

    Reuses the existing LLMProvider and LearningMemory — zero duplication.
    """

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        memory: Optional[LearningMemory] = None,
        retriever=None,
    ):
        self.llm = llm or get_llm_provider()
        self.memory = memory or LearningMemory()
        self.retriever = retriever

    # ── Main entry point ─────────────────────────────────────────────

    def ask(
        self,
        user_id: str,
        question: str,
        mode: str = "auto",
        coding_level: str = "auto",
        preferred_language: str = "python",
        preferred_style: str = "simple",
        code_context: Optional[str] = None,
        doc_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Main method: classify → retrieve → generate → reflect → remember."""

        # 1. Load profile
        profile = self.memory.get(user_id)
        profile.touch()

        # 2. Detect level
        level = self._resolve_level(coding_level, profile)
        language = preferred_language if preferred_language != "auto" else "python"

        # 3. Classify mode
        effective_mode = mode if mode != "auto" else classify(question)
        logger.info("DocAsk user=%s mode=%s level=%s", user_id, effective_mode, level)

        # 4. Retrieve chunks
        docs = self._retrieve(question if not code_context else code_context, top_k, doc_id=doc_id)
        used_doc_ids = self._collect_doc_ids(docs, doc_id)
        safety_flags = self._detect_prompt_injection(question)

        unsupported = self._unsupported_if_needed(
            docs,
            doc_id=doc_id,
            safety_flags=safety_flags,
        )
        if unsupported:
            return {
                "mode": effective_mode,
                "detected_level": level,
                "answer": unsupported["answer"],
                "status": "unsupported",
                "citations": [],
                "confidence": None,
                "unsupported_reason": unsupported["reason"],
                "used_doc_ids": used_doc_ids,
                "safety_flags": safety_flags or None,
                "code_example": "",
                "mini_challenge": "",
                "quick_check": "",
                "next_step": "",
            }

        # 5. Build mode-specific prompt
        prompt = self._build_prompt(effective_mode, question, docs, level, language, code_context, 3)

        # 6. Generate
        raw = self.llm.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        if raw.strip() == SAFE_LLM_ERROR:
            return {
                "mode": effective_mode,
                "detected_level": level,
                "answer": SAFE_LLM_ERROR,
                "status": "error",
                "citations": [],
                "confidence": None,
                "unsupported_reason": None,
                "used_doc_ids": used_doc_ids,
                "safety_flags": safety_flags or None,
                "code_example": "",
                "mini_challenge": "",
                "quick_check": "",
                "next_step": "",
            }

        # 7. Reflect
        reflected = self._reflect(raw, level, docs)

        # 8. Parse
        result = self._parse(reflected) or self._parse(raw) or {}

        # 9. Update memory (record weak topics if mode is quiz/challenge)
        if effective_mode in ("quiz", "challenge") and not result.get("answer"):
            profile.mark_weak(question[:50])
        self.memory.save(profile)

        answer = result.get("answer", "") or ""
        grounding = verify_grounding(answer, [d["text"] for d in docs])
        if not grounding["is_grounded"]:
            reason = "Answer contains unsupported claims."
            safety_flags = (safety_flags or []) + ["grounding_failed"]
            return {
                "mode": effective_mode,
                "detected_level": level,
                "answer": _UNSUPPORTED_MESSAGE,
                "status": "unsupported",
                "citations": [],
                "confidence": grounding.get("support_score"),
                "unsupported_reason": reason,
                "used_doc_ids": used_doc_ids,
                "safety_flags": safety_flags or None,
                "code_example": "",
                "mini_challenge": "",
                "quick_check": "",
                "next_step": "",
            }

        citations = self._build_citations(docs)

        return {
            "mode": effective_mode,
            "detected_level": level,
            "answer": answer,
            "status": "answered",
            "citations": citations,
            "confidence": grounding.get("support_score"),
            "unsupported_reason": None,
            "used_doc_ids": used_doc_ids,
            "safety_flags": safety_flags or None,
            "code_example": result.get("code_example", ""),
            "mini_challenge": result.get("mini_challenge", ""),
            "quick_check": result.get("quick_check", ""),
            "next_step": result.get("next_step", ""),
        }

    # ── Helpers ──────────────────────────────────────────────────────

    def _retrieve(self, query: str, top_k: int, *, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.retriever:
            return []
        try:
            return self.retriever.retrieve(query, top_k=top_k, doc_id=doc_id)
        except Exception as exc:
            logger.warning("DocAsk retrieval failed: %s", exc)
            return []

    def _unsupported_if_needed(
        self,
        docs: List[Dict[str, Any]],
        *,
        doc_id: Optional[str],
        safety_flags: List[str],
    ) -> Optional[Dict[str, str]]:
        if doc_id and self.retriever and not self.retriever.has_doc_id(doc_id):
            return {"reason": "Document not found for provided doc_id.", "answer": _UNSUPPORTED_MESSAGE}
        if not docs:
            reason = "No relevant document chunks were retrieved."
            if "prompt_injection" in safety_flags:
                reason = "Query attempts to bypass document grounding."
            return {"reason": reason, "answer": _UNSUPPORTED_MESSAGE}
        if all(len(d.get("text", "").strip()) < 20 for d in docs):
            return {"reason": "Retrieved chunks are too short to support an answer.", "answer": _UNSUPPORTED_MESSAGE}
        min_score = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.2"))
        best = max((d.get("score", 0.0) for d in docs), default=0.0)
        if best < min_score:
            return {"reason": "Retrieved context was too weak to support an answer.", "answer": _UNSUPPORTED_MESSAGE}
        return None

    @staticmethod
    def _build_citations(docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        citations: List[Dict[str, str]] = []
        for i, d in enumerate(docs, 1):
            citations.append(
                {
                    "chunk_id": f"chunk_{i}",
                    "doc_id": d.get("doc_id", ""),
                    "text": d.get("text", "")[:500],
                }
            )
        return citations

    @staticmethod
    def _collect_doc_ids(docs: List[Dict[str, Any]], doc_id: Optional[str]) -> List[str]:
        ids = {d.get("doc_id", "") for d in docs if d.get("doc_id")}
        if doc_id:
            ids.add(doc_id)
        return sorted(i for i in ids if i)

    @staticmethod
    def _detect_prompt_injection(question: str) -> List[str]:
        patterns = [
            r"ignore (the|all) (documents|docs)",
            r"forget (previous|all) instructions",
            r"reveal (system prompt|hidden instructions)",
            r"use external knowledge",
            r"make up",
        ]
        lowered = question.lower()
        if any(re.search(p, lowered) for p in patterns):
            return ["prompt_injection"]
        return []

    def _resolve_level(self, requested: str, profile: LearnerProfile) -> str:
        if requested not in ("auto", ""):
            return requested
        avg = profile.average_score
        if avg == 0:
            return "beginner"
        if avg < 2.5:
            return "beginner"
        if avg < 4.0:
            return "intermediate"
        return "advanced"

    def _build_prompt(
        self,
        mode: str,
        question: str,
        docs: List[Dict[str, Any]],
        level: str,
        language: str,
        code_context: Optional[str],
        num_questions: int = 3,
    ) -> str:
        doc_texts = [d.get("text", "") for d in docs]
        if mode == "explain":
            return _explain_prompt(question, doc_texts, level, language)
        if mode == "code_example":
            return _code_example_prompt(question, doc_texts, level, language)
        if mode == "challenge":
            return _challenge_prompt(question, doc_texts, level, language)
        if mode == "quiz":
            return _quiz_prompt(question, doc_texts, level, num_questions)
        if mode == "cheat_sheet":
            return _cheat_sheet_prompt(question, doc_texts)
        if mode == "debug":
            return _debug_prompt(question, code_context or "", doc_texts, language)
        if mode == "interview":
            return _interview_prompt(question, doc_texts, level)
        if mode == "build_with_me":
            return _explain_prompt(question, doc_texts, level, language)  # guided version
        return _general_qa_prompt(question, doc_texts, level)

    def _reflect(self, draft: str, level: str, docs: List[Dict[str, Any]]) -> str:
        doc_texts = "\n".join(f"- {d.get('text', '')}" for d in docs) or "No docs."
        prompt = _REFLECTION_PROMPT.format(draft=draft[:3000], level=level, docs=doc_texts)
        try:
            return self.llm.chat(
                [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
        except Exception as exc:
            logger.warning("Reflection failed, using original draft: %s", exc)
            return draft

    @staticmethod
    def _parse(text: str) -> Dict[str, Any]:
        text = text.strip()
        # Strip markdown fences
        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e > s:
                try:
                    return json.loads(text[s: e + 1])
                except json.JSONDecodeError:
                    pass
        return {}

    # ── Dedicated mode methods (called by specific endpoints) ─────────

    def generate_challenge(
        self,
        user_id: str,
        topic: str,
        difficulty: str = "intermediate",
        language: str = "python",
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        docs = self._retrieve(topic, top_k, doc_id=doc_id)
        used_doc_ids = self._collect_doc_ids(docs, doc_id)
        unsupported = self._unsupported_if_needed(docs, doc_id=doc_id, safety_flags=[])
        if unsupported:
            return {
                "challenge_title": f"{difficulty.capitalize()} Challenge: {topic}",
                "difficulty": difficulty,
                "task": _UNSUPPORTED_MESSAGE,
                "requirements": [],
                "hints": [],
                "expected_behavior": "",
                "optional_solution": "",
                "status": "unsupported",
                "citations": [],
                "unsupported_reason": unsupported["reason"],
                "used_doc_ids": used_doc_ids,
            }

        prompt = _challenge_prompt(topic, [d.get("text", "") for d in docs], difficulty, language)
        raw = self.llm.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=1024, temperature=0.7,
        )
        if raw.strip() == SAFE_LLM_ERROR:
            return {
                "challenge_title": f"{difficulty.capitalize()} Challenge: {topic}",
                "difficulty": difficulty,
                "task": SAFE_LLM_ERROR,
                "requirements": [],
                "hints": [],
                "expected_behavior": "",
                "optional_solution": "",
                "status": "error",
                "citations": [],
                "unsupported_reason": None,
                "used_doc_ids": used_doc_ids,
            }
        result = self._parse(raw) or {}
        return {
            "challenge_title": f"{difficulty.capitalize()} Challenge: {topic}",
            "difficulty": difficulty,
            "task": result.get("mini_challenge", result.get("answer", "")),
            "requirements": [],
            "hints": [],
            "expected_behavior": result.get("quick_check", ""),
            "optional_solution": result.get("code_example", ""),
            "status": "answered",
            "citations": self._build_citations(docs),
            "unsupported_reason": None,
            "used_doc_ids": used_doc_ids,
        }

    def generate_quiz(
        self,
        user_id: str,
        topic: str,
        difficulty: str = "intermediate",
        num_questions: int = 5,
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        docs = self._retrieve(topic, top_k, doc_id=doc_id)
        used_doc_ids = self._collect_doc_ids(docs, doc_id)
        unsupported = self._unsupported_if_needed(docs, doc_id=doc_id, safety_flags=[])
        if unsupported:
            return {
                "questions": [],
                "status": "unsupported",
                "unsupported_reason": unsupported["reason"],
                "used_doc_ids": used_doc_ids,
            }

        prompt = _quiz_prompt(topic, [d.get("text", "") for d in docs], difficulty, num_questions)
        raw = self.llm.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=1024, temperature=0.7,
        )
        if raw.strip() == SAFE_LLM_ERROR:
            return {
                "questions": [],
                "status": "error",
                "unsupported_reason": None,
                "used_doc_ids": used_doc_ids,
            }
        result = self._parse(raw) or {}
        questions = []
        for q in result.get("questions", []) or []:
            options = [o for o in q.get("options", []) if isinstance(o, str) and o.strip()]
            if len(options) != 4 or len(set(options)) != 4:
                continue
            correct = q.get("correct_answer", "")
            if correct not in options:
                continue
            explanation = q.get("explanation", "")
            if not explanation:
                continue
            source_indices = q.get("source_indices", [])
            citation = None
            if source_indices:
                idx = int(source_indices[0]) - 1
                if 0 <= idx < len(docs):
                    citation = self._build_citations([docs[idx]])[0]
            questions.append(
                {
                    "question": q.get("question", ""),
                    "options": options,
                    "correct_answer": correct,
                    "explanation": explanation,
                    "citation": citation,
                }
            )

        if len(questions) < num_questions:
            return {
                "questions": questions,
                "status": "unsupported",
                "unsupported_reason": "Insufficient grounded content to generate full quiz.",
                "used_doc_ids": used_doc_ids,
            }

        return {
            "questions": questions[:num_questions],
            "status": "answered",
            "unsupported_reason": None,
            "used_doc_ids": used_doc_ids,
        }

    def generate_cheatsheet(
        self,
        user_id: str,
        topic: str = "",
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        query = topic or "main concepts"
        docs = self._retrieve(query, top_k, doc_id=doc_id)
        used_doc_ids = self._collect_doc_ids(docs, doc_id)
        unsupported = self._unsupported_if_needed(docs, doc_id=doc_id, safety_flags=[])
        if unsupported:
            return {
                "title": f"Cheat Sheet: {topic or 'Key Concepts'}",
                "important_concepts": [],
                "syntax": [],
                "code_patterns": [],
                "common_errors": [],
                "best_practices": [],
                "status": "unsupported",
                "citations": [],
                "unsupported_reason": unsupported["reason"],
                "used_doc_ids": used_doc_ids,
            }

        prompt = _cheat_sheet_prompt(query, [d.get("text", "") for d in docs])
        raw = self.llm.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=1024, temperature=0.5,
        )
        if raw.strip() == SAFE_LLM_ERROR:
            return {
                "title": f"Cheat Sheet: {topic or 'Key Concepts'}",
                "important_concepts": [],
                "syntax": [],
                "code_patterns": [],
                "common_errors": [],
                "best_practices": [],
                "status": "error",
                "citations": [],
                "unsupported_reason": None,
                "used_doc_ids": used_doc_ids,
            }
        result = self._parse(raw) or {}
        return {
            "title": f"Cheat Sheet: {topic or 'Key Concepts'}",
            "important_concepts": [],
            "syntax": [result.get("answer", "")],
            "code_patterns": [result.get("code_example", "")],
            "common_errors": [],
            "best_practices": [result.get("next_step", "")],
            "status": "answered",
            "citations": self._build_citations(docs),
            "unsupported_reason": None,
            "used_doc_ids": used_doc_ids,
        }

    def debug_code(
        self,
        user_id: str,
        code: str,
        error_message: str = "",
        language: str = "python",
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        query = error_message or code[:200]
        docs = self._retrieve(query, top_k, doc_id=doc_id)
        used_doc_ids = self._collect_doc_ids(docs, doc_id)
        unsupported = self._unsupported_if_needed(docs, doc_id=doc_id, safety_flags=[])
        if unsupported:
            return {
                "issue_found": _UNSUPPORTED_MESSAGE,
                "why_it_happens": "",
                "corrected_code": "",
                "explanation": "",
                "prevention_tip": "",
                "status": "unsupported",
                "citations": [],
                "unsupported_reason": unsupported["reason"],
                "used_doc_ids": used_doc_ids,
            }

        prompt = _debug_prompt(error_message or "debug this", code, [d.get("text", "") for d in docs], language)
        raw = self.llm.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=1024, temperature=0.3,
        )
        if raw.strip() == SAFE_LLM_ERROR:
            return {
                "issue_found": SAFE_LLM_ERROR,
                "why_it_happens": "",
                "corrected_code": "",
                "explanation": "",
                "prevention_tip": "",
                "status": "error",
                "citations": [],
                "unsupported_reason": None,
                "used_doc_ids": used_doc_ids,
            }
        result = self._parse(raw) or {}
        return {
            "issue_found": result.get("answer", ""),
            "why_it_happens": "",
            "corrected_code": result.get("code_example", ""),
            "explanation": result.get("mini_challenge", ""),
            "prevention_tip": result.get("quick_check", ""),
            "status": "answered",
            "citations": self._build_citations(docs),
            "unsupported_reason": None,
            "used_doc_ids": used_doc_ids,
        }

    def generate_interview(
        self,
        user_id: str,
        topic: str,
        difficulty: str = "intermediate",
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        docs = self._retrieve(topic, top_k, doc_id=doc_id)
        used_doc_ids = self._collect_doc_ids(docs, doc_id)
        unsupported = self._unsupported_if_needed(docs, doc_id=doc_id, safety_flags=[])
        if unsupported:
            return {
                "questions": [],
                "status": "unsupported",
                "citations": [],
                "unsupported_reason": unsupported["reason"],
                "used_doc_ids": used_doc_ids,
            }

        prompt = _interview_prompt(topic, [d.get("text", "") for d in docs], difficulty)
        raw = self.llm.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=1024, temperature=0.7,
        )
        if raw.strip() == SAFE_LLM_ERROR:
            return {
                "questions": [],
                "status": "error",
                "citations": [],
                "unsupported_reason": None,
                "used_doc_ids": used_doc_ids,
            }
        result = self._parse(raw) or {}
        return {
            "questions": [
                {
                    "question": result.get("answer", topic),
                    "model_answer": result.get("code_example", ""),
                    "difficulty": difficulty,
                }
            ],
            "status": "answered",
            "citations": self._build_citations(docs),
            "unsupported_reason": None,
            "used_doc_ids": used_doc_ids,
        }


_UNSUPPORTED_MESSAGE = "I don't have enough information in the uploaded documents to answer this reliably."
