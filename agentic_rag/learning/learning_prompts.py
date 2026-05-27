"""Prompt templates for the LearnMate AI learning pipeline."""

from __future__ import annotations

from typing import List, Optional

# ---------------------------------------------------------------------------
# System prompt – used as the ``system`` message in every learning call
# ---------------------------------------------------------------------------

LEARNMATE_SYSTEM_PROMPT = """\
You are LearnMate AI, an adaptive personalized learning assistant.
Your job is not just to answer, but to teach.

You must:
- Identify the learner's level.
- Explain concepts step by step.
- Use simple examples when needed.
- Ask short check questions.
- Adapt pace based on user answers.
- Avoid hallucinations.
- Use retrieved context when available.
- Clearly separate facts, examples, and assumptions.
- Encourage learning without being verbose unnecessarily.
"""

# ---------------------------------------------------------------------------
# Concept explanation prompt
# ---------------------------------------------------------------------------

def concept_explanation_prompt(
    topic: str,
    level: str,
    goal: str,
    strategy_instruction: str,
    retrieved_context: Optional[List[str]] = None,
    preferred_style: str = "simple",
    learner_weak_topics: Optional[List[str]] = None,
) -> str:
    ctx = ""
    if retrieved_context:
        ctx = "Retrieved context (use this to ground your explanation):\n"
        ctx += "\n".join(f"- {c}" for c in retrieved_context)
        ctx += "\n\n"

    weak = ""
    if learner_weak_topics:
        weak = f"The learner has previously struggled with: {', '.join(learner_weak_topics)}.\n"

    return f"""{ctx}{weak}Teaching strategy: {strategy_instruction}

Topic: {topic}
Learner level: {level}
Learning goal: {goal}
Preferred style: {preferred_style}

Produce a JSON object with exactly these keys (no markdown fencing):
{{
  "explanation": "<clear, step-by-step explanation>",
  "analogy": "<a relatable analogy>",
  "real_life_example": "<a concrete real-world example>",
  "key_points": ["point 1", "point 2", "..."],
  "quick_check_question": "<a short question to test understanding>",
  "next_step": "<what the learner should study next>"
}}
"""

# ---------------------------------------------------------------------------
# Reflection / self-check prompt
# ---------------------------------------------------------------------------

REFLECTION_PROMPT_TEMPLATE = """\
Review the following draft teaching response.

Draft:
---
{draft}
---

Retrieved context (use this to verify grounding):
{context}

Check:
1. Is it factually correct?
2. Is it appropriate for a {level} learner?
3. Is it too hard or too easy?
4. Are claims supported by retrieved context when retrieval was used?
5. Does it include a useful quick-check question?
6. Should the pace be slowed down or increased?

Return ONLY the improved JSON response (same keys), with corrections applied.
Do NOT wrap in markdown fences.
"""

def reflection_prompt(draft: str, level: str, context: str) -> str:
  return REFLECTION_PROMPT_TEMPLATE.format(draft=draft, level=level, context=context)


# ---------------------------------------------------------------------------
# Answer evaluation prompt
# ---------------------------------------------------------------------------

EVALUATION_PROMPT_TEMPLATE = """\
Evaluate the learner's answer. Do not be harsh.

Topic: {topic}
Question: {question}
Learner's answer: {answer}

Identify:
- What is correct
- What is missing
- Any misconception
- Score out of 5
- Next teaching move

Return a JSON object with exactly these keys (no markdown fencing):
{{
  "score": <0-5>,
  "correctness": "incorrect" | "partially_correct" | "correct",
  "feedback": "<constructive feedback>",
  "missing_points": ["..."],
  "misconception": "<detected misconception or empty string>",
  "pace_decision": "slow_down" | "continue" | "increase_difficulty",
  "next_explanation_style": "<recommended style>",
  "next_question": "<follow-up question>"
}}
"""

def evaluation_prompt(topic: str, question: str, answer: str) -> str:
    return EVALUATION_PROMPT_TEMPLATE.format(topic=topic, question=question, answer=answer)


# ---------------------------------------------------------------------------
# Flashcard generation prompt
# ---------------------------------------------------------------------------

FLASHCARD_PROMPT_TEMPLATE = """\
Generate {count} flashcards for the topic "{topic}" at a {level} level.

Return a JSON object:
{{
  "flashcards": [
    {{"front": "question or term", "back": "answer or definition"}},
    ...
  ]
}}
No markdown fencing.
"""

def flashcard_prompt(topic: str, count: int, level: str) -> str:
    return FLASHCARD_PROMPT_TEMPLATE.format(topic=topic, count=count, level=level)


# ---------------------------------------------------------------------------
# Learning path generation prompt
# ---------------------------------------------------------------------------

LEARNING_PATH_PROMPT_TEMPLATE = """\
Create a structured learning path for the topic "{topic}".
The learner's goal is: {goal}.

Return a JSON object:
{{
  "learning_path": [
    {{
      "step": 1,
      "subtopic": "...",
      "why_it_matters": "...",
      "estimated_difficulty": "easy | medium | hard"
    }},
    ...
  ]
}}
Include 5-10 steps. No markdown fencing.
"""

def learning_path_prompt(topic: str, goal: str) -> str:
    return LEARNING_PATH_PROMPT_TEMPLATE.format(topic=topic, goal=goal)


# ---------------------------------------------------------------------------
# Teach-from-docs prompt
# ---------------------------------------------------------------------------

TEACH_FROM_DOCS_TEMPLATE = """\
Use the following retrieved documents to teach the learner about their query.

Retrieved documents:
{documents}

Learner's query: {query}
Learner's level: {level}

Produce a JSON object with exactly these keys (no markdown fencing):
{{
  "explanation": "<clear explanation grounded in retrieved docs>",
  "analogy": "<a relatable analogy>",
  "real_life_example": "<a concrete real-world example>",
  "key_points": ["point 1", "point 2", "..."],
  "quick_check_question": "<a short question>",
  "next_step": "<what to study next>"
}}
"""

def teach_from_docs_prompt(documents: List[str], query: str, level: str) -> str:
    docs = "\n".join(f"- {d}" for d in documents) if documents else "No documents found."
    return TEACH_FROM_DOCS_TEMPLATE.format(documents=docs, query=query, level=level)


# ---------------------------------------------------------------------------
# Misconception detection prompt
# ---------------------------------------------------------------------------

MISCONCEPTION_PROMPT_TEMPLATE = """\
Analyse the learner's answer for misconceptions.

Topic: {topic}
Question: {question}
Answer: {answer}

Common misconception categories:
- Confused definition
- Wrong formula
- Incomplete reasoning
- Memorised but not understood
- Calculation mistake

Return a short string describing the misconception, or "none" if there is none.
"""

def misconception_prompt(topic: str, question: str, answer: str) -> str:
    return MISCONCEPTION_PROMPT_TEMPLATE.format(topic=topic, question=question, answer=answer)
