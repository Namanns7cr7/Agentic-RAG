"""Teaching strategy selector.

Decides *how* to teach based on learner profile, goal, and topic.

Supported strategies:
  beginner_simple | analogy_first | example_first | exam_focused
  quiz_first      | revision_mode | deep_dive
"""

from __future__ import annotations

from typing import Optional

from .learner_profile import LearnerProfile


# Strategy → short description (used in prompts)
STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "beginner_simple":
        "Use very simple language, short sentences, no jargon. "
        "Start from the absolute basics.",
    "analogy_first":
        "Begin with a relatable analogy or metaphor, then link it to the formal concept.",
    "example_first":
        "Lead with a concrete, worked-out example before explaining the theory.",
    "exam_focused":
        "Focus on exam-relevant points: definitions, formulas, common question patterns, "
        "and marking-scheme tips.",
    "quiz_first":
        "Pose a warm-up question first to gauge understanding, then teach based on the response.",
    "revision_mode":
        "Provide a concise summary with key points, formulas, and a quick self-test.",
    "deep_dive":
        "Go in-depth: cover edge cases, proofs, derivations, and advanced intuition.",
}


def choose_strategy(
    level: str,
    goal: str,
    preferred_style: str,
    profile: Optional[LearnerProfile] = None,
) -> str:
    """Return the best strategy name given context."""

    # 1. Explicit style overrides
    style_map: dict[str, str] = {
        "analogy": "analogy_first",
        "example-first": "example_first",
        "exam": "exam_focused",
        "formal": "deep_dive",
    }
    if preferred_style in style_map:
        return style_map[preferred_style]

    # 2. Goal-driven
    if goal == "exam":
        return "exam_focused"
    if goal == "revision":
        return "revision_mode"
    if goal == "interview":
        return "example_first"

    # 3. Level-driven
    if level in ("beginner", "auto"):
        # If we have profile data indicating the learner is struggling, simplify
        if profile and profile.average_score < 2.5 and len(profile.scores) >= 2:
            return "beginner_simple"
        return "beginner_simple" if level == "beginner" else "analogy_first"

    if level == "intermediate":
        return "example_first"

    if level in ("advanced", "exam-focused"):
        return "deep_dive" if level == "advanced" else "exam_focused"

    return "beginner_simple"


def get_strategy_instruction(strategy: str) -> str:
    """Return the prompt instruction for a given strategy."""
    return STRATEGY_DESCRIPTIONS.get(strategy, STRATEGY_DESCRIPTIONS["beginner_simple"])
