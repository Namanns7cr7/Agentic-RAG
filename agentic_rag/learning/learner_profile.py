"""Learner profile model & helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class LearnerProfile:
    """In-memory representation of a learner's state."""

    user_id: str
    preferred_style: str = "simple"
    level: str = "beginner"
    current_topics: List[str] = field(default_factory=list)
    completed_topics: List[str] = field(default_factory=list)
    weak_topics: List[str] = field(default_factory=list)
    misconception_history: List[str] = field(default_factory=list)
    scores: List[int] = field(default_factory=list)
    pace: str = "normal"
    last_interaction_at: Optional[str] = None

    # ── Derived properties ────────────────────────────────────────

    @property
    def average_score(self) -> float:
        return round(sum(self.scores) / len(self.scores), 2) if self.scores else 0.0

    # ── Mutation helpers ──────────────────────────────────────────

    def touch(self) -> None:
        self.last_interaction_at = datetime.utcnow().isoformat()

    def add_score(self, score: int) -> None:
        self.scores.append(score)

    def add_misconception(self, misconception: str) -> None:
        if misconception and misconception not in self.misconception_history:
            self.misconception_history.append(misconception)

    def mark_weak(self, topic: str) -> None:
        if topic and topic not in self.weak_topics:
            self.weak_topics.append(topic)

    def mark_completed(self, topic: str) -> None:
        if topic and topic not in self.completed_topics:
            self.completed_topics.append(topic)
        if topic in self.current_topics:
            self.current_topics.remove(topic)

    def set_current_topic(self, topic: str) -> None:
        if topic and topic not in self.current_topics:
            self.current_topics.append(topic)

    def update_pace(self, decision: str) -> None:
        mapping = {
            "slow_down": "slow",
            "continue": "normal",
            "increase_difficulty": "fast",
        }
        self.pace = mapping.get(decision, self.pace)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "preferred_style": self.preferred_style,
            "level": self.level,
            "current_topics": self.current_topics,
            "completed_topics": self.completed_topics,
            "weak_topics": self.weak_topics,
            "misconception_history": self.misconception_history,
            "average_score": self.average_score,
            "pace": self.pace,
            "last_interaction_at": self.last_interaction_at,
        }
