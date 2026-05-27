"""File-based learner memory store.

Stores and retrieves LearnerProfile objects using JSON files.
Interface is kept clean for future replacement with SQLite/Postgres.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from ..rag.utils_logger import get_logger
from .learner_profile import LearnerProfile

logger = get_logger(__name__)

_DEFAULT_DIR = Path(__file__).parent.parent / "data" / "learner_profiles"


class LearningMemory:
    """Manages per-user learner profiles with JSON file persistence."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or _DEFAULT_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, LearnerProfile] = {}

    def _path(self, user_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
        return self.storage_dir / f"{safe}.json"

    def get(self, user_id: str) -> LearnerProfile:
        if user_id in self._cache:
            return self._cache[user_id]
        path = self._path(user_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                profile = LearnerProfile(
                    user_id=data.get("user_id", user_id),
                    preferred_style=data.get("preferred_style", "simple"),
                    level=data.get("level", "beginner"),
                    current_topics=data.get("current_topics", []),
                    completed_topics=data.get("completed_topics", []),
                    weak_topics=data.get("weak_topics", []),
                    misconception_history=data.get("misconception_history", []),
                    scores=data.get("scores", []),
                    pace=data.get("pace", "normal"),
                    last_interaction_at=data.get("last_interaction_at"),
                )
                self._cache[user_id] = profile
                return profile
            except Exception as exc:
                logger.warning("Failed to load profile for %s: %s", user_id, exc)
        profile = LearnerProfile(user_id=user_id)
        self._cache[user_id] = profile
        return profile

    def save(self, profile: LearnerProfile) -> None:
        self._cache[profile.user_id] = profile
        path = self._path(profile.user_id)
        try:
            data = {
                "user_id": profile.user_id,
                "preferred_style": profile.preferred_style,
                "level": profile.level,
                "current_topics": profile.current_topics,
                "completed_topics": profile.completed_topics,
                "weak_topics": profile.weak_topics,
                "misconception_history": profile.misconception_history,
                "scores": profile.scores,
                "pace": profile.pace,
                "last_interaction_at": profile.last_interaction_at,
            }
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to save profile for %s: %s", profile.user_id, exc)

    def exists(self, user_id: str) -> bool:
        return user_id in self._cache or self._path(user_id).exists()
