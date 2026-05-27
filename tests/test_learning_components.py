"""Unit tests for LearnMate AI components."""

import pytest
from agentic_rag.learning.learner_profile import LearnerProfile
from agentic_rag.learning.teaching_strategy import choose_strategy, get_strategy_instruction
from agentic_rag.learning.learning_memory import LearningMemory
from pathlib import Path
import tempfile
import shutil


# ── LearnerProfile ────────────────────────────────────────────────

class TestLearnerProfile:
    def test_default_values(self):
        p = LearnerProfile(user_id="u1")
        assert p.user_id == "u1"
        assert p.level == "beginner"
        assert p.average_score == 0.0
        assert p.pace == "normal"

    def test_add_score(self):
        p = LearnerProfile(user_id="u1")
        p.add_score(3)
        p.add_score(5)
        assert p.average_score == 4.0

    def test_add_misconception(self):
        p = LearnerProfile(user_id="u1")
        p.add_misconception("wrong formula")
        p.add_misconception("wrong formula")  # duplicate
        assert len(p.misconception_history) == 1

    def test_mark_weak(self):
        p = LearnerProfile(user_id="u1")
        p.mark_weak("calculus")
        assert "calculus" in p.weak_topics

    def test_mark_completed(self):
        p = LearnerProfile(user_id="u1")
        p.set_current_topic("algebra")
        p.mark_completed("algebra")
        assert "algebra" in p.completed_topics
        assert "algebra" not in p.current_topics

    def test_update_pace(self):
        p = LearnerProfile(user_id="u1")
        p.update_pace("slow_down")
        assert p.pace == "slow"
        p.update_pace("increase_difficulty")
        assert p.pace == "fast"

    def test_to_dict(self):
        p = LearnerProfile(user_id="u1")
        d = p.to_dict()
        assert d["user_id"] == "u1"
        assert "average_score" in d


# ── TeachingStrategy ──────────────────────────────────────────────

class TestTeachingStrategy:
    def test_beginner_simple(self):
        s = choose_strategy("beginner", "conceptual", "simple")
        assert s == "beginner_simple"

    def test_exam_goal(self):
        s = choose_strategy("intermediate", "exam", "simple")
        assert s == "exam_focused"

    def test_revision_goal(self):
        s = choose_strategy("advanced", "revision", "simple")
        assert s == "revision_mode"

    def test_analogy_style(self):
        s = choose_strategy("beginner", "conceptual", "analogy")
        assert s == "analogy_first"

    def test_strategy_instruction(self):
        desc = get_strategy_instruction("beginner_simple")
        assert len(desc) > 10

    def test_unknown_strategy(self):
        desc = get_strategy_instruction("nonexistent")
        assert len(desc) > 0  # Falls back to beginner_simple


# ── LearningMemory ────────────────────────────────────────────────

class TestLearningMemory:
    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mem = LearningMemory(storage_dir=self.tmp)

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_new_user(self):
        p = self.mem.get("new_user")
        assert p.user_id == "new_user"
        assert p.level == "beginner"

    def test_save_and_load(self):
        p = self.mem.get("save_user")
        p.add_score(4)
        p.level = "intermediate"
        self.mem.save(p)

        # Clear cache to force file load
        self.mem._cache.clear()
        p2 = self.mem.get("save_user")
        assert p2.level == "intermediate"
        assert p2.scores == [4]

    def test_exists(self):
        assert not self.mem.exists("ghost")
        self.mem.save(LearnerProfile(user_id="ghost"))
        assert self.mem.exists("ghost")


# ── LLM Provider ─────────────────────────────────────────────────

class TestLLMProvider:
    def test_provider_classes_importable(self):
        from agentic_rag.rag.llm_provider import (
            LLMProvider, OllamaProvider, OpenAICompatibleProvider,
            LocalHFProvider, get_llm_provider,
        )
        # All provider classes should be importable
        assert issubclass(OllamaProvider, LLMProvider)
        assert issubclass(OpenAICompatibleProvider, LLMProvider)
        assert issubclass(LocalHFProvider, LLMProvider)
        assert callable(get_llm_provider)

    def test_app_llm_provider_is_set(self):
        """The app module should have initialised a valid LLM provider."""
        from agentic_rag import app as app_module
        from agentic_rag.rag.llm_provider import LLMProvider
        assert isinstance(app_module._services.llm_provider(), LLMProvider)
