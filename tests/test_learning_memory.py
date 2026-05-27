"""Tests for LearningMemory — profile CRUD without LLM."""

import pytest
from agentic_rag.learning.learning_memory import LearningMemory
from agentic_rag.learning.learner_profile import LearnerProfile


@pytest.fixture()
def memory(tmp_path):
    """Fresh in-memory LearningMemory backed by a temp directory."""
    return LearningMemory(storage_dir=tmp_path)


class TestLearningMemory:
    def test_create_new_profile(self, memory):
        profile = memory.get("new_user_abc")
        assert profile.user_id == "new_user_abc"
        assert profile.level == "beginner"
        assert profile.average_score == 0.0

    def test_profile_persists_after_save(self, memory, tmp_path):
        profile = memory.get("persist_user")
        profile.add_score(4)
        profile.mark_weak("asyncio")
        memory.save(profile)

        # Re-instantiate memory from same dir to simulate server restart
        memory2 = LearningMemory(storage_dir=tmp_path)
        loaded = memory2.get("persist_user")
        assert 4 in loaded.scores
        assert "asyncio" in loaded.weak_topics

    def test_update_weak_topic(self, memory):
        profile = memory.get("weak_topic_user")
        profile.mark_weak("decorators")
        profile.mark_weak("generators")
        assert "decorators" in profile.weak_topics
        assert "generators" in profile.weak_topics

    def test_no_duplicate_weak_topics(self, memory):
        profile = memory.get("dup_user")
        profile.mark_weak("async")
        profile.mark_weak("async")
        assert profile.weak_topics.count("async") == 1

    def test_quiz_score_update(self, memory):
        profile = memory.get("score_user")
        profile.add_score(3)
        profile.add_score(5)
        assert profile.average_score == 4.0

    def test_pace_update(self, memory):
        profile = memory.get("pace_user")
        profile.update_pace("slow_down")
        assert profile.pace == "slow"
        profile.update_pace("increase_difficulty")
        assert profile.pace == "fast"

    def test_exists_check(self, memory):
        assert not memory.exists("non_existent_user_xyz")
        _ = memory.get("exists_user")
        assert memory.exists("exists_user")
