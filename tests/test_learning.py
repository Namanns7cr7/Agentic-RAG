"""Tests for LearnMate AI learning endpoints and components."""

import pytest
from fastapi.testclient import TestClient
from agentic_rag.app import app

client = TestClient(app)


# ── /learn ────────────────────────────────────────────────────────

class TestLearnEndpoint:
    def test_learn_basic(self):
        r = client.post("/learn", json={
            "user_id": "test_user_1",
            "topic": "photosynthesis",
            "level": "beginner",
            "goal": "conceptual",
            "preferred_style": "simple",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["topic"] == "photosynthesis"
        assert "detected_level" in data
        assert "teaching_strategy" in data
        assert "explanation" in data
        assert "key_points" in data

    def test_learn_auto_level(self):
        r = client.post("/learn", json={
            "user_id": "test_user_auto",
            "topic": "gravity",
            "level": "auto",
            "goal": "conceptual",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["detected_level"] in ("beginner", "intermediate", "advanced")

    def test_learn_missing_topic(self):
        r = client.post("/learn", json={"user_id": "u1", "topic": ""})
        assert r.status_code == 422

    def test_learn_missing_user(self):
        r = client.post("/learn", json={"user_id": "", "topic": "math"})
        assert r.status_code == 422


# ── /evaluate ─────────────────────────────────────────────────────

class TestEvaluateEndpoint:
    def test_evaluate_basic(self):
        r = client.post("/evaluate", json={
            "user_id": "test_user_eval",
            "topic": "photosynthesis",
            "question": "What is the role of sunlight in photosynthesis?",
            "answer": "Sunlight provides energy for the plant to make food.",
        })
        assert r.status_code == 200
        data = r.json()
        assert "score" in data
        assert 0 <= data["score"] <= 5
        assert "correctness" in data
        assert "feedback" in data

    def test_evaluate_empty_answer(self):
        r = client.post("/evaluate", json={
            "user_id": "u1",
            "topic": "math",
            "question": "What is 2+2?",
            "answer": "",
        })
        assert r.status_code == 422


# ── /profile ──────────────────────────────────────────────────────

class TestProfileEndpoint:
    def test_get_profile_new_user(self):
        r = client.get("/profile/brand_new_user_xyz")
        assert r.status_code == 200
        data = r.json()
        assert data["user_id"] == "brand_new_user_xyz"
        assert data["level"] == "beginner"
        assert data["average_score"] == 0.0

    def test_get_profile_existing(self):
        # First interact to create profile
        client.post("/learn", json={
            "user_id": "profile_test_user",
            "topic": "algebra",
        })
        r = client.get("/profile/profile_test_user")
        assert r.status_code == 200
        data = r.json()
        assert data["user_id"] == "profile_test_user"
        assert "algebra" in data["current_topics"]


# ── /flashcards ───────────────────────────────────────────────────

class TestFlashcardsEndpoint:
    def test_flashcards_basic(self):
        r = client.post("/flashcards", json={
            "user_id": "test_user_fc",
            "topic": "Newton's Laws",
            "count": 3,
            "level": "beginner",
        })
        assert r.status_code == 200
        data = r.json()
        assert "flashcards" in data
        assert len(data["flashcards"]) >= 1
        for card in data["flashcards"]:
            assert "front" in card
            assert "back" in card

    def test_flashcards_invalid_count(self):
        r = client.post("/flashcards", json={
            "user_id": "u1",
            "topic": "math",
            "count": 0,
        })
        assert r.status_code == 422


# ── /learning-path ────────────────────────────────────────────────

class TestLearningPathEndpoint:
    def test_learning_path_basic(self):
        r = client.post("/learning-path", json={
            "user_id": "test_user_lp",
            "topic": "Newton's Laws",
            "goal": "exam",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["topic"] == "Newton's Laws"
        assert "learning_path" in data
        assert len(data["learning_path"]) >= 1
        step = data["learning_path"][0]
        assert "step" in step
        assert "subtopic" in step

    def test_learning_path_missing_topic(self):
        r = client.post("/learning-path", json={"user_id": "u1", "topic": ""})
        assert r.status_code == 422


# ── /teach-from-docs ──────────────────────────────────────────────

class TestTeachFromDocsEndpoint:
    def test_teach_from_docs_basic(self):
        r = client.post("/teach-from-docs", json={
            "user_id": "test_user_tfd",
            "query": "What is RAG?",
            "level": "beginner",
        })
        assert r.status_code == 200
        data = r.json()
        assert "explanation" in data
        assert "retrieved_sources_used" in data
