"""Tests for the DocuMentor AI /docs/* endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from agentic_rag.app import app
    yield TestClient(app)


class TestDocsAskEndpoint:
    def test_basic_ask(self, client):
        r = client.post("/docs/ask", json={
            "user_id": "test_dev",
            "question": "explain FastAPI dependency injection",
        })
        assert r.status_code == 200
        data = r.json()
        assert "mode" in data
        assert "detected_level" in data
        assert "answer" in data
        assert data["status"] in ("answered", "unsupported", "error")
        assert "citations" in data
        assert "used_doc_ids" in data
        assert "code_example" in data
        assert "next_step" in data

    def test_explicit_mode(self, client):
        r = client.post("/docs/ask", json={
            "user_id": "test_dev",
            "question": "show me a code example",
            "mode": "code_example",
        })
        assert r.status_code == 200

    def test_missing_user_id(self, client):
        r = client.post("/docs/ask", json={
            "user_id": "",
            "question": "explain async",
        })
        assert r.status_code == 422

    def test_missing_question(self, client):
        r = client.post("/docs/ask", json={
            "user_id": "dev1",
            "question": "",
        })
        assert r.status_code == 422


class TestDocsChallengeEndpoint:
    def test_basic_challenge(self, client):
        r = client.post("/docs/challenge", json={
            "user_id": "test_dev",
            "topic": "async functions",
        })
        assert r.status_code == 200
        data = r.json()
        assert "challenge_title" in data
        assert "difficulty" in data
        assert "task" in data
        assert "status" in data
        assert "citations" in data


class TestDocsQuizEndpoint:
    def test_basic_quiz(self, client):
        r = client.post("/docs/quiz", json={
            "user_id": "test_dev",
            "topic": "FastAPI routing",
        })
        assert r.status_code == 200
        data = r.json()
        assert "questions" in data
        assert isinstance(data["questions"], list)
        assert "status" in data


class TestDocsCheatsheetEndpoint:
    def test_basic_cheatsheet(self, client):
        r = client.post("/docs/cheatsheet", json={
            "user_id": "test_dev",
            "topic": "pydantic models",
        })
        assert r.status_code == 200
        data = r.json()
        assert "title" in data
        assert "syntax" in data
        assert "status" in data

    def test_cheatsheet_no_topic(self, client):
        r = client.post("/docs/cheatsheet", json={
            "user_id": "test_dev",
        })
        assert r.status_code == 200


class TestDocsDebugEndpoint:
    def test_basic_debug(self, client):
        r = client.post("/docs/debug", json={
            "user_id": "test_dev",
            "code": "def foo():\n    return bar",
            "error_message": "NameError: name 'bar' is not defined",
        })
        assert r.status_code == 200
        data = r.json()
        assert "issue_found" in data
        assert "corrected_code" in data
        assert "status" in data


class TestDocsInterviewEndpoint:
    def test_basic_interview(self, client):
        r = client.post("/docs/interview", json={
            "user_id": "test_dev",
            "topic": "REST API design",
        })
        assert r.status_code == 200
        data = r.json()
        assert "questions" in data
        assert isinstance(data["questions"], list)
        assert "status" in data
