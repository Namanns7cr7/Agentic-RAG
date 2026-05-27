"""Tests for the rule-based query classifier."""

import pytest
from agentic_rag.learning.query_classifier import classify


class TestQueryClassifier:
    def test_explain_classification(self):
        assert classify("explain dependency injection") == "explain"
        assert classify("what is a decorator?") == "explain"
        assert classify("how does async work?") == "explain"

    def test_code_example_classification(self):
        assert classify("give me a code example of list comprehension") == "code_example"
        assert classify("show me how to use FastAPI Depends") == "code_example"
        assert classify("sample code for async function") == "code_example"

    def test_quiz_classification(self):
        assert classify("quiz me on Python decorators") == "quiz"
        assert classify("test me on async programming") == "quiz"
        assert classify("ask me questions about FastAPI") == "quiz"

    def test_challenge_classification(self):
        assert classify("give me a practice task on generators") == "challenge"
        assert classify("coding exercise on asyncio") == "challenge"

    def test_cheat_sheet_classification(self):
        assert classify("make a cheat sheet for FastAPI") == "cheat_sheet"
        assert classify("quick reference for pydantic") == "cheat_sheet"

    def test_debug_classification(self):
        assert classify("I have a TypeError: NoneType in my code") == "debug"
        assert classify("how do I fix this KeyError?") == "debug"
        assert classify("my code is not working, traceback included") == "debug"

    def test_interview_classification(self):
        assert classify("give me interview questions on microservices") == "interview"
        assert classify("technical interview prep for FastAPI") == "interview"

    def test_build_with_me_classification(self):
        assert classify("build with me a REST API step by step") == "build_with_me"
        assert classify("walk through creating a FastAPI app") == "build_with_me"

    def test_general_qa_fallback(self):
        # These should NOT match any specific pattern and fall to general_qa
        assert classify("best practices for logging") == "general_qa"
        assert classify("project structure recommendations") == "general_qa"
