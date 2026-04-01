"""API endpoint tests — smoke tests and edge cases."""

import pytest
from fastapi.testclient import TestClient
from agentic_rag.app import app

client = TestClient(app)


# ── Health & Home ─────────────────────────────────────────────────

def test_home_endpoint():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["documents_loaded"] > 0
    assert "model" in data


# ── Query ─────────────────────────────────────────────────────────

def test_query_basic():
    r = client.post("/query", json={"question": "What is RAG?", "top_k": 3})
    assert r.status_code == 200
    payload = r.json()
    assert "plan" in payload
    assert "draft" in payload
    assert "final" in payload


def test_query_with_top_k():
    r = client.post("/query", json={"question": "What are transformers?", "top_k": 2})
    assert r.status_code == 200
    assert "final" in r.json()


def test_query_default_top_k():
    r = client.post("/query", json={"question": "What is FAISS?"})
    assert r.status_code == 200
    assert "final" in r.json()


def test_query_empty_rejected():
    r = client.post("/query", json={"question": ""})
    assert r.status_code == 422  # Pydantic validation error


def test_query_too_short_rejected():
    r = client.post("/query", json={"question": "x"})
    assert r.status_code == 422


def test_query_invalid_top_k():
    r = client.post("/query", json={"question": "test query", "top_k": 0})
    assert r.status_code == 422


def test_query_top_k_too_large():
    r = client.post("/query", json={"question": "test query", "top_k": 100})
    assert r.status_code == 422


# ── Load ──────────────────────────────────────────────────────────

def test_load_single_doc():
    r = client.post("/load", json={"documents": ["A new fact appears."]})
    assert r.status_code == 200
    data = r.json()
    assert data["added"] == 1
    assert data["total"] > 0


def test_load_multiple_docs():
    r = client.post("/load", json={"documents": ["Fact one.", "Fact two.", "Fact three."]})
    assert r.status_code == 200
    assert r.json()["added"] == 3


def test_load_empty_list():
    r = client.post("/load", json={"documents": []})
    # Pydantic min_length=1 rejects empty list
    assert r.status_code == 422


# ── Memory ────────────────────────────────────────────────────────

def test_clear_memory():
    # First ask a question to populate memory
    client.post("/query", json={"question": "What is Python?"})
    # Then clear it
    r = client.post("/memory/clear")
    assert r.status_code == 200
    assert r.json()["status"] == "memory cleared"
