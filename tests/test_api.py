import pytest
from fastapi.testclient import TestClient
from agentic_rag.app import app

client = TestClient(app)

def test_query_endpoint():
    r = client.post('/query', json={'question':'What is RAG?','top_k':2})
    assert r.status_code == 200
    payload = r.json()
    assert 'final' in payload

def test_load_endpoint():
    r = client.post('/load', json={'documents':['A new fact appears.']})
    assert r.status_code == 200
    assert r.json().get('added') == 1
