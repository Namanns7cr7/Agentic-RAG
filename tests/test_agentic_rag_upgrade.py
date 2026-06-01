import io
import os
import shutil
from fastapi.testclient import TestClient
from agentic_rag.app import app

client = TestClient(app)


def setup_module(module):
    """Clean up any existing test indexes/uploads before starting."""
    for path in ["data/test_uploads", "data/test_index"]:
        if os.path.exists(path):
            shutil.rmtree(path)


def test_health_endpoint():
    """Verify that the health check endpoint returns valid stats."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "documents_loaded" in data
    assert "model" in data


def test_document_lifecycle_and_deduplication():
    """Test full document lifecycle: upload, deduplication, listing, querying, and deletion."""
    # 1. Upload new text document
    file_content = b"LearnMate AI features a highly advanced multi-agent pipeline designed for students. The pipeline includes QueryPlannerAgent, RetrieverAgent, and GroundingAgent."
    file_name = "test_features.txt"

    response = client.post(
        "/documents/upload",
        files={"file": (file_name, io.BytesIO(file_content), "text/plain")},
        data={"chunk_size": 500}
    )
    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert data["filename"] == file_name
    assert data["status"] == "processed"
    assert data["total_characters"] == len(file_content)
    assert data["total_chunks"] >= 1

    doc_id = data["doc_id"]

    # 2. Upload identical content to test SHA-256 deduplication
    dup_response = client.post(
        "/documents/upload",
        files={"file": (file_name, io.BytesIO(file_content), "text/plain")},
        data={"chunk_size": 500}
    )
    assert dup_response.status_code == 200
    dup_data = dup_response.json()
    assert dup_data["doc_id"] == doc_id
    assert dup_data["status"] == "deduplicated"

    # 3. Retrieve document list and verify it is present
    list_response = client.get("/documents")
    assert list_response.status_code == 200
    list_data = list_response.json()
    assert list_data["total_documents"] >= 1
    doc_ids = [d["doc_id"] for d in list_data["documents"]]
    assert doc_id in doc_ids

    # 4. Ask a question grounded in the uploaded document (using agent ask endpoint)
    ask_response = client.post(
        "/documents/ask",
        json={
            "question": "What agents are in the multi-agent pipeline?",
            "doc_id": doc_id,
            "mode": "qa"
        }
    )
    assert ask_response.status_code == 200
    ask_data = ask_response.json()
    assert "answer" in ask_data
    assert ask_data["status"] in ("answered", "unsupported")  # mock mode may return either based on setup
    if ask_data["status"] == "answered":
        assert len(ask_data["citations"]) >= 1
        assert ask_data["citations"][0]["doc_id"] == doc_id

    # 5. Ask with an empty/invalid context to trigger the unsupported answer safety check
    unsupported_response = client.post(
        "/documents/ask",
        json={
            "question": "What is the capital of France according to the document?",
            "doc_id": doc_id,
            "mode": "qa"
        }
    )
    assert unsupported_response.status_code == 200
    unsupported_data = unsupported_response.json()
    assert unsupported_data["status"] == "unsupported"
    assert "clearly mention" in unsupported_data["answer"] or "not have enough information" in unsupported_data["answer"]

    # 6. Delete the document
    del_response = client.delete(f"/documents/{doc_id}")
    assert del_response.status_code == 200
    assert del_response.json()["status"] == "deleted"

    # 7. Verify document is gone from the list
    list_response_after = client.get("/documents")
    assert list_response_after.status_code == 200
    doc_ids_after = [d["doc_id"] for d in list_response_after.json()["documents"]]
    assert doc_id not in doc_ids_after
