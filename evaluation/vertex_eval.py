"""Vertex AI evaluation runner for LearnMate AI.

Runs the same grounding evaluation cases from ``eval_queries.json`` but
against the live Vertex AI Gemini endpoint so you can compare quality
vs. mock / local results.

Usage:
    # 1. Authenticate
    gcloud auth application-default login

    # 2. Set environment variables
    export GOOGLE_CLOUD_PROJECT=your-project-id
    export GOOGLE_CLOUD_LOCATION=us-central1          # optional, defaults to us-central1
    export VERTEX_GEMINI_MODEL=gemini-2.5-flash        # optional
    export VERTEX_EMBEDDING_MODEL=text-embedding-004  # optional

    # 3. Run
    python evaluation/vertex_eval.py

Output: evaluation/vertex_results.json
"""

import json
import os
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _check_prerequisites() -> bool:
    """Return True only if all required env vars and SDK are available."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        print(
            "Vertex evaluation skipped: GOOGLE_CLOUD_PROJECT is not set.\n"
            "Set it with: export GOOGLE_CLOUD_PROJECT=your-project-id"
        )
        return False

    try:
        import vertexai  # type: ignore  # noqa: F401
    except ImportError:
        print(
            "Vertex evaluation skipped: google-cloud-aiplatform is not installed.\n"
            "Run: pip install 'google-cloud-aiplatform>=1.60.0'"
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _contains_keywords(text: str, keywords: list) -> bool:
    lowered = text.lower()
    return all(k.lower() in lowered for k in keywords)


def _contains_forbidden(text: str, keywords: list) -> bool:
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _check_prerequisites():
        return

    project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    llm_model = os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.5-flash")
    emb_model = os.getenv("VERTEX_EMBEDDING_MODEL", "text-embedding-004")

    print(f"Vertex AI Evaluation")
    print(f"  Project  : {project}")
    print(f"  Location : {location}")
    print(f"  LLM      : {llm_model}")
    print(f"  Embeddings: {emb_model}")
    print()

    # Activate Vertex AI providers for the evaluation run
    os.environ["RAG_PROVIDER"] = "vertex"
    os.environ["EMBEDDING_PROVIDER"] = "vertex"
    os.environ["GOOGLE_CLOUD_PROJECT"] = project
    os.environ["GOOGLE_CLOUD_LOCATION"] = location
    os.environ["VERTEX_GEMINI_MODEL"] = llm_model
    os.environ["VERTEX_EMBEDDING_MODEL"] = emb_model

    # Import app *after* setting env vars so provider selection is correct
    from fastapi.testclient import TestClient  # type: ignore
    from agentic_rag.app import app  # type: ignore

    client = TestClient(app)

    data_path = Path(__file__).with_name("eval_queries.json")
    if not data_path.exists():
        print(f"eval_queries.json not found at {data_path}")
        return

    data = json.loads(data_path.read_text(encoding="utf-8"))

    # ── Upload documents ──────────────────────────────────────────────
    doc_ids: dict = {}
    for doc in data.get("documents", []):
        files = {"file": (f"{doc['key']}.txt", doc["text"].encode("utf-8"), "text/plain")}
        res = client.post("/upload-document", files=files)
        if res.status_code == 200:
            doc_ids[doc["key"]] = res.json()["doc_id"]
            print(f"  Uploaded: {doc['key']} → {doc_ids[doc['key']][:8]}…")
        else:
            print(f"  Upload failed for {doc['key']}: {res.text}")

    print()

    # ── Run cases ────────────────────────────────────────────────────
    results = []
    timings = []
    forbidden_failures = 0
    citation_pass = 0
    doc_scope_pass = 0
    quiz_valid_pass = 0

    cases = data.get("cases", [])
    total = len(cases)
    answered = 0
    unsupported = 0
    status_correct = 0

    for i, case in enumerate(cases, 1):
        endpoint = case.get("endpoint", "ask")
        doc_key = case.get("doc_key")
        doc_id = doc_ids.get(doc_key, "missing")
        expected_status = case.get("expected_status")

        start = time.perf_counter()

        if endpoint == "quiz":
            res = client.post(
                "/docs/quiz",
                json={
                    "user_id": "vertex_eval",
                    "topic": case["query"],
                    "doc_id": doc_id if doc_key in doc_ids else "missing",
                    "num_questions": case.get("num_questions", 3),
                },
            )
        else:
            res = client.post(
                "/docs/ask",
                json={
                    "user_id": "vertex_eval",
                    "question": case["query"],
                    "doc_id": doc_id if doc_key in doc_ids else "missing",
                },
            )

        elapsed = (time.perf_counter() - start) * 1000
        timings.append(elapsed)

        if res.status_code != 200:
            print(f"  [{i}/{total}] CASE {case['id']} → HTTP {res.status_code}")
            results.append({"id": case["id"], "status": "error", "latency_ms": round(elapsed, 2)})
            continue

        payload = res.json()
        status = payload.get("status", "error")
        answer_text = payload.get("answer", "") or ""

        if status == "answered":
            answered += 1
        if status == "unsupported":
            unsupported += 1
        if status == expected_status:
            status_correct += 1

        # Keywords
        expected_keywords = case.get("expected_keywords", [])
        forbidden_keywords = case.get("forbidden_keywords", [])
        contains_expected = _contains_keywords(answer_text, expected_keywords) if expected_keywords else True
        contains_forbidden = _contains_forbidden(answer_text, forbidden_keywords) if forbidden_keywords else False
        if contains_forbidden:
            forbidden_failures += 1

        # Citations
        citations = payload.get("citations", [])
        expected_doc_key = case.get("expected_citation_doc_key")
        expected_doc_id = doc_ids.get(expected_doc_key)
        if citations and expected_doc_id:
            if any(c.get("doc_id") == expected_doc_id for c in citations):
                citation_pass += 1

        # Doc scope
        if case.get("category") == "doc_scope" and expected_doc_id:
            if status == "answered" and any(c.get("doc_id") == expected_doc_id for c in citations):
                doc_scope_pass += 1

        # Quiz validity
        quiz_valid = True
        if endpoint == "quiz" and status == "answered":
            questions = payload.get("questions", [])
            if len(questions) != case.get("num_questions", 3):
                quiz_valid = False
            for q in questions:
                if len(q.get("options", [])) != 4:
                    quiz_valid = False
                if len(set(q.get("options", []))) != 4:
                    quiz_valid = False
                if q.get("correct_answer") not in q.get("options", []):
                    quiz_valid = False
                if not q.get("explanation"):
                    quiz_valid = False
            if quiz_valid:
                quiz_valid_pass += 1

        mark = "✓" if status == expected_status else "✗"
        print(f"  [{i}/{total}] {mark} CASE {case['id']} → {status} ({elapsed:.0f}ms)")

        results.append({
            "id": case["id"],
            "status": status,
            "expected_status": expected_status,
            "contains_expected_keywords": contains_expected,
            "contains_forbidden_keywords": contains_forbidden,
            "latency_ms": round(elapsed, 2),
        })

    # ── Summary ───────────────────────────────────────────────────────
    status_accuracy = status_correct / total if total else 0.0
    citation_accuracy = citation_pass / max(1, total)
    average_latency = sum(timings) / max(1, len(timings))
    safety_pass_rate = 1.0 - (forbidden_failures / max(1, total))
    doc_scope_cases = [c for c in cases if c.get("category") == "doc_scope"]
    doc_scope_pass_rate = doc_scope_pass / max(1, len(doc_scope_cases))
    quiz_cases = [c for c in cases if c.get("endpoint") == "quiz"]
    quiz_validity_rate = quiz_valid_pass / max(1, len(quiz_cases))

    summary = {
        "provider": "vertex",
        "llm_model": llm_model,
        "embedding_model": emb_model,
        "project": project,
        "location": location,
        "total_cases": total,
        "answered_cases": answered,
        "unsupported_cases": unsupported,
        "status_accuracy": round(status_accuracy, 3),
        "citation_accuracy": round(citation_accuracy, 3),
        "forbidden_keyword_failures": forbidden_failures,
        "average_latency_ms": round(average_latency, 2),
        "safety_pass_rate": round(safety_pass_rate, 3),
        "doc_scope_pass_rate": round(doc_scope_pass_rate, 3),
        "quiz_validity_rate": round(quiz_validity_rate, 3),
        "cases": results,
    }

    output_path = Path(__file__).with_name("vertex_results.json")
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("── Vertex AI Evaluation Summary ──────────────────────────")
    print(json.dumps({k: summary[k] for k in summary if k != "cases"}, indent=2))
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
