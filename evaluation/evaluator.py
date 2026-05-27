import json
import os
import time
from pathlib import Path

os.environ.setdefault("RAG_MOCK_MODE", "true")

from fastapi.testclient import TestClient

from agentic_rag.app import app


def _upload_doc(client: TestClient, text: str, filename: str) -> str:
    files = {"file": (filename, text.encode("utf-8"), "text/plain")}
    res = client.post("/upload-document", files=files)
    res.raise_for_status()
    return res.json()["doc_id"]


def _contains_keywords(text: str, keywords) -> bool:
    lowered = text.lower()
    return all(k.lower() in lowered for k in keywords)


def _contains_forbidden(text: str, keywords) -> bool:
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)


def main() -> None:
    data = json.loads(Path(__file__).with_name("eval_queries.json").read_text(encoding="utf-8"))
    client = TestClient(app)

    doc_ids = {}
    for doc in data.get("documents", []):
        doc_ids[doc["key"]] = _upload_doc(client, doc["text"], f"{doc['key']}.txt")

    results = []
    timings = []
    forbidden_failures = 0
    citation_pass = 0
    doc_scope_pass = 0
    quiz_valid_pass = 0

    total = len(data.get("cases", []))
    answered = 0
    unsupported = 0
    status_correct = 0

    for case in data.get("cases", []):
        endpoint = case.get("endpoint", "ask")
        doc_key = case.get("doc_key")
        doc_id = doc_ids.get(doc_key, "missing")
        expected_status = case.get("expected_status")

        prev_min = os.getenv("MIN_RETRIEVAL_SCORE")
        if "min_score_override" in case:
            os.environ["MIN_RETRIEVAL_SCORE"] = str(case["min_score_override"])

        start = time.perf_counter()
        if endpoint == "quiz":
            res = client.post(
                "/docs/quiz",
                json={
                    "user_id": "eval_user",
                    "topic": case["query"],
                    "doc_id": doc_id if doc_key in doc_ids else "missing",
                    "num_questions": case.get("num_questions", 3),
                },
            )
        else:
            res = client.post(
                "/docs/ask",
                json={
                    "user_id": "eval_user",
                    "question": case["query"],
                    "doc_id": doc_id if doc_key in doc_ids else "missing",
                },
            )
        elapsed = (time.perf_counter() - start) * 1000
        timings.append(elapsed)

        if prev_min is None:
            os.environ.pop("MIN_RETRIEVAL_SCORE", None)
        else:
            os.environ["MIN_RETRIEVAL_SCORE"] = prev_min

        payload = res.json()
        status = payload.get("status", "error")
        answer_text = payload.get("answer", "") or ""

        if status == "answered":
            answered += 1
        if status == "unsupported":
            unsupported += 1
        if status == expected_status:
            status_correct += 1

        expected_keywords = case.get("expected_keywords", [])
        forbidden_keywords = case.get("forbidden_keywords", [])
        contains_expected = _contains_keywords(answer_text, expected_keywords) if expected_keywords else True
        contains_forbidden = _contains_forbidden(answer_text, forbidden_keywords) if forbidden_keywords else False
        if contains_forbidden:
            forbidden_failures += 1

        citations = payload.get("citations", [])
        expected_doc_key = case.get("expected_citation_doc_key")
        expected_doc_id = doc_ids.get(expected_doc_key, None)
        if citations and expected_doc_id:
            if any(c.get("doc_id") == expected_doc_id for c in citations):
                citation_pass += 1

        if case.get("category") == "doc_scope" and expected_doc_id:
            if status == "answered" and any(c.get("doc_id") == expected_doc_id for c in citations):
                doc_scope_pass += 1

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

        results.append(
            {
                "id": case["id"],
                "status": status,
                "expected_status": expected_status,
                "contains_expected_keywords": contains_expected,
                "contains_forbidden_keywords": contains_forbidden,
                "latency_ms": round(elapsed, 2),
            }
        )

    status_accuracy = status_correct / total if total else 0.0
    citation_accuracy = citation_pass / max(1, total)
    average_latency = sum(timings) / max(1, len(timings))
    safety_pass_rate = 1 - (forbidden_failures / max(1, total))
    doc_scope_pass_rate = doc_scope_pass / max(1, len([c for c in data["cases"] if c.get("category") == "doc_scope"]))
    quiz_cases = len([c for c in data["cases"] if c.get("endpoint") == "quiz"])
    quiz_validity_rate = quiz_valid_pass / max(1, quiz_cases)

    summary = {
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

    output_path = Path(__file__).with_name("results.json")
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Evaluation summary")
    print(json.dumps({k: summary[k] for k in summary if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
