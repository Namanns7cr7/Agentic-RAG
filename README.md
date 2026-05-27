# Production-Style Agentic RAG Platform

## What It Does
This project provides a document-grounded AI assistant for developer learning. It supports document upload, retrieval, doc-scoped queries, citations, grounding verification, unsupported-answer fallback, quiz generation, tests, and an evaluation harness.

This project reduces hallucination risk using retrieval grounding, citations, fallback behavior, and grounding verification. It does not claim zero hallucinations.

## Architecture Overview
1. Document ingestion and chunking
2. Chunk metadata with `doc_id`
3. Vector retrieval
4. Doc-scoped retrieval (optional)
5. Generation
6. Grounding verification (deterministic check + reflection)
7. Guardrails for empty or weak retrieval
8. Quiz generation with validation
9. Evaluation harness + CI

Text diagram:
```
User query
	-> Retriever (doc_id scoped)
	-> Top-k chunks + citations
	-> LLM generation
	-> Grounding verification
	-> Answer or unsupported
```

## API Contract (DocuMentor)
The core grounded response contract for document queries:

- `answer`: string
- `status`: one of `answered | unsupported | error`
- `citations`: list of source chunk references
- `confidence`: optional float
- `unsupported_reason`: optional string
- `used_doc_ids`: list of document ids
- `safety_flags`: optional list

Rules:
- If retrieval is empty or weak, `status` is `unsupported`.
- Every `answered` response includes citations.
- If grounding verification fails, the response is `unsupported`.

## Key Endpoints
- `POST /upload-document`
- `POST /docs/ask`
- `POST /docs/quiz`
- `POST /docs/challenge`
- `POST /docs/cheatsheet`
- `POST /docs/debug`
- `POST /docs/interview`

## Testing
Test coverage includes:
- Unit tests
- API tests
- Retrieval tests
- Grounding verifier tests
- Guardrail tests
- Quiz validation tests
- LLM failure tests
- Mock-mode tests

Run tests:
```powershell
python -m pytest -q
```

To run in mock mode (no model downloads):
```powershell
$env:RAG_MOCK_MODE="true"
python -m pytest -q
```

## Vertex AI Setup
This project supports an optional Vertex AI provider for Gemini and Vertex embeddings.

Environment variables:
- `RAG_PROVIDER=vertex`
- `GOOGLE_CLOUD_PROJECT=<your-project>`
- `GOOGLE_CLOUD_LOCATION=us-central1`
- `VERTEX_GEMINI_MODEL=gemini-1.5-flash`
- `VERTEX_EMBEDDING_MODEL=text-embedding-004`

Run with Vertex AI:
```powershell
$env:RAG_PROVIDER="vertex"
$env:GOOGLE_CLOUD_PROJECT="your-project"
$env:GOOGLE_CLOUD_LOCATION="us-central1"
$env:VERTEX_GEMINI_MODEL="gemini-1.5-flash"
$env:VERTEX_EMBEDDING_MODEL="text-embedding-004"
uvicorn agentic_rag.app:app --reload
```

Cost warning: Vertex AI usage incurs cost. Use budgets and alerts.

## Evaluation Harness
The evaluation harness runs a set of doc-grounding cases and writes results to `evaluation/results.json`.

Run:
```powershell
python evaluation/evaluator.py
```

Optional Vertex evaluation (requires credentials):
```powershell
python evaluation/vertex_eval.py
```

Metrics reported:
- status accuracy
- citation accuracy
- forbidden keyword failures
- safety pass rate
- doc scope pass rate
- quiz validity rate

## Interview Talking Points
How I would explain this project in an interview:
- Simple RAG is not enough; empty retrieval must refuse to answer.
- Doc scoping prevents mixing conflicting sources.
- Citations reduce hallucination risk and improve auditability.
- Grounding verification adds a deterministic safety layer.
- Prompt-injection attempts are detected and refused.
- CI ensures the project runs in mock mode without external services.
- Next improvements: better semantic grounding, richer citations, and dynamic scoring.

## Local Setup
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn agentic_rag.app:app --reload
```
