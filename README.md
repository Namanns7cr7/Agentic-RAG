# 🎓 LearnMate AI — Production-Style Agentic RAG Learning Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-CPU-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Vertex%20AI-Optional-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white" />
  <img src="https://img.shields.io/badge/Tests-13%20suites-brightgreen?style=for-the-badge" />
</p>

> **LearnMate AI** is a production-grade, multi-agent RAG (Retrieval-Augmented Generation) platform that transforms documents into interactive, personalized learning experiences. Upload any document — PDF, DOCX, PPTX, images, or plain text — and instantly get AI-powered explanations, quizzes, flashcards, cheatsheets, coding challenges, and interview prep, all grounded in your own content.

---

## 📖 Table of Contents

- [What It Does](#-what-it-does)
- [Key Features](#-key-features)
- [Architecture Overview](#-architecture-overview)
- [Module Breakdown](#-module-breakdown)
- [API Reference](#-api-reference)
- [LLM Providers](#-llm-providers)
- [Local Setup](#-local-setup)
- [Environment Variables](#-environment-variables)
- [Docker](#-docker)
- [Vertex AI Setup](#-vertex-ai-setup)
- [Testing](#-testing)
- [Evaluation Harness](#-evaluation-harness)
- [Grounding & Safety Contract](#-grounding--safety-contract)
- [Interview Talking Points](#-interview-talking-points)

---

## 🤖 What It Does

LearnMate AI is built around two complementary AI agents:

| Agent | Purpose |
|---|---|
| **LearningAgent** | Adaptive concept teaching, answer evaluation, flashcard & learning-path generation using the seed knowledge base |
| **DocLearningAgent** | Document-scoped Q&A, quizzes, coding challenges, cheatsheets, debugging help, and interview prep — all grounded strictly in *your* uploaded files |

The platform **refuses to hallucinate**: if retrieved context is empty or too weak, every endpoint returns `unsupported` rather than an invented answer.

---

## ✨ Key Features

- 📄 **Multi-format document ingestion** — PDF, DOCX, PPTX, TXT, CSV, and scanned images (via OCR)
- 🔍 **FAISS vector search** with `all-MiniLM-L6-v2` embeddings
- 🧠 **Adaptive learner profiles** — tracks skill level, learning history, and preferred style
- 🛡️ **Grounding verification** — deterministic citation checks + self-reflection loop
- 🚦 **Hard guardrails** — empty/weak retrieval → `unsupported` answer, never hallucinated
- 🔌 **Pluggable LLM backends** — Ollama, OpenAI-compatible (vLLM / LM-Studio), local HuggingFace, Vertex AI Gemini
- 🧪 **13 test suites** covering unit, API, retrieval, guardrails, mock-mode, and LLM-failure scenarios
- 📊 **Evaluation harness** with 6 automated metrics
- 🐳 **Docker-ready** — single `docker build` + `docker run` to production

---

## 🏗️ Architecture Overview

```
User Query / Uploaded Document
        │
        ▼
┌─────────────────────┐
│   FastAPI App       │  ← CORS, routing, dependency injection (ServiceContainer)
└────────┬────────────┘
         │
   ┌─────┴──────────────────────┐
   │                            │
   ▼                            ▼
┌──────────────┐     ┌──────────────────────┐
│ LearningAgent│     │  DocLearningAgent    │
│  /learn      │     │  /docs/ask           │
│  /evaluate   │     │  /docs/quiz          │
│  /flashcards │     │  /docs/challenge     │
│  /learn-path │     │  /docs/cheatsheet    │
└──────┬───────┘     │  /docs/debug         │
       │             │  /docs/interview      │
       │             └──────────┬───────────┘
       │                        │
       └──────────┬─────────────┘
                  ▼
        ┌─────────────────┐
        │    Pipeline     │  ← Orchestrates retrieval + generation
        └────────┬────────┘
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
┌─────────┐ ┌────────┐ ┌──────────────┐
│Retriever│ │FAISS   │ │  LLMProvider │
│doc_id   │ │Vector  │ │  (pluggable) │
│scoping  │ │Store   │ │              │
└─────────┘ └────────┘ └──────────────┘
                 │
                 ▼
        ┌────────────────┐
        │  Grounding     │  ← Citation check + reflection
        │  Verification  │
        └────────────────┘
                 │
                 ▼
        answered | unsupported | error
```

---

## 📦 Module Breakdown

### `agentic_rag/rag/` — Core RAG Engine

| File | Responsibility |
|---|---|
| `config.py` | Pydantic `Settings` (embedding model, top-k, temperature, memory turns) |
| `pipeline.py` | End-to-end query orchestration — retrieve → generate → verify |
| `retriever.py` | FAISS-backed vector retriever with optional `doc_id` scoping |
| `vectorstore.py` | FAISS index wrapper; supports add, search, and per-doc filtering |
| `embeddings.py` | `all-MiniLM-L6-v2` sentence-transformer embeddings |
| `llm_provider.py` | Pluggable LLM factory (`OllamaProvider`, `OpenAICompatibleProvider`, `LocalHFProvider`, `VertexGeminiProvider`) |
| `llm.py` | Local HuggingFace `LocalGenerator` (fallback) |
| `grounding.py` | Deterministic citation overlap check + LLM self-reflection |
| `agent.py` | Plan-draft-reflect agentic loop |
| `tools.py` | Tool definitions for the agentic loop |
| `prompts.py` | System prompt templates |
| `mock_components.py` | Mock embeddings, generator, and LLM provider for CI/CD |
| `utils_logger.py` | Structured logger |

### `agentic_rag/learning/` — Learning & DocuMentor Agents

| File | Responsibility |
|---|---|
| `learning_agent.py` | Adaptive teaching, answer evaluation, flashcards, learning paths |
| `doc_learning_agent.py` | Document-scoped ask, quiz, challenge, cheatsheet, debug, interview |
| `learning_memory.py` | Per-user conversation memory (capped at `max_memory_turns`) |
| `learner_profile.py` | Tracks skill level, sessions, topics, preferred style |
| `concept_explainer.py` | Generates structured explanations with analogies and examples |
| `teaching_strategy.py` | Selects teaching approach based on learner profile |
| `query_classifier.py` | Classifies query type for routing |
| `quiz_generator.py` | Generates validated multiple-choice questions |
| `answer_evaluator.py` | Evaluates free-text answers and detects misconceptions |
| `misconception_detector.py` | Identifies common learner misconceptions |
| `learning_path.py` | Builds structured multi-step learning plans |
| `document_processor.py` | Extracts text from PDF, DOCX, PPTX, TXT, CSV |
| `ocr_engine.py` | OCR for scanned PDFs and images (Pillow-based) |
| `text_chunker.py` | Character / sentence / paragraph chunking strategies |
| `learning_prompts.py` | All prompt templates for learning flows |
| `schemas.py` | Pydantic request/response models for every endpoint |

---

## 🔌 API Reference

### Core RAG

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns model name and document count |
| `POST` | `/query` | Plan-draft-reflect RAG query against seed knowledge base |
| `POST` | `/load` | Add raw text documents to the knowledge base |
| `POST` | `/memory/clear` | Clear conversation memory |

### Adaptive Learning

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/learn` | Personalised concept explanation (level + style adaptive) |
| `POST` | `/evaluate` | Evaluate a learner's answer with adaptive feedback |
| `GET` | `/profile/{user_id}` | Fetch learner profile and history |
| `POST` | `/flashcards` | Generate flashcards for a topic |
| `POST` | `/learning-path` | Generate a structured multi-step learning plan |
| `POST` | `/teach-from-docs` | RAG-based teaching from seed documents |

### Document Management

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload-document` | Upload one file (PDF/DOCX/PPTX/TXT/CSV/image) |
| `POST` | `/upload-documents` | Batch upload multiple files |
| `GET` | `/documents` | List all uploaded documents with metadata |
| `POST` | `/upload-and-learn` | Upload a file and immediately teach from it |

### DocuMentor AI (`/docs/*`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/docs/ask` | Grounded Q&A from uploaded docs |
| `POST` | `/docs/quiz` | Generate quiz questions from uploaded docs |
| `POST` | `/docs/challenge` | Generate coding challenges from uploaded docs |
| `POST` | `/docs/cheatsheet` | Generate a cheatsheet from uploaded docs |
| `POST` | `/docs/debug` | Debug code using uploaded docs as context |
| `POST` | `/docs/interview` | Generate interview-style questions from uploaded docs |

Interactive Swagger docs available at `http://localhost:8000/docs`.

---

## 🔌 LLM Providers

LearnMate AI auto-detects your LLM backend from environment variables:

```
LLM_PROVIDER=ollama             →  OllamaProvider (POST /api/chat)
LLM_PROVIDER=openai_compatible  →  OpenAICompatibleProvider (vLLM, LM-Studio, etc.)
LLM_PROVIDER=local              →  LocalHFProvider (google/flan-t5-base, CPU)
RAG_PROVIDER=vertex             →  VertexGeminiProvider (Gemini 2.5 Flash on GCP)
```

**Automatic fallback**: if Ollama or an OpenAI-compatible server is unreachable at startup, the system silently falls back to the local HuggingFace model — no crash, no user impact.

**Silent retry**: `VertexGeminiProvider` performs one automatic retry on transient Google API errors (`ServiceUnavailable`, `DeadlineExceeded`).

**`<think>` block stripping** is applied to **all** providers (Ollama, OpenAI-compatible, and Vertex) so reasoning traces never leak into answers.

### Embedding Providers

| `EMBEDDING_PROVIDER` | Backend | Notes |
|---|---|---|
| `local` (default) | `all-MiniLM-L6-v2` via sentence-transformers | CPU, no cloud cost |
| `vertex` | `text-embedding-004` via Vertex AI | Higher quality; uses `task_type` for doc vs. query |

Set `EMBEDDING_PROVIDER=vertex` independently of `RAG_PROVIDER` for mixed configs (e.g., local LLM + Vertex embeddings).

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11+
- (Optional) [Ollama](https://ollama.com/) for local LLM inference

```powershell
# 1. Clone and create virtual environment
git clone https://github.com/Namanns7cr7/Agentic-RAG.git
cd Agentic-RAG

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env with your preferred LLM provider settings

# 4. Start the server
uvicorn agentic_rag.app:app --reload
```

The API will be available at `http://localhost:8000`.  
Swagger UI: `http://localhost:8000/docs`

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and adjust:

```env
# ── LLM Provider ──────────────────────────────────────────────────
LLM_PROVIDER=vertex               # ollama | openai_compatible | local | vertex
LLM_MODEL=qwen3.5:9b              # model tag / name

# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI-compatible (vLLM, LM-Studio, etc.)
OPENAI_COMPATIBLE_BASE_URL=http://localhost:8000/v1
OPENAI_COMPATIBLE_API_KEY=dummy

# HuggingFace fallback model
FALLBACK_MODEL=google/flan-t5-base

# ── Vertex AI (optional) ──────────────────────────────────────────
# Activate: gcloud auth application-default login
# Set LLM_PROVIDER=vertex to use Vertex AI Gemini on Google Cloud.
GOOGLE_CLOUD_PROJECT=your-project-id  # your GCP project ID
GOOGLE_CLOUD_LOCATION=us-central1
VERTEX_MODEL=gemini-2.0-flash         # default Vertex Gemini model
VERTEX_EMBEDDING_MODEL=text-embedding-005 # default Vertex Embedding model

# ── Token & Cost Controls ──────────────────────────────────────────
MAX_CONTEXT_CHUNKS=5                  # limit retrieved context chunks
MAX_OUTPUT_TOKENS=700                 # limit max output response tokens
TEMPERATURE=0.2                       # low temperature for RAG correctness
MAX_CHARS_PER_CHUNK=2500              # limit max characters per chunk text

# ── Embedding Provider ────────────────────────────────────────────
# Options: local | vertex  (independent of RAG_PROVIDER)
EMBEDDING_PROVIDER=local              # local (MiniLM) or vertex (text-embedding-005)

# ── Testing ───────────────────────────────────────────────────────
RAG_MOCK_MODE=false               # set to "true" for CI (no model downloads)
```

---

## 🐳 Docker

```bash
# Build
docker build -t learnmate-ai .

# Run (with Ollama on host)
docker run -p 8000:8000 \
  -e LLM_PROVIDER=ollama \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  learnmate-ai

# Run in mock mode (no LLM needed)
docker run -p 8000:8000 -e RAG_MOCK_MODE=true learnmate-ai
```

---

## ☁️ Vertex AI Setup

### Authentication (local development)

```powershell
# One-time login — uses your personal Google account credentials
gcloud auth application-default login
```

For production / Cloud Run / GKE, attach a service account with the `Vertex AI User` IAM role to your compute instance instead.

### Enable Vertex AI (Gemini + embeddings)

```powershell
# Required
$env:RAG_PROVIDER="vertex"
$env:GOOGLE_CLOUD_PROJECT="your-project-id"

# Optional (defaults shown)
$env:GOOGLE_CLOUD_LOCATION="us-central1"
$env:VERTEX_GEMINI_MODEL="gemini-2.5-flash"     # or gemini-2.0-flash
$env:EMBEDDING_PROVIDER="vertex"                 # enable Vertex embeddings too
$env:VERTEX_EMBEDDING_MODEL="text-embedding-004"

uvicorn agentic_rag.app:app --reload
```

### Mixed config (Vertex embeddings + local LLM)

```powershell
# Use text-embedding-004 for higher-quality retrieval, keep local LLM
$env:EMBEDDING_PROVIDER="vertex"
$env:GOOGLE_CLOUD_PROJECT="your-project-id"
$env:LLM_PROVIDER="ollama"
$env:LLM_MODEL="qwen3.5:9b"

uvicorn agentic_rag.app:app --reload
```

> ⚠️ **Cost warning**: Vertex AI usage incurs charges. Configure GCP budgets and billing alerts before enabling in production.
>
> `gemini-2.5-flash` pricing: ~$0.0075 / 1K input tokens. A full eval run costs < $0.05.

### Embedding task types

`text-embedding-004` supports `task_type` — the integration automatically uses:
- `RETRIEVAL_DOCUMENT` when indexing uploaded chunks
- `RETRIEVAL_QUERY` when embedding a user's question

This improves retrieval precision vs. using a single generic task type.

---

## 🧪 Testing

14 test suites with full mock-mode support (no model downloads required for CI):

```powershell
# Run all tests
python -m pytest -q

# Run in mock mode (fast, no GPU/downloads)
$env:RAG_MOCK_MODE="true"
python -m pytest -q

# Run a specific suite
python -m pytest tests/test_doc_guardrails.py -v
```

### Test Coverage

| Suite | What it covers |
|---|---|
| `test_api.py` | Core RAG endpoints (`/query`, `/load`, `/health`) |
| `test_api_docs_endpoints.py` | All `/docs/*` DocuMentor endpoints |
| `test_learning.py` | `/learn`, `/evaluate`, `/flashcards`, `/learning-path` |
| `test_learning_components.py` | Individual learning module unit tests |
| `test_learning_memory.py` | Per-user memory accumulation and capping |
| `test_retriever.py` | FAISS retrieval and `doc_id` scoping |
| `test_agent.py` | Plan-draft-reflect agentic loop |
| `test_doc_guardrails.py` | Empty retrieval → `unsupported`, prompt injection detection |
| `test_query_classifier.py` | Query type routing |
| `test_llm_provider.py` | LLM provider abstraction (Ollama, OpenAI-compatible) |
| `test_vertex_provider.py` | **Vertex AI** — `VertexGeminiProvider`, `VertexEmbeddingProvider`, factories, SDK-missing degradation |
| `test_llm_failure.py` | Graceful degradation on LLM errors |
| `test_mock_mode.py` | Mock-mode smoke test |
| `conftest.py` | Shared fixtures |

---

## 📊 Evaluation Harness

The evaluation harness runs a suite of grounding test cases and writes results to `evaluation/results.json`.

```powershell
# Standard evaluation (mock mode — fast, no credentials)
python evaluation/evaluator.py

# Vertex AI evaluation (requires gcloud auth + GOOGLE_CLOUD_PROJECT)
python evaluation/vertex_eval.py
```

`vertex_eval.py` runs the **same 6 metrics** against the live Vertex AI endpoint and writes `evaluation/vertex_results.json` for side-by-side comparison with the local run.

### Metrics Reported

| Metric | Description |
|---|---|
| **Status accuracy** | % of queries returning correct `answered` / `unsupported` |
| **Citation accuracy** | % of `answered` responses with valid citations |
| **Forbidden keyword failures** | Responses containing disallowed content |
| **Safety pass rate** | % of responses passing safety checks |
| **Doc scope pass rate** | % of responses respecting `doc_id` boundaries |
| **Quiz validity rate** | % of generated quizzes with valid structure |

---

## 🛡️ Grounding & Safety Contract

Every response from the DocuMentor `/docs/*` endpoints adheres to this contract:

```json
{
  "answer": "...",
  "status": "answered | unsupported | error",
  "citations": [{ "chunk_id": "...", "text": "...", "doc_id": "..." }],
  "confidence": 0.87,
  "unsupported_reason": null,
  "used_doc_ids": ["abc-123"],
  "safety_flags": []
}
```

**Hard rules:**
- If retrieval returns no chunks or similarity is below threshold → `status: unsupported`
- Every `answered` response **must** include `citations`
- If grounding verification fails → response is downgraded to `unsupported`
- Prompt injection attempts are detected and refused

---

## 💬 Interview Talking Points

Key design decisions worth discussing:

- **Why refuse instead of hallucinate?** — Empty retrieval returns `unsupported`. Users trust the system more when it admits uncertainty.
- **Why `doc_id` scoping?** — Prevents conflicting information from different documents bleeding into each other's answers.
- **Why citations?** — Reduces hallucination risk and makes answers auditable; users can verify every claim.
- **Why grounding verification?** — Deterministic overlap check + LLM self-reflection adds a dual safety layer over pure generation.
- **Why pluggable LLM backends?** — Hardware varies; the same codebase works on a laptop with Ollama, a workstation with vLLM, or Google Cloud with Vertex AI.
- **Why mock mode in CI?** — Integration tests run in ~seconds without GPU or cloud credentials.
- **Next improvements:** Streaming responses, re-ranking with cross-encoders, richer citation metadata, dynamic difficulty scoring.

---

## 📁 Project Structure

```
agentic_rag/
├── agentic_rag/
│   ├── app.py                  # FastAPI application & all route definitions
│   ├── data/
│   │   └── seed_documents.json # Pre-loaded knowledge base
│   ├── static/
│   │   └── index.html          # Web UI
│   ├── rag/                    # Core RAG engine
│   │   ├── config.py
│   │   ├── pipeline.py
│   │   ├── retriever.py
│   │   ├── vectorstore.py
│   │   ├── embeddings.py
│   │   ├── llm_provider.py
│   │   ├── llm.py
│   │   ├── grounding.py
│   │   ├── agent.py
│   │   └── mock_components.py
│   └── learning/               # Adaptive learning & DocuMentor agents
│       ├── learning_agent.py
│       ├── doc_learning_agent.py
│       ├── learning_memory.py
│       ├── document_processor.py
│       ├── ocr_engine.py
│       ├── text_chunker.py
│       └── schemas.py
├── tests/                      # 13 test suites
├── evaluation/                 # Evaluation harness + queries
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 📄 License

This project is for educational and portfolio purposes.

---

<p align="center">Built with ❤️ using FastAPI · FAISS · Sentence Transformers · Pydantic · Ollama · Vertex AI Gemini</p>
