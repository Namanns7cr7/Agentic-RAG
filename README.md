# Agentic RAG

A production-style **Agentic Retrieval-Augmented Generation** system featuring:

- 🧠 **LLM-driven planning** — the agent uses the language model to decide whether to retrieve, use tools, or act
- 🔧 **Tool use** — Calculator (safe AST-based), Time, and Summarize tools
- 💬 **Conversation memory** — maintains context across multiple turns
- 🪞 **Self-reflection** — the agent reviews and corrects its own draft before responding
- 🔍 **Vector search** — FAISS-backed similarity search with automatic NumPy fallback
- 🚀 **FastAPI REST API** — fully documented with Swagger UI
- ✅ **Comprehensive tests** — 25+ unit & integration tests with pytest
- 🐳 **Docker-ready** — containerized deployment out of the box
- ⚙️ **CI/CD** — GitHub Actions workflow for automated testing

> **Architecture**: Clean modular design with proper separation of concerns — embeddings, vector store, retriever, LLM, tools, agent, and pipeline are all independent modules.

---

## Architecture

```
Plan (LLM decides action)
  │
  ├── retrieve → Embed query → FAISS search → Build context prompt → Generate draft
  ├── use_tool:calculator → Safe AST eval → Generate answer from observation
  ├── use_tool:time → Get timestamp → Generate answer from observation
  └── use_tool:summarize → Extract bullet points → Return summary
  │
  ▼
Reflect (LLM reviews and corrects the draft)
  │
  ▼
Remember (store Q&A in conversation memory)
```

---

## Requirements
- Python **3.10–3.13**
- First run downloads model weights (~90 MB, cached thereafter)

---

## Quickstart

### Windows (PowerShell)
```powershell
cd "D:\4 year\agentic_rag"

# Create + activate venv
py -3.13 -m venv .venv
& ".\.venv\Scripts\Activate.ps1"

# Install deps
pip install -r requirements.txt

# Run API
uvicorn agentic_rag.app:app --reload --port 8000

# Open Swagger UI → http://127.0.0.1:8000/docs
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn agentic_rag.app:app --reload --port 8000
# Open http://127.0.0.1:8000/docs
```

### Docker
```bash
docker build -t agentic-rag .
docker run -p 8000:8000 agentic-rag
```

---

## API Endpoints

### `GET /` — Home
```json
{ "message": "Agentic-RAG is running. Visit /docs for the Swagger UI." }
```

### `GET /health` — Health Check
```json
{ "status": "ok", "documents_loaded": 57, "model": "google/flan-t5-small" }
```

### `POST /query` — Ask a Question
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is RAG?","top_k":3}'
```

**Response:**
```json
{
  "plan": "retrieve",
  "draft": "...",
  "final": "...",
  "observation": null
}
```

### `POST /load` — Add Documents
```bash
curl -X POST http://127.0.0.1:8000/load \
  -H "Content-Type: application/json" \
  -d '{"documents":["RAG retrieves facts before generation.","VS Code is by Microsoft."]}'
```

**Response:**
```json
{ "added": 2, "total": 59 }
```

### `POST /memory/clear` — Clear Conversation Memory
```bash
curl -X POST http://127.0.0.1:8000/memory/clear
```

---

## Project Layout
```
agentic_rag/
  app.py                  # FastAPI REST API (/, /health, /query, /load, /memory/clear)
  rag/
    config.py             # Pydantic settings (models, top_k, temperature, memory)
    embeddings.py         # SentenceTransformers wrapper
    vectorstore.py        # FAISS index with NumPy fallback
    retriever.py          # Top-k document retrieval
    prompts.py            # Prompt templates (RAG, reflection, planner)
    llm.py                # HF pipeline wrapper with graceful fallback
    tools.py              # Safe tools (Calculator, Time, Summarize)
    agent.py              # Agent loop (Plan → Act → Reflect → Remember)
    pipeline.py           # High-level RAG orchestrator
    utils_logger.py       # Logging helper
  data/
    seed_documents.json   # 57-document seed corpus
tests/
  test_api.py             # API endpoint tests (13 tests)
  test_agent.py           # Agent & pipeline tests (6 tests)
  test_retriever.py       # Tool & component unit tests (13 tests)
Dockerfile                # Container deployment
.github/workflows/ci.yml  # GitHub Actions CI
requirements.txt
README.md
```

---

## Key Features in Detail

### LLM-Driven Planning
Instead of simple keyword matching, the agent asks the language model to decide the best action. A fast-path heuristic handles obvious cases (math expressions → calculator), while ambiguous queries are routed through the LLM planner.

### Conversation Memory
The agent remembers the last N question-answer pairs (configurable via `max_memory_turns`) and includes them in the prompt context. This enables multi-turn conversations.

### Safe Calculator
Uses Python's `ast` module for expression parsing — no `eval()`. Only arithmetic operators are supported, preventing code injection.

### Self-Reflection
After generating a draft answer, the agent runs a reflection step that checks for hallucinations and unsupported claims, producing a refined final answer.

### FAISS + NumPy Fallback
If FAISS isn't installed, the vector store automatically falls back to a pure-NumPy brute-force search — no code changes needed.

---

## Configuration
Edit defaults in `rag/config.py`:
```python
embedding_model_name = "all-MiniLM-L6-v2"
generator_model_name = "google/flan-t5-small"
top_k_default = 3
max_new_tokens = 128
temperature = 0.7
max_memory_turns = 5
```

---

## Testing
```bash
pip install pytest httpx
pytest -v
```

Expected: **25+ tests passing**, covering API endpoints, agent behavior, tools, and vector store.

---

## Troubleshooting

**`ModuleNotFoundError: sentence_transformers`**
```powershell
pip install sentence-transformers transformers torch
```

**FAISS install fails**
Not required — the code falls back to NumPy automatically:
```powershell
pip uninstall -y faiss-cpu
```

**PowerShell "running scripts is disabled"**
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
& ".\.venv\Scripts\Activate.ps1"
```

---

## Tech Stack
| Component | Technology |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector Store | FAISS / NumPy fallback |
| Generator | `google/flan-t5-small` (HuggingFace) |
| API | FastAPI + Pydantic v2 |
| Server | Uvicorn (ASGI) |
| Tests | pytest + httpx |
| CI/CD | GitHub Actions |
| Container | Docker |

---

## License
MIT
