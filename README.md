# Agentic RAG (Improved)

A compact, **agentic RAG** template featuring:
- Clear project structure (API + core modules)
- Robust error handling & logging
- Simple agent loop (plan → retrieve → use tools → generate → reflect)
- Unit tests (pytest)
- Documentation and quickstart
- Works locally with FAISS + `all-MiniLM-L6-v2` + HuggingFace Transformers

> Upgrades vs your original script: module boundaries, FastAPI server, typed configs, graceful failures, tests, and a small tool-use agent on top of RAG. Your earlier version was a monolithic script with no app layer or tests (see `rag.py`).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn agentic_rag.app:app --reload
```

### Query the API

```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"question":"What is RAG?", "top_k": 3}'
```

### Load extra docs (optional)
```bash
curl -X POST http://127.0.0.1:8000/load -H "Content-Type: application/json" \
  -d '{"documents":["RAG retrieves facts before generation.","VS Code is by Microsoft."]}'
```

## Project Layout

```
agentic_rag/
  app.py                # FastAPI app (REST API)
  rag/
    config.py           # Settings
    embeddings.py       # SentenceTransformers wrapper
    vectorstore.py      # FAISS store wrapper
    retriever.py        # Top-k retrieval
    prompts.py          # Prompt templates
    llm.py              # HF pipeline wrapper
    tools.py            # Example tools (Calculator, Time)
    agent.py            # Agent loop (plan/retrieve/tools/generate/reflect)
    pipeline.py         # High-level RAG orchestrator
    utils_logger.py     # Logging helper
  data/
    seed_documents.json # Seed corpus
tests/
  test_retriever.py
  test_agent.py
  test_api.py
requirements.txt
README.md
```

## Notes

- Defaults to `google/flan-t5-small` for lightweight local generation. Change in `rag/llm.py` if you like.
- Designed to run CPU-only. If you have GPU, `torch` will use it automatically if available.
