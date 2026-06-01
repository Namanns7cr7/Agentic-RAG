"""FastAPI application — LearnMate AI: Agentic Personalized Learning Assistant."""

from datetime import datetime
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from .rag.config import Settings
from .rag.pipeline import Pipeline
from .rag.llm_provider import get_llm_provider
from .rag.vectorstore import VectorStore
from .rag.mock_components import MockEmbeddings, MockGenerator, MockLLMProvider
from .learning.learning_agent import LearningAgent
from .learning.learning_memory import LearningMemory
from .learning.document_processor import get_document_processor
from .learning.schemas import (
    LearnRequest, LearnResponse,
    EvaluateRequest, EvaluateResponse,
    LearnerProfileResponse,
    FlashcardRequest, FlashcardResponse,
    LearningPathRequest, LearningPathResponse,
    TeachFromDocsRequest,
    DocumentUploadResponse,
    DocumentListResponse, DocumentInfo,
    # DocuMentor AI schemas
    DocAskRequest, DocAskResponse,
    DocChallengeRequest, DocChallengeResponse,
    DocQuizRequest, DocQuizResponse,
    DocCheatsheetRequest, DocCheatsheetResponse,
    DocDebugRequest, DocDebugResponse,
    DocInterviewRequest, DocInterviewResponse,
)
from .learning.doc_learning_agent import DocLearningAgent
import json
from pathlib import Path
from uuid import uuid4

STATIC_DIR = Path(__file__).parent / "static"


# ── Request / Response Models (original RAG) ─────────────────────

class QueryIn(BaseModel):
    question: str = Field(..., min_length=2, description="The question to ask")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of documents to retrieve")


class QueryOut(BaseModel):
    plan: str
    draft: str
    final: str
    observation: Optional[str] = None


class LoadIn(BaseModel):
    documents: List[str] = Field(..., min_length=1, description="Documents to add to the knowledge base")


class LoadOut(BaseModel):
    added: int
    total: int


class HealthOut(BaseModel):
    status: str
    documents_loaded: int
    model: str


class UploadDocumentResponse(BaseModel):
    doc_id: str
    filename: str
    total_characters: int
    total_chunks: int
    status: str


class AskCitation(BaseModel):
    doc_id: str
    filename: str
    chunk_id: str
    page_number: Optional[int] = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    status: str
    citations: List[AskCitation]
    retrieved_chunks_count: int
    provider: str
    model: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, description="The learning or documentation question")
    doc_id: Optional[str] = Field(default=None, description="Optional document ID scoping")
    mode: str = Field(default="qa", description="Workflow mode: qa, teach, quiz, or interview")


# ── App Setup ─────────────────────────────────────────────────────

def load_seed() -> list:
    path = Path(__file__).parent / "data" / "seed_documents.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _is_mock_mode() -> bool:
    return os.getenv("RAG_MOCK_MODE", "").lower() in ("1", "true", "yes") or os.getenv("TESTING", "").lower() in (
        "1",
        "true",
        "yes",
    )


from .rag.utils_logger import get_logger
logger = get_logger(__name__)


class ServiceContainer:
    def __init__(self, mock_mode: bool):
        self.mock_mode = mock_mode
        self._settings: Optional[Settings] = None
        self._pipe: Optional[Pipeline] = None
        self._llm_provider: Optional[object] = None
        self._learning_memory: Optional[LearningMemory] = None
        self._learning_agent: Optional[LearningAgent] = None
        self._doc_agent: Optional[DocLearningAgent] = None
        self._doc_store: Optional[object] = None
        self._agent_pipeline: Optional[object] = None

    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = Settings()
        return self._settings

    def pipeline(self) -> Pipeline:
        if self._pipe is None:
            settings = self.settings()
            if self.mock_mode:
                embedder = MockEmbeddings()
                store = VectorStore(embedder.dim)
                generator = MockGenerator()
                self._pipe = Pipeline(
                    settings,
                    seed_docs=load_seed(),
                    embedder=embedder,
                    store=store,
                    generator=generator,
                )
            else:
                self._pipe = Pipeline(settings, seed_docs=[])
                # Load saved FAISS index if it exists
                faiss_path = os.path.join("data", "index", "faiss.index")
                if os.path.exists(faiss_path):
                    try:
                        self._pipe.store.load(faiss_path)
                        logger.info("Loaded persisted FAISS index from %s", faiss_path)
                    except Exception as e:
                        logger.error("Failed to load persisted FAISS index: %s", e)
                
                # If the index is empty, add the seed docs
                if self._pipe.store.size == 0:
                    logger.info("Vector index is empty. Ingesting seed documents.")
                    self._pipe.add_documents(load_seed(), doc_id="seed")
        return self._pipe

    def llm_provider(self):
        if self._llm_provider is None:
            self._llm_provider = MockLLMProvider() if self.mock_mode else get_llm_provider()
        return self._llm_provider

    def learning_memory(self) -> LearningMemory:
        if self._learning_memory is None:
            self._learning_memory = LearningMemory()
        return self._learning_memory

    def learning_agent(self) -> LearningAgent:
        if self._learning_agent is None:
            pipe = self.pipeline()
            self._learning_agent = LearningAgent(
                llm=self.llm_provider(),
                memory=self.learning_memory(),
                retriever=pipe.retriever,
            )
        return self._learning_agent

    def doc_agent(self) -> DocLearningAgent:
        if self._doc_agent is None:
            pipe = self.pipeline()
            self._doc_agent = DocLearningAgent(
                llm=self.llm_provider(),
                memory=self.learning_memory(),
                retriever=pipe.retriever,
            )
        return self._doc_agent

    def doc_store(self):
        if self._doc_store is None:
            from .rag.document_store import DocumentStore
            self._doc_store = DocumentStore()
        return self._doc_store

    def agent_pipeline(self):
        if self._agent_pipeline is None:
            from .rag.agent_pipeline import MultiAgentPipeline
            self._agent_pipeline = MultiAgentPipeline(self.pipeline().retriever, self.llm_provider())
        return self._agent_pipeline


_services = ServiceContainer(mock_mode=_is_mock_mode())


def create_app() -> FastAPI:
    app = FastAPI(
        title="LearnMate AI",
        description=(
            "LearnMate AI is a production-style Agentic-RAG learning assistant that helps "
            "users learn concepts effectively through personalized explanations, adaptive "
            "quizzes, misconception detection, conversational memory, and self-reflection."
        ),
        version="3.0.0",
    )

    # CORS — allow the frontend to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


# ── Original RAG Endpoints ───────────────────────────────────────

@app.get("/", tags=["info"], include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthOut, tags=["info"])
def health():
    pipe = _services.pipeline()
    settings = _services.settings()
    return HealthOut(
        status="ok",
        documents_loaded=pipe.doc_count,
        model=settings.generator_model_name,
    )


@app.post("/query", response_model=QueryOut, tags=["rag"])
def query(payload: QueryIn):
    try:
        out = _services.pipeline().answer(payload.question, top_k=payload.top_k)
        return QueryOut(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer: {e}")


@app.post("/load", response_model=LoadOut, tags=["rag"])
def load(payload: LoadIn):
    try:
        pipe = _services.pipeline()
        n = pipe.add_documents(payload.documents)
        return LoadOut(added=n, total=pipe.doc_count)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load docs: {e}")


@app.post("/memory/clear", tags=["memory"])
def clear_memory():
    _services.pipeline().clear_memory()
    return {"status": "memory cleared"}


# ── LearnMate AI Learning Endpoints ──────────────────────────────

@app.post("/learn", response_model=LearnResponse, tags=["learning"])
def learn_endpoint(payload: LearnRequest):
    """Personalised concept explanation with adaptive teaching strategy."""
    try:
        result = _services.learning_agent().learn(
            user_id=payload.user_id,
            topic=payload.topic,
            level=payload.level.value,
            goal=payload.goal.value,
            context=payload.context,
            preferred_style=payload.preferred_style.value,
        )
        return LearnResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning failed: {e}")


@app.post("/evaluate", response_model=EvaluateResponse, tags=["learning"])
def evaluate_endpoint(payload: EvaluateRequest):
    """Evaluate a learner's answer and provide adaptive feedback."""
    try:
        result = _services.learning_agent().evaluate(
            user_id=payload.user_id,
            topic=payload.topic,
            question=payload.question,
            answer=payload.answer,
        )
        return EvaluateResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")


@app.get("/profile/{user_id}", response_model=LearnerProfileResponse, tags=["learning"])
def profile_endpoint(user_id: str):
    """Return the learner profile."""
    try:
        result = _services.learning_agent().get_profile(user_id)
        return LearnerProfileResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile retrieval failed: {e}")


@app.post("/flashcards", response_model=FlashcardResponse, tags=["learning"])
def flashcards_endpoint(payload: FlashcardRequest):
    """Generate flashcards for a topic."""
    try:
        cards = _services.learning_agent().flashcards(
            user_id=payload.user_id,
            topic=payload.topic,
            count=payload.count,
            level=payload.level.value,
        )
        return FlashcardResponse(flashcards=cards)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flashcard generation failed: {e}")


@app.post("/learning-path", response_model=LearningPathResponse, tags=["learning"])
def learning_path_endpoint(payload: LearningPathRequest):
    """Generate a structured learning path for a topic."""
    try:
        result = _services.learning_agent().learning_path(
            user_id=payload.user_id,
            topic=payload.topic,
            goal=payload.goal.value,
        )
        return LearningPathResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning path generation failed: {e}")


@app.post("/teach-from-docs", response_model=LearnResponse, tags=["learning"])
def teach_from_docs_endpoint(payload: TeachFromDocsRequest):
    """Retrieve relevant documents and teach the concept using RAG."""
    try:
        result = _services.learning_agent().teach_from_docs(
            user_id=payload.user_id,
            query=payload.query,
            level=payload.level.value,
        )
        return LearnResponse(
            topic=payload.query,
            detected_level=payload.level.value,
            teaching_strategy="rag_based",
            explanation=result.get("explanation", ""),
            analogy=result.get("analogy", ""),
            real_life_example=result.get("real_life_example", ""),
            key_points=result.get("key_points", []),
            quick_check_question=result.get("quick_check_question", ""),
            next_step=result.get("next_step", ""),
            retrieved_sources_used=result.get("retrieved_sources_used", False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Teach from docs failed: {e}")


# ── Document Upload Endpoints ────────────────────────────────────

# In-memory document registry (tracks uploaded files)
_uploaded_documents: List[dict] = []


@app.post("/upload-document", response_model=DocumentUploadResponse, tags=["documents"])
async def upload_document(
    file: UploadFile = File(..., description="Document to upload (PDF, DOCX, TXT, images, PPTX, CSV)"),
    chunk_strategy: str = Form(default="sentence", description="Chunking strategy: character, sentence, or paragraph"),
    chunk_size: int = Form(default=500, ge=100, le=2000, description="Max chunk size in characters"),
):
    """Upload a document file, extract text (with OCR for images/scanned PDFs), 
    chunk it, and add to the knowledge base for RAG retrieval."""
    try:
        file_bytes = await file.read()
        processor = get_document_processor()
        result = processor.process(
            file_bytes,
            file.filename or "unknown",
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
        )

        # Add chunks to the RAG knowledge base
        doc_id = str(uuid4())
        chunks = result["chunks"]
        if chunks:
            _services.pipeline().add_documents(chunks, doc_id=doc_id)

        # Track the upload
        _uploaded_documents.append({
            "doc_id": doc_id,
            "filename": result["filename"],
            "file_type": result["file_type"],
            "total_chars": result["total_chars"],
            "num_chunks": result["num_chunks"],
            "uploaded_at": datetime.utcnow().isoformat(),
        })

        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=result["filename"],
            file_type=result["file_type"],
            total_chars=result["total_chars"],
            num_chunks=result["num_chunks"],
            chunks_added_to_knowledge_base=len(chunks),
            preview=result["preview"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {e}")


@app.post("/upload-documents", response_model=List[DocumentUploadResponse], tags=["documents"])
async def upload_multiple_documents(
    files: List[UploadFile] = File(..., description="Multiple documents to upload"),
    chunk_strategy: str = Form(default="sentence"),
    chunk_size: int = Form(default=500, ge=100, le=2000),
):
    """Upload multiple documents at once. Each file is processed, chunked, 
    and added to the knowledge base."""
    results = []
    processor = get_document_processor()

    for file in files:
        try:
            file_bytes = await file.read()
            result = processor.process(
                file_bytes,
                file.filename or "unknown",
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
            )

            doc_id = str(uuid4())
            chunks = result["chunks"]
            if chunks:
                _services.pipeline().add_documents(chunks, doc_id=doc_id)

            _uploaded_documents.append({
                "doc_id": doc_id,
                "filename": result["filename"],
                "file_type": result["file_type"],
                "total_chars": result["total_chars"],
                "num_chunks": result["num_chunks"],
                "uploaded_at": datetime.utcnow().isoformat(),
            })

            results.append(DocumentUploadResponse(
                doc_id=doc_id,
                filename=result["filename"],
                file_type=result["file_type"],
                total_chars=result["total_chars"],
                num_chunks=result["num_chunks"],
                chunks_added_to_knowledge_base=len(chunks),
                preview=result["preview"],
            ))
        except Exception as e:
            results.append(DocumentUploadResponse(
                doc_id="",
                filename=file.filename or "unknown",
                file_type="error",
                total_chars=0,
                num_chunks=0,
                chunks_added_to_knowledge_base=0,
                preview=f"Error: {e}",
            ))

    return results


@app.get("/documents", tags=["documents"])
def list_documents():
    """List all uploaded documents and their metadata."""
    store = _services.doc_store()
    docs = store.list_documents()
    mapped = []
    for d in docs:
        ext = os.path.splitext(d["filename"])[1].lower()
        try:
            up_time = datetime.fromtimestamp(d["uploaded_at"]).isoformat()
        except Exception:
            up_time = datetime.utcnow().isoformat()
        
        mapped.append({
            "doc_id": d["doc_id"],
            "filename": d["filename"],
            "file_type": ext,
            "total_chars": d["total_characters"],
            "num_chunks": d["total_chunks"],
            "uploaded_at": up_time
        })
    return {
        "total_documents": len(mapped),
        "documents": mapped
    }


@app.post("/documents/upload", response_model=UploadDocumentResponse, tags=["documents"])
async def upload_document_endpoint(
    file: UploadFile = File(..., description="Document to upload (.pdf, .txt, .md, .docx)"),
    chunk_size: int = Form(default=1000, ge=100, le=5000, description="Chunk size in characters"),
):
    """Upload a document, calculate file hash for deduplication, extract text, 
    generate chunks with overlap, and add to FAISS/persist locally.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".pdf", ".txt", ".md", ".docx"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: .pdf, .txt, .md, .docx"
        )
    try:
        file_bytes = await file.read()
        store = _services.doc_store()
        pipe = _services.pipeline()
        
        res = store.add_document(
            filename=file.filename or "unknown",
            file_bytes=file_bytes,
            pipeline=pipe,
            chunk_size=chunk_size,
        )
        return UploadDocumentResponse(**res)
    except Exception as e:
        logger.error("Failed to upload document: %s", e)
        raise HTTPException(status_code=500, detail=f"Document upload failed: {e}")


@app.post("/documents/ask", response_model=AskResponse, tags=["documents"])
def ask_document_endpoint(payload: AskRequest):
    """Query the multi-agent pipeline scoped to a specific document or global index."""
    try:
        pipeline = _services.agent_pipeline()
        res = pipeline.ask(
            question=payload.question,
            doc_id=payload.doc_id,
            mode=payload.mode,
            top_k=5,
        )
        return AskResponse(**res)
    except Exception as e:
        logger.error("Failed to answer via agent pipeline: %s", e)
        raise HTTPException(status_code=500, detail=f"Agent workflow failed: {e}")


@app.delete("/documents/{doc_id}", tags=["documents"])
def delete_document_endpoint(doc_id: str):
    """Delete a document and all its chunks from local storage and vector store."""
    store = _services.doc_store()
    pipe = _services.pipeline()
    success = store.delete_document(doc_id, pipe)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found.")
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/upload-and-learn", response_model=LearnResponse, tags=["documents"])
async def upload_and_learn(
    file: UploadFile = File(..., description="Document to upload and learn from"),
    user_id: str = Form(..., min_length=1, description="Learner identifier"),
    topic: str = Form(default="", description="Topic to focus on (auto-detected from document if empty)"),
    level: str = Form(default="auto", description="Learner level"),
    goal: str = Form(default="conceptual", description="Learning goal"),
    preferred_style: str = Form(default="simple", description="Preferred teaching style"),
):
    """Upload a document, extract its content, add to knowledge base, 
    then immediately teach the user from it."""
    try:
        # 1. Process the document
        file_bytes = await file.read()
        processor = get_document_processor()
        result = processor.process(file_bytes, file.filename or "unknown")

        # 2. Add to knowledge base
        doc_id = str(uuid4())
        chunks = result["chunks"]
        if chunks:
            _services.pipeline().add_documents(chunks, doc_id=doc_id)

        _uploaded_documents.append({
            "doc_id": doc_id,
            "filename": result["filename"],
            "file_type": result["file_type"],
            "total_chars": result["total_chars"],
            "num_chunks": result["num_chunks"],
            "uploaded_at": datetime.utcnow().isoformat(),
        })

        # 3. Determine topic
        effective_topic = topic if topic.strip() else f"Content from {result['filename']}"

        # 4. Teach from the newly uploaded content
        teach_result = _services.learning_agent().teach_from_docs(
            user_id=user_id,
            query=effective_topic,
            level=level,
        )

        return LearnResponse(
            topic=effective_topic,
            detected_level=level,
            teaching_strategy="document_based",
            explanation=teach_result.get("explanation", ""),
            analogy=teach_result.get("analogy", ""),
            real_life_example=teach_result.get("real_life_example", ""),
            key_points=teach_result.get("key_points", []),
            quick_check_question=teach_result.get("quick_check_question", ""),
            next_step=teach_result.get("next_step", ""),
            retrieved_sources_used=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload and learn failed: {e}")


# ── DocuMentor AI Endpoints (/docs/*) ────────────────────────────

@app.post("/docs/ask", response_model=DocAskResponse, tags=["documenter"])
def docs_ask(payload: DocAskRequest):
    """Ask a learning question about uploaded coding documentation."""
    try:
        result = _services.doc_agent().ask(
            user_id=payload.user_id,
            question=payload.question,
            mode=payload.mode.value,
            coding_level=payload.coding_level.value,
            preferred_language=payload.preferred_language.value,
            preferred_style=payload.preferred_style,
            code_context=payload.code_context,
            doc_id=payload.doc_id,
        )
        return DocAskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DocAsk failed: {e}")


@app.post("/docs/challenge", response_model=DocChallengeResponse, tags=["documenter"])
def docs_challenge(payload: DocChallengeRequest):
    """Generate a coding challenge grounded in uploaded docs."""
    try:
        result = _services.doc_agent().generate_challenge(
            user_id=payload.user_id,
            topic=payload.topic,
            difficulty=payload.difficulty.value,
            language=payload.language.value,
            doc_id=payload.doc_id,
        )
        return DocChallengeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Challenge generation failed: {e}")


@app.post("/docs/quiz", response_model=DocQuizResponse, tags=["documenter"])
def docs_quiz(payload: DocQuizRequest):
    """Generate quiz questions grounded in uploaded docs."""
    try:
        result = _services.doc_agent().generate_quiz(
            user_id=payload.user_id,
            topic=payload.topic,
            difficulty=payload.difficulty.value,
            num_questions=payload.num_questions,
            doc_id=payload.doc_id,
        )
        return DocQuizResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {e}")


@app.post("/docs/cheatsheet", response_model=DocCheatsheetResponse, tags=["documenter"])
def docs_cheatsheet(payload: DocCheatsheetRequest):
    """Generate a cheat sheet from uploaded documentation."""
    try:
        result = _services.doc_agent().generate_cheatsheet(
            user_id=payload.user_id,
            topic=payload.topic or "",
            doc_id=payload.doc_id,
        )
        return DocCheatsheetResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cheatsheet generation failed: {e}")


@app.post("/docs/debug", response_model=DocDebugResponse, tags=["documenter"])
def docs_debug(payload: DocDebugRequest):
    """Debug code using uploaded documentation as grounding context."""
    try:
        result = _services.doc_agent().debug_code(
            user_id=payload.user_id,
            code=payload.code,
            error_message=payload.error_message or "",
            language=payload.language.value,
            doc_id=payload.doc_id,
        )
        return DocDebugResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {e}")


@app.post("/docs/interview", response_model=DocInterviewResponse, tags=["documenter"])
def docs_interview(payload: DocInterviewRequest):
    """Generate interview-style questions from uploaded docs."""
    try:
        result = _services.doc_agent().generate_interview(
            user_id=payload.user_id,
            topic=payload.topic,
            difficulty=payload.difficulty.value,
            doc_id=payload.doc_id,
        )
        return DocInterviewResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interview generation failed: {e}")
