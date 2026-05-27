"""Pydantic schemas for the LearnMate AI learning endpoints."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────

class LevelEnum(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"
    exam_focused = "exam-focused"
    auto = "auto"


class GoalEnum(str, Enum):
    conceptual = "conceptual"
    exam = "exam"
    interview = "interview"
    revision = "revision"


class StyleEnum(str, Enum):
    simple = "simple"
    analogy = "analogy"
    example_first = "example-first"
    formal = "formal"
    exam = "exam"


class CorrectnessEnum(str, Enum):
    incorrect = "incorrect"
    partially_correct = "partially_correct"
    correct = "correct"


class PaceDecisionEnum(str, Enum):
    slow_down = "slow_down"
    continue_ = "continue"
    increase_difficulty = "increase_difficulty"


# ── /learn ────────────────────────────────────────────────────────

class LearnRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Unique learner identifier")
    topic: str = Field(..., min_length=1, description="Concept the user wants to learn")
    level: LevelEnum = Field(default=LevelEnum.auto, description="Learner level")
    goal: GoalEnum = Field(default=GoalEnum.conceptual, description="Learning goal")
    context: Optional[str] = Field(default=None, description="Optional extra context")
    preferred_style: StyleEnum = Field(default=StyleEnum.simple, description="Preferred teaching style")


class LearnResponse(BaseModel):
    topic: str
    detected_level: str
    teaching_strategy: str
    explanation: str
    analogy: str
    real_life_example: str
    key_points: List[str]
    quick_check_question: str
    next_step: str
    retrieved_sources_used: bool


# ── /evaluate ─────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class EvaluateResponse(BaseModel):
    score: int = Field(ge=0, le=5)
    correctness: CorrectnessEnum
    feedback: str
    missing_points: List[str]
    misconception: str
    pace_decision: PaceDecisionEnum
    next_explanation_style: str
    next_question: str


# ── /profile/{user_id} ───────────────────────────────────────────

class LearnerProfileResponse(BaseModel):
    user_id: str
    preferred_style: str
    level: str
    current_topics: List[str]
    completed_topics: List[str]
    weak_topics: List[str]
    misconception_history: List[str]
    average_score: float
    pace: str
    last_interaction_at: Optional[str] = None


# ── /flashcards ───────────────────────────────────────────────────

class FlashcardRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    count: int = Field(default=5, ge=1, le=20)
    level: LevelEnum = Field(default=LevelEnum.beginner)


class Flashcard(BaseModel):
    front: str
    back: str


class FlashcardResponse(BaseModel):
    flashcards: List[Flashcard]


# ── /learning-path ───────────────────────────────────────────────

class LearningPathRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    goal: GoalEnum = Field(default=GoalEnum.conceptual)


class LearningPathStep(BaseModel):
    step: int
    subtopic: str
    why_it_matters: str
    estimated_difficulty: str


class LearningPathResponse(BaseModel):
    topic: str
    learning_path: List[LearningPathStep]


# ── /teach-from-docs ─────────────────────────────────────────────

class TeachFromDocsRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=2)
    level: LevelEnum = Field(default=LevelEnum.beginner)


# ── /upload-document ─────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    total_chars: int
    num_chunks: int
    chunks_added_to_knowledge_base: int
    preview: str = Field(description="First 500 characters of extracted text")


# ── /documents ───────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    total_chars: int
    num_chunks: int
    uploaded_at: str


class DocumentListResponse(BaseModel):
    total_documents: int
    documents: List[DocumentInfo]


# ── DocuMentor AI schemas (/docs/*) ──────────────────────────────


class ModeEnum(str, Enum):
    auto = "auto"
    explain = "explain"
    code_example = "code_example"
    build_with_me = "build_with_me"
    challenge = "challenge"
    quiz = "quiz"
    cheat_sheet = "cheat_sheet"
    debug = "debug"
    interview = "interview"
    general_qa = "general_qa"


class CodingLevelEnum(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"
    auto = "auto"


class LanguageEnum(str, Enum):
    python = "python"
    javascript = "javascript"
    java = "java"
    cpp = "cpp"
    auto = "auto"


class ResponseStatusEnum(str, Enum):
    answered = "answered"
    unsupported = "unsupported"
    error = "error"


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    text: str


class DocAskRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=2)
    doc_id: Optional[str] = None
    mode: ModeEnum = Field(default=ModeEnum.auto)
    coding_level: CodingLevelEnum = Field(default=CodingLevelEnum.auto)
    preferred_language: LanguageEnum = Field(default=LanguageEnum.python)
    preferred_style: str = Field(default="simple")
    code_context: Optional[str] = None


class DocAskResponse(BaseModel):
    mode: str
    detected_level: str
    answer: str
    status: ResponseStatusEnum
    citations: List[Citation]
    confidence: Optional[float] = None
    unsupported_reason: Optional[str] = None
    used_doc_ids: List[str]
    safety_flags: Optional[List[str]] = None
    code_example: str
    mini_challenge: str
    quick_check: str
    next_step: str


class DocChallengeRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    doc_id: Optional[str] = None
    difficulty: CodingLevelEnum = Field(default=CodingLevelEnum.intermediate)
    language: LanguageEnum = Field(default=LanguageEnum.python)


class DocChallengeResponse(BaseModel):
    challenge_title: str
    difficulty: str
    task: str
    requirements: List[str]
    hints: List[str]
    expected_behavior: str
    optional_solution: str
    status: ResponseStatusEnum
    citations: List[Citation]
    unsupported_reason: Optional[str] = None
    used_doc_ids: List[str]


class DocQuizRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    doc_id: Optional[str] = None
    difficulty: CodingLevelEnum = Field(default=CodingLevelEnum.intermediate)
    num_questions: int = Field(default=5, ge=1, le=20)


class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    citation: Optional[Citation] = None


class DocQuizResponse(BaseModel):
    questions: List[QuizQuestion]
    status: ResponseStatusEnum
    unsupported_reason: Optional[str] = None
    used_doc_ids: List[str]


class DocCheatsheetRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    doc_id: Optional[str] = None
    topic: Optional[str] = None


class DocCheatsheetResponse(BaseModel):
    title: str
    important_concepts: List[str]
    syntax: List[str]
    code_patterns: List[str]
    common_errors: List[str]
    best_practices: List[str]
    status: ResponseStatusEnum
    citations: List[Citation]
    unsupported_reason: Optional[str] = None
    used_doc_ids: List[str]


class DocDebugRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    doc_id: Optional[str] = None
    code: str = Field(..., min_length=1)
    error_message: Optional[str] = None
    language: LanguageEnum = Field(default=LanguageEnum.python)


class DocDebugResponse(BaseModel):
    issue_found: str
    why_it_happens: str
    corrected_code: str
    explanation: str
    prevention_tip: str
    status: ResponseStatusEnum
    citations: List[Citation]
    unsupported_reason: Optional[str] = None
    used_doc_ids: List[str]


class DocInterviewRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    doc_id: Optional[str] = None
    difficulty: CodingLevelEnum = Field(default=CodingLevelEnum.intermediate)


class InterviewQuestion(BaseModel):
    question: str
    model_answer: str
    difficulty: str


class DocInterviewResponse(BaseModel):
    questions: List[InterviewQuestion]
    status: ResponseStatusEnum
    citations: List[Citation]
    unsupported_reason: Optional[str] = None
    used_doc_ids: List[str]
