import json
import os
from typing import Dict, Any, List, Optional
from .llm_provider import LLMProvider, get_llm_provider
from .retriever import Retriever
from .grounding import verify_grounding
from .utils_logger import get_logger

logger = get_logger(__name__)

UNSUPPORTED_MESSAGE = "The uploaded docs do not clearly mention this."


class QueryPlannerAgent:
    """Agent responsible for rewriting the user query to optimize vector database search."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def plan_query(self, question: str) -> str:
        prompt = (
            "You are an AI Query Planner. Your task is to rewrite the user's question into a single optimized search query "
            "suitable for retrieving highly relevant technical documentation chunks. "
            "Remove unnecessary conversational words and focus entirely on core technical concepts and keywords.\n"
            f"User Question: {question}\n"
            "Return only the optimized search query, with no prefixes or formatting.\n"
            "Optimized Query:"
        )
        try:
            rewritten = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=100, temperature=0.1)
            rewritten_clean = rewritten.strip()
            if rewritten_clean and len(rewritten_clean) > 3 and "Configuration Error" not in rewritten_clean:
                logger.info("QueryPlannerAgent rewritten query: '%s' -> '%s'", question, rewritten_clean)
                return rewritten_clean
        except Exception as e:
            logger.warning("QueryPlannerAgent failed: %s. Using original query.", e)
        return question


class RetrieverAgent:
    """Agent responsible for retrieving context chunks from the vector database, optionally scoped by doc_id."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def retrieve_chunks(self, query: str, top_k: int = 5, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            return self.retriever.retrieve(query, top_k=top_k, doc_id=doc_id)
        except Exception as e:
            logger.error("RetrieverAgent failed: %s", e)
            return []


class LearningAgent:
    """Agent responsible for conceptual explanations or simple grounded answers in student-friendly terms."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def generate_explanation(self, question: str, chunks: List[Dict[str, Any]], mode: str = "qa") -> str:
        context_text = "\n\n".join(f"[Chunk {i+1}]: {c['text']}" for i, c in enumerate(chunks))
        
        system_instruction = (
            "You are an expert Learning Assistant. Your goal is to explain technical concepts clearly and simply, "
            "making them accessible to students. Use simple analogies and clean markdown formatting where helpful.\n"
            "CRITICAL: Answer ONLY using the provided retrieved documentation chunks. Do not assume or extrapolate. "
            "If the context does not contain the answer, say exactly 'The uploaded docs do not clearly mention this.' and nothing else.\n"
            "If you can answer it, include citations in your answer referring to [Chunk 1], [Chunk 2], etc. where you derived your facts."
        )
        
        prompt = (
            f"Retrieved Documentation:\n{context_text}\n\n"
            f"User Question: {question}\n\n"
            f"Mode: {mode}\n"
            "Answer:"
        )

        return self.llm.chat(
            [{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.2
        )


class QuizAgent:
    """Agent responsible for creating structured multiple-choice quiz questions based on retrieved chunks."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def generate_quiz(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        context_text = "\n\n".join(f"[Chunk {i+1}]: {c['text']}" for i, c in enumerate(chunks))
        
        system_instruction = (
            "You are a Teacher AI. Create a high-quality multiple choice quiz based strictly on the retrieved documentation provided.\n"
            "Format the quiz in clean Markdown. Include 3 questions, each with options A, B, C, D, followed by the correct answer with brief explanation.\n"
            "CRITICAL: Base all questions and answers strictly on the provided context. If the context does not contain enough information to create a quiz about this topic, "
            "say exactly 'The uploaded docs do not clearly mention this.' and nothing else."
        )
        
        prompt = (
            f"Retrieved Documentation:\n{context_text}\n\n"
            f"Topic Context: {question}\n\n"
            "Quiz Questions:"
        )

        return self.llm.chat(
            [{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.2
        )


class InterviewPrepAgent:
    """Agent responsible for creating technical interview questions and model answers based on chunks."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def generate_prep(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        context_text = "\n\n".join(f"[Chunk {i+1}]: {c['text']}" for i, c in enumerate(chunks))
        
        system_instruction = (
            "You are a Technical Recruiter AI. Generate 3 professional interview-style questions along with detailed, "
            "rigorous model answers based strictly on the retrieved documentation provided.\n"
            "Format the output using clear Markdown headers.\n"
            "CRITICAL: Base everything strictly on the provided context. If the context does not contain enough details, "
            "say exactly 'The uploaded docs do not clearly mention this.' and nothing else."
        )
        
        prompt = (
            f"Retrieved Documentation:\n{context_text}\n\n"
            f"Topic Context: {question}\n\n"
            "Interview Prep Q&A:"
        )

        return self.llm.chat(
            [{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.2
        )


class GroundingAgent:
    """Agent responsible for checking if the generated answer is fully supported by the retrieved chunks.

    Combines a deterministic lexical check and LLM reflection to control hallucinations.
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def verify(self, answer: str, chunks: List[Dict[str, Any]]) -> bool:
        # 1. Deterministic lexical overlap check
        texts = [c["text"] for c in chunks]
        res = verify_grounding(answer, texts)
        if not res["is_grounded"]:
            logger.warning("Deterministic grounding check failed: %s", res["explanation"])
            return False

        # 2. LLM self-reflection validation
        context_text = "\n\n".join(f"[Chunk {i+1}]: {c['text']}" for i, c in enumerate(chunks))
        prompt = (
            "Analyze if the following generated answer is fully grounded in and supported by the retrieved documentation chunks. "
            "Answer 'yes' if it is supported and does not contain outside knowledge or hallucinations. "
            "Answer 'no' if it contains claims not found in the documentation.\n\n"
            f"Retrieved Documentation:\n{context_text}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            "Is the answer grounded? (yes/no):"
        )
        try:
            judgment = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=10, temperature=0.1)
            is_grounded = "yes" in judgment.strip().lower()
            logger.info("GroundingAgent judgment: %s", is_grounded)
            return is_grounded
        except Exception as e:
            logger.warning("GroundingAgent LLM check failed: %s. Relying on deterministic check.", e)
            return True


class MultiAgentPipeline:
    """Wires together the multi-agent pipeline and orchestrates the workflow."""

    def __init__(self, retriever: Retriever, llm: Optional[LLMProvider] = None):
        self.llm = llm or get_llm_provider()
        self.planner = QueryPlannerAgent(self.llm)
        self.retriever_agent = RetrieverAgent(retriever)
        self.grounding_agent = GroundingAgent(self.llm)
        self.learning_agent = LearningAgent(self.llm)
        self.quiz_agent = QuizAgent(self.llm)
        self.interview_agent = InterviewPrepAgent(self.llm)

    def ask(self, question: str, doc_id: Optional[str] = None, mode: str = "qa", top_k: int = 5) -> Dict[str, Any]:
        # Get provider metadata
        provider_name = getattr(self.llm, "model", "local")
        model_name = os.getenv("VERTEX_MODEL", os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.0-flash"))

        # Workflow: 1. Plan
        search_query = self.planner.plan_query(question)

        # Workflow: 2. Retrieve
        chunks = self.retriever_agent.retrieve_chunks(search_query, top_k=top_k, doc_id=doc_id)
        if not chunks:
            logger.info("No chunks retrieved for query '%s' and doc_id '%s'", search_query, doc_id)
            return {
                "answer": UNSUPPORTED_MESSAGE,
                "status": "unsupported",
                "citations": [],
                "retrieved_chunks_count": 0,
                "provider": provider_name,
                "model": model_name,
            }

        # Workflow: 3. Mode-specific generation
        if mode == "teach":
            answer = self.learning_agent.generate_explanation(question, chunks, mode="teach")
        elif mode == "quiz":
            answer = self.quiz_agent.generate_quiz(question, chunks)
        elif mode == "interview":
            answer = self.interview_agent.generate_prep(question, chunks)
        else:  # default "qa"
            answer = self.learning_agent.generate_explanation(question, chunks, mode="qa")

        # Handle raw config/provider errors
        if "Configuration Error" in answer or "unavailable" in answer:
            return {
                "answer": answer,
                "status": "error",
                "citations": [],
                "retrieved_chunks_count": len(chunks),
                "provider": provider_name,
                "model": model_name,
            }

        # Handle direct unsupported outputs
        if UNSUPPORTED_MESSAGE.lower() in answer.lower():
            return {
                "answer": UNSUPPORTED_MESSAGE,
                "status": "unsupported",
                "citations": [],
                "retrieved_chunks_count": len(chunks),
                "provider": provider_name,
                "model": model_name,
            }

        # Workflow: 4. Grounding Check
        is_grounded = self.grounding_agent.verify(answer, chunks)
        if not is_grounded:
            logger.warning("Answer failed grounding checks: downgrading to unsupported.")
            return {
                "answer": UNSUPPORTED_MESSAGE,
                "status": "unsupported",
                "citations": [],
                "retrieved_chunks_count": len(chunks),
                "provider": provider_name,
                "model": model_name,
            }

        # Workflow: 5. Build citations
        citations = []
        for i, c in enumerate(chunks, 1):
            # Try to match chunk reference in the answer text to only cite used chunks
            if f"[Chunk {i}]" in answer or mode in ("quiz", "interview") or len(chunks) <= 2:
                meta = c.get("index")  # retrieve returns index
                # Try to load file/chunk meta
                c_meta = self.retriever.store.metas[meta] if meta is not None and meta < len(self.retriever.store.metas) else {}
                
                citations.append({
                    "doc_id": c_meta.get("doc_id", c.get("doc_id", "")),
                    "filename": c_meta.get("filename", "unknown"),
                    "chunk_id": c_meta.get("chunk_id", f"chunk_{i}"),
                    "page_number": c_meta.get("page_number", i),
                    "snippet": c.get("text", "")[:250],
                })

        return {
            "answer": answer,
            "status": "answered",
            "citations": citations,
            "retrieved_chunks_count": len(chunks),
            "provider": provider_name,
            "model": model_name,
        }
