"""LangChain model abstraction and structured RAG quality checks."""
from __future__ import annotations

from functools import lru_cache
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)
_model_disabled_reason: Optional[str] = None


class EvidenceGrade(BaseModel):
    """Structured evidence sufficiency decision."""

    sufficient: bool = Field(description="Whether the evidence can answer the question")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class AnswerGrade(BaseModel):
    """Structured grounded-answer decision."""

    supported: bool = Field(description="Whether the answer is supported by the evidence")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


@lru_cache(maxsize=4)
def get_chat_model(provider: Optional[str] = None):
    """Create a LangChain chat model for the configured provider."""
    selected_provider = (provider or settings.llm_provider or "simple").lower()

    if selected_provider == "simple":
        return None

    if selected_provider == "openai":
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY is missing; using deterministic graph fallbacks")
            return None
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
            timeout=60,
            max_retries=3,
        )

    if selected_provider == "grok":
        if not settings.grok_api_key:
            logger.warning("GROK_API_KEY is missing; using deterministic graph fallbacks")
            return None
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.grok_model,
            api_key=settings.grok_api_key,
            base_url="https://api.x.ai/v1",
            temperature=0,
            timeout=60,
            max_retries=3,
        )

    if selected_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    logger.warning("Unsupported LangChain model provider: %s", selected_provider)
    return None


def _usable_chat_model():
    if _model_disabled_reason:
        return None
    return get_chat_model()


def model_calls_disabled() -> bool:
    """Return whether a permanent provider failure opened the circuit breaker."""
    return bool(_model_disabled_reason)


def _record_model_failure(exc: Exception) -> None:
    """Open a process-local circuit breaker for permanent provider failures."""
    global _model_disabled_reason
    message = str(exc)
    permanent_markers = (
        "401",
        "403",
        "authentication",
        "permission-denied",
        "no credits",
        "no licenses",
        "invalid api key",
    )
    if any(marker in message.lower() for marker in permanent_markers):
        _model_disabled_reason = message
        logger.warning(
            "Disabling LangChain model calls for this process after a permanent "
            "provider error; deterministic fallbacks remain active"
        )


def _context_text(context_chunks: List[Dict[str, Any]], limit: int = 12000) -> str:
    parts = []
    for index, chunk in enumerate(context_chunks, 1):
        parts.append(
            f"[Context {index}; paper={chunk.get('paper_id')}; "
            f"score={float(chunk.get('score', 0.0)):.3f}]\n"
            f"{str(chunk.get('text') or '')[:3000]}"
        )
    return "\n\n".join(parts)[:limit]


def deterministic_evidence_grade(
    context_chunks: List[Dict[str, Any]],
    query: str = "",
) -> Dict[str, Any]:
    """Grade evidence without an LLM so the graph also works in free mode."""
    scores = sorted(
        (float(chunk.get("score", 0.0)) for chunk in context_chunks),
        reverse=True,
    )
    best_score = scores[0] if scores else 0.0
    top_average = sum(scores[:3]) / min(len(scores), 3) if scores else 0.0
    enough_chunks = len(context_chunks) >= settings.langgraph_min_context_chunks
    query_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) > 2
        and token
        not in {"what", "when", "where", "which", "with", "from", "that", "this", "how"}
    }
    lexical_coverage = 1.0
    if query_terms:
        lexical_coverage = max(
            (
                len(query_terms.intersection(
                    re.findall(r"[a-z0-9]+", str(chunk.get("text") or "").lower())
                ))
                / len(query_terms)
                for chunk in context_chunks
            ),
            default=0.0,
        )
    sufficient = (
        enough_chunks
        and best_score >= settings.langgraph_min_relevance_score
        and top_average >= max(settings.langgraph_min_relevance_score - 0.15, 0.2)
        and lexical_coverage >= 0.5
    )
    return {
        "sufficient": sufficient,
        "confidence": min(
            max((best_score + top_average + lexical_coverage) / 3, 0.0),
            1.0,
        ),
        "reason": (
            f"{len(context_chunks)} chunks; best score {best_score:.3f}; "
            f"top-3 average {top_average:.3f}; "
            f"best query-term coverage {lexical_coverage:.2f}"
        ),
        "method": "deterministic",
        "lexical_coverage": lexical_coverage,
    }


def grade_evidence(
    query: str,
    context_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Use structured LLM grading when configured, otherwise deterministic grading."""
    fallback = deterministic_evidence_grade(context_chunks, query=query)
    if not settings.langgraph_use_llm_grader or not context_chunks:
        return fallback

    model = _usable_chat_model()
    if model is None:
        return fallback

    try:
        grader = model.with_structured_output(EvidenceGrade)
        result = grader.invoke([
            SystemMessage(
                content=(
                    "You grade research evidence. Mark evidence sufficient only when "
                    "it directly supports a careful answer to the question. Do not "
                    "reward mere keyword overlap."
                )
            ),
            HumanMessage(
                content=f"Question:\n{query}\n\nEvidence:\n{_context_text(context_chunks)}"
            ),
        ])
        return {
            "sufficient": result.sufficient,
            "confidence": result.confidence,
            "reason": result.reason,
            "method": "llm",
        }
    except Exception as exc:
        _record_model_failure(exc)
        logger.warning("LLM evidence grading failed, using deterministic grade: %s", exc)
        return {**fallback, "fallback_reason": str(exc)}


def generate_grounded_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
) -> Optional[str]:
    """Generate a grounded answer through the configured LangChain model."""
    model = _usable_chat_model()
    if model is None:
        return None

    prompt = system_prompt or (
        "You are a research assistant. Answer only from the supplied paper "
        "evidence. Cite claims using [Context N]. If evidence is insufficient, "
        "say exactly what is missing. Do not invent citations."
    )
    try:
        response = model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(
                content=f"Question:\n{query}\n\nPaper evidence:\n{_context_text(context_chunks)}"
            ),
        ])
        content = response.content
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()
    except Exception as exc:
        _record_model_failure(exc)
        logger.warning("LangChain generation failed, using existing generator: %s", exc)
        return None


def deterministic_answer_grade(
    answer: str,
    citations: List[Dict[str, Any]],
    context_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate citation integrity without making another model call."""
    context_paper_ids = {
        str(chunk.get("paper_id"))
        for chunk in context_chunks
        if chunk.get("paper_id")
    }
    cited_paper_ids = {
        str(citation.get("paper_id"))
        for citation in citations
        if citation.get("paper_id")
    }
    citations_valid = bool(cited_paper_ids) and cited_paper_ids.issubset(context_paper_ids)
    has_answer = bool(answer and len(answer.strip()) >= 40)
    supported = has_answer and citations_valid and bool(context_chunks)
    return {
        "supported": supported,
        "confidence": 0.8 if supported else 0.25,
        "reason": (
            "Answer has valid citations mapped to retrieved context."
            if supported
            else "Answer is missing usable evidence or contains invalid citations."
        ),
        "method": "deterministic",
    }


def validate_answer(
    query: str,
    answer: str,
    citations: List[Dict[str, Any]],
    context_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check whether the answer is supported, with deterministic fallback."""
    fallback = deterministic_answer_grade(answer, citations, context_chunks)
    if not settings.langgraph_use_llm_grader or not fallback["supported"]:
        return fallback

    model = _usable_chat_model()
    if model is None:
        return fallback

    try:
        grader = model.with_structured_output(AnswerGrade)
        result = grader.invoke([
            SystemMessage(
                content=(
                    "Check whether the answer is fully supported by the supplied "
                    "research evidence. Reject unsupported claims and fabricated "
                    "citations."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{query}\n\nAnswer:\n{answer}\n\n"
                    f"Evidence:\n{_context_text(context_chunks)}"
                )
            ),
        ])
        return {
            "supported": result.supported,
            "confidence": result.confidence,
            "reason": result.reason,
            "method": "llm",
        }
    except Exception as exc:
        _record_model_failure(exc)
        logger.warning("LLM answer validation failed, using deterministic grade: %s", exc)
        return {**fallback, "fallback_reason": str(exc)}


def rewrite_query_for_retry(
    query: str,
    evidence_reason: str,
) -> str:
    """Produce one improved retrieval query for the controlled retry."""
    model = _usable_chat_model()
    if model is None:
        return f"{query} methods results evaluation evidence"

    try:
        response = model.invoke([
            SystemMessage(
                content=(
                    "Rewrite the research query for scholarly retrieval. Return one "
                    "concise query only, with no explanation."
                )
            ),
            HumanMessage(
                content=(
                    f"Original query: {query}\n"
                    f"Why the previous evidence was weak: {evidence_reason}"
                )
            ),
        ])
        rewritten = str(response.content).strip()
        return rewritten or f"{query} methods results evaluation evidence"
    except Exception as exc:
        _record_model_failure(exc)
        logger.warning("Query rewrite failed, using deterministic retry query: %s", exc)
        return f"{query} methods results evaluation evidence"
