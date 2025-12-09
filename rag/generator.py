"""RAG answer generation using OpenAI, Ollama, or simple template."""
from typing import List, Dict, Any
from dataclasses import dataclass
from config.settings import settings
from vectorstore.query import QueryResult
from utils.logger import get_logger

logger = get_logger(__name__)

# Lazy loading
_openai_client = None


def _get_openai_client():
    """Lazy load OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from config.openai_client import client
        _openai_client = client
    return _openai_client


@dataclass
class RAGResponse:
    """Structured RAG response."""
    answer: str
    citations: List[Dict[str, Any]]
    context_chunks: List[Dict[str, Any]]
    query: str


def format_context_for_prompt(results: List[QueryResult]) -> str:
    """Format context chunks for prompt (helper function)."""
    context_parts = []
    
    for idx, result in enumerate(results, 1):
        context_parts.append(
            f"[Context {idx} - Paper ID: {result.paper_id}, Chunk {result.chunk_index}]\n"
            f"{result.text}\n"
        )
    
    return "\n---\n\n".join(context_parts)


def _generate_with_openai(query: str, context_text: str, system_prompt: str) -> str:
    """Generate answer using OpenAI."""
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n\n{context_text}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content or "No answer generated"


def _generate_with_ollama(query: str, context_text: str, system_prompt: str) -> str:
    """Generate answer using Ollama (local LLM)."""
    import requests
    
    prompt = f"{system_prompt}\n\nContext:\n\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
    
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No answer generated")
    except Exception as e:
        logger.warning(f"Ollama request failed: {e}. Falling back to simple template.")
        return None


def _generate_simple_answer(query: str, context_chunks: List[QueryResult]) -> str:
    """Generate simple template-based answer without LLM."""
    # Extract relevant sentences from top chunks
    answer_parts = []
    answer_parts.append(f"Based on the research papers, here's what I found about '{query}':\n\n")
    
    for idx, chunk in enumerate(context_chunks[:3], 1):  # Top 3 chunks
        # Extract first few sentences
        sentences = chunk.text.split('. ')[:2]  # First 2 sentences
        text_snippet = '. '.join(sentences)
        if text_snippet:
            answer_parts.append(f"[Context {idx}] {text_snippet}...")
            answer_parts.append(f"(Source: Paper {chunk.paper_id}, Chunk {chunk.chunk_index})\n")
    
    answer_parts.append("\nFor more details, please refer to the cited papers and their full context.")
    
    return "\n".join(answer_parts)


def generate_answer(
    query: str,
    context_chunks: List[QueryResult],
    system_prompt: str = None
) -> RAGResponse:
    """
    Generate RAG answer using configured LLM provider.
    
    Args:
        query: User query
        context_chunks: Retrieved context chunks
        system_prompt: Custom system prompt (optional)
        
    Returns:
        RAGResponse with answer, citations, and context
    """
    if not context_chunks:
        raise ValueError("No context chunks provided for answer generation")
    
    # Default system prompt
    if system_prompt is None:
        system_prompt = (
            "You are a helpful research assistant. "
            "Use only the provided context from research papers to answer the user's question. "
            "Cite chunks/papers clearly by referencing the context number and paper ID. "
            "If the context doesn't contain enough information to answer the question, say so explicitly. "
            "Be concise and accurate."
        )
    
    # Format context for prompt
    context_text = format_context_for_prompt(context_chunks)
    
    # Generate answer based on provider
    logger.info(f"Generating answer using {settings.llm_provider}")
    
    answer = None
    
    if settings.llm_provider == "openai":
        try:
            answer = _generate_with_openai(query, context_text, system_prompt)
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            logger.info("Falling back to simple template")
            answer = _generate_simple_answer(query, context_chunks)
    
    elif settings.llm_provider == "ollama":
        answer = _generate_with_ollama(query, context_text, system_prompt)
        if not answer:
            answer = _generate_simple_answer(query, context_chunks)
    
    else:  # "simple" or fallback
        answer = _generate_simple_answer(query, context_chunks)
    
    # Build citations from context chunks
    citations = [
        {
            "paper_id": chunk.paper_id,
            "chunk_index": chunk.chunk_index,
            "score": chunk.score
        }
        for chunk in context_chunks
    ]
    
    # Build context chunks info
    context_info = [
        {
            "paper_id": chunk.paper_id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "score": chunk.score
        }
        for chunk in context_chunks
    ]
    
    logger.info("Successfully generated RAG answer")
    
    # Log query
    try:
        from api.query_logger import log_query
        papers_used = list(set(chunk.paper_id for chunk in context_chunks))
        log_query(
            query=query,
            answer=answer,
            papers_used=papers_used,
            chunks_retrieved=len(context_chunks),
            model_used=settings.llm_provider,
            mode="default"
        )
    except Exception as e:
        logger.debug(f"Query logging failed: {e}")
    
    return RAGResponse(
        answer=answer,
        citations=citations,
        context_chunks=context_info,
        query=query
    )
