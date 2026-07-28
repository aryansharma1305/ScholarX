"""RAG answer generation using OpenAI, Grok (xAI), Ollama, or simple template."""
import re
from typing import List, Dict, Any
from dataclasses import dataclass, field
from config.settings import settings
from vectorstore.query import QueryResult
from utils.logger import get_logger

logger = get_logger(__name__)

# Lazy loading
_openai_client = None
_grok_client = None


def _get_openai_client():
    """Lazy load OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from config.openai_client import get_openai_client
        _openai_client = get_openai_client()
    return _openai_client


def _get_grok_client():
    """Lazy load Grok client (OpenAI-compatible, xAI endpoint)."""
    global _grok_client
    if _grok_client is None:
        from openai import OpenAI
        _grok_client = OpenAI(
            api_key=settings.grok_api_key,
            base_url="https://api.x.ai/v1",
        )
    return _grok_client


@dataclass
class RAGResponse:
    """Structured RAG response."""
    answer: str
    citations: List[Dict[str, Any]]
    context_chunks: List[Dict[str, Any]]
    query: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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


def _generate_with_grok(query: str, context_text: str, system_prompt: str) -> str:
    """Generate answer using Grok (xAI) — OpenAI-compatible API."""
    client = _get_grok_client()
    response = client.chat.completions.create(
        model=settings.grok_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n\n{context_text}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        temperature=0.7,
        max_tokens=1024
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
    """Build a conservative extractive answer when no chat model is available."""
    stop_words = {
        "about", "from", "have", "into", "that", "their", "these", "this",
        "what", "when", "where", "which", "with", "would",
    }
    query_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) > 2 and token not in stop_words
    }
    candidates = []
    for context_index, chunk in enumerate(context_chunks, 1):
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text.replace("\n", " "))
        for sentence_index, sentence in enumerate(sentences):
            clean_sentence = " ".join(sentence.split()).strip()
            if len(clean_sentence) < 30:
                continue
            sentence_terms = set(re.findall(r"[a-z0-9]+", clean_sentence.lower()))
            matched_terms = len(query_terms.intersection(sentence_terms))
            if query_terms and matched_terms == 0:
                continue
            coverage = matched_terms / len(query_terms) if query_terms else 0.0
            score = coverage + (0.2 * float(chunk.score)) - (0.01 * sentence_index)
            candidates.append((score, context_index, chunk, clean_sentence))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    seen_sentences = set()
    for _, context_index, chunk, sentence in candidates:
        normalized = sentence.lower()
        if normalized in seen_sentences:
            continue
        selected.append((context_index, chunk, sentence))
        seen_sentences.add(normalized)
        if len(selected) >= 3:
            break

    if not selected:
        return (
            "The retrieved papers do not contain a sufficiently direct passage to "
            f"answer '{query}' without an LLM. Try a more specific query or fetch "
            "additional papers."
        )

    answer_parts = [
        f"Most relevant evidence found for '{query}':",
        "",
    ]
    for context_index, chunk, sentence in selected:
        answer_parts.append(f"[Context {context_index}] {sentence}")
        answer_parts.append(
            f"(Source: Paper {chunk.paper_id}, Chunk {chunk.chunk_index})"
        )

    answer_parts.append("")
    answer_parts.append(
        "This is an extractive fallback response; configure a working LLM "
        "provider for a synthesized literature-review answer."
    )
    return "\n\n".join(answer_parts)


def generate_answer(
    query: str,
    context_chunks: List[QueryResult],
    system_prompt: str = None,
    force_simple: bool = False,
) -> RAGResponse:
    """
    Generate RAG answer using configured LLM provider.
    
    Args:
        query: User query
        context_chunks: Retrieved context chunks
        system_prompt: Custom system prompt (optional)
        force_simple: Skip external model calls and use extractive generation
        
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
    provider = "simple" if force_simple else settings.llm_provider
    logger.info(f"Generating answer using {provider}")
    
    answer = None
    
    if force_simple:
        answer = _generate_simple_answer(query, context_chunks)

    elif settings.llm_provider == "grok":
        try:
            answer = _generate_with_grok(query, context_text, system_prompt)
        except Exception as e:
            logger.error(f"Grok generation failed: {e}")
            logger.info("Falling back to simple template")
            answer = _generate_simple_answer(query, context_chunks)

    elif settings.llm_provider == "openai":
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
