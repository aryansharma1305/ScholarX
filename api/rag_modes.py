"""Advanced RAG modes for different use cases."""
from typing import Dict, List
from rag.pipeline import run_rag_pipeline
from rag.generator import generate_answer, RAGResponse
from vectorstore.query import QueryResult
from utils.logger import get_logger

logger = get_logger(__name__)


def rag_query_concise(query: str, top_k: int = 3) -> Dict:
    """RAG query in concise mode - short, direct answers."""
    system_prompt = (
        "You are a research assistant. Provide a concise, direct answer "
        "in 2-3 sentences. Use only the provided context."
    )
    
    response = run_rag_pipeline(
        query=query,
        top_k=top_k,
        system_prompt=system_prompt
    )
    
    return {
        "mode": "concise",
        "query": response.query,
        "answer": response.answer,
        "citations": response.citations
    }


def rag_query_detailed(query: str, top_k: int = 8) -> Dict:
    """RAG query in detailed mode - comprehensive answers."""
    system_prompt = (
        "You are a research assistant. Provide a detailed, comprehensive answer "
        "with explanations. Include multiple perspectives from different papers. "
        "Use the provided context extensively."
    )
    
    response = run_rag_pipeline(
        query=query,
        top_k=top_k,
        system_prompt=system_prompt
    )
    
    return {
        "mode": "detailed",
        "query": response.query,
        "answer": response.answer,
        "citations": response.citations,
        "context_chunks": response.context_chunks
    }


def rag_query_explain_simple(query: str, top_k: int = 5) -> Dict:
    """RAG query in 'explain like I'm 5' mode."""
    system_prompt = (
        "You are a research assistant explaining complex concepts simply. "
        "Explain the answer in simple terms, avoiding jargon. "
        "Use analogies when helpful. Make it accessible to non-experts."
    )
    
    response = run_rag_pipeline(
        query=query,
        top_k=top_k,
        system_prompt=system_prompt
    )
    
    return {
        "mode": "explain_simple",
        "query": response.query,
        "answer": response.answer,
        "citations": response.citations
    }


def rag_query_compare(query: str, top_k: int = 10) -> Dict:
    """RAG query in compare mode - compare multiple papers."""
    system_prompt = (
        "You are a research assistant. Compare and contrast different approaches "
        "or findings from the provided papers. Highlight similarities, differences, "
        "and relative strengths/weaknesses. Organize by paper or by theme."
    )
    
    response = run_rag_pipeline(
        query=query,
        top_k=top_k,
        system_prompt=system_prompt
    )
    
    # Group citations by paper
    papers = {}
    for citation in response.citations:
        pid = citation["paper_id"]
        if pid not in papers:
            papers[pid] = []
        papers[pid].append(citation)
    
    return {
        "mode": "compare",
        "query": response.query,
        "answer": response.answer,
        "citations_by_paper": papers,
        "all_citations": response.citations
    }


def rag_query_literature_survey(topic: str, top_k: int = 15) -> Dict:
    """Generate a literature survey on a topic."""
    query = f"Provide a comprehensive literature survey on {topic}"
    
    system_prompt = (
        "You are a research assistant writing a literature survey. "
        "Organize the answer into sections: Introduction, Key Approaches, "
        "Recent Advances, Challenges, and Future Directions. "
        "Cite papers appropriately throughout."
    )
    
    response = run_rag_pipeline(
        query=query,
        top_k=top_k,
        system_prompt=system_prompt
    )
    
    return {
        "mode": "literature_survey",
        "topic": topic,
        "survey": response.answer,
        "cited_papers": response.citations,
        "total_papers_cited": len(set(c["paper_id"] for c in response.citations))
    }


def rag_query_multi_document(papers: List[str], query: str, top_k: int = 5) -> Dict:
    """
    RAG query across multiple specific papers.
    
    Args:
        papers: List of paper IDs to search within
        query: User query
        top_k: Chunks per paper
    """
    from vectorstore.query import query_vectors
    from processing.embeddings import generate_embedding
    
    query_embedding = generate_embedding(query)
    
    # Search within each paper
    all_chunks = []
    for paper_id in papers:
        chunks = query_vectors(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata={"paper_id": paper_id}
        )
        all_chunks.extend(chunks)
    
    if not all_chunks:
        return {
            "mode": "multi_document",
            "query": query,
            "answer": "No relevant content found in specified papers.",
            "citations": []
        }
    
    # Generate answer from all chunks
    response = generate_answer(
        query=query,
        context_chunks=all_chunks,
        system_prompt=(
            "You are a research assistant. Synthesize information from multiple papers. "
            "Highlight agreements, disagreements, and complementary findings. "
            "Cite each paper clearly."
        )
    )
    
    return {
        "mode": "multi_document",
        "query": query,
        "papers_searched": papers,
        "answer": response.answer,
        "citations": response.citations,
        "papers_used": list(set(c["paper_id"] for c in response.citations))
    }



