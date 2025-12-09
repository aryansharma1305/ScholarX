"""RAG retrieval component."""
from typing import List
from processing.embeddings import generate_embedding
from vectorstore.query import query_vectors, QueryResult
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def retrieve_context(
    query: str,
    top_k: int = None
) -> List[QueryResult]:
    """
    Retrieve relevant context chunks for a query.
    
    Args:
        query: User query string
        top_k: Number of chunks to retrieve (defaults to settings)
        
    Returns:
        List of QueryResult objects sorted by relevance
    """
    top_k = top_k or settings.default_top_k
    
    logger.info(f"Retrieving context for query: {query[:50]}...")
    
    # Generate embedding for query
    query_embedding = generate_embedding(query)
    
    # Query Pinecone
    results = query_vectors(
        query_embedding=query_embedding,
        top_k=top_k
    )
    
    logger.info(f"Retrieved {len(results)} context chunks")
    return results


def format_context_for_prompt(results: List[QueryResult]) -> str:
    """
    Format retrieved context chunks for inclusion in LLM prompt.
    
    Args:
        results: List of QueryResult objects
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for idx, result in enumerate(results, 1):
        context_parts.append(
            f"[Context {idx} - Paper ID: {result.paper_id}, Chunk {result.chunk_index}]\n"
            f"{result.text}\n"
        )
    
    return "\n---\n\n".join(context_parts)

