"""Hybrid search combining semantic and keyword matching."""
from typing import List
from vectorstore.query import QueryResult, query_vectors
from processing.embeddings import generate_embedding
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def simple_keyword_match(query: str, text: str) -> float:
    """
    Simple keyword matching score (BM25-like).
    
    Args:
        query: Query string
        text: Text to match against
        
    Returns:
        Keyword match score (0-1)
    """
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    
    if not query_words:
        return 0.0
    
    # Calculate overlap
    overlap = len(query_words.intersection(text_words))
    score = overlap / len(query_words)
    
    return score


def hybrid_search(
    query: str,
    top_k: int = 10,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[QueryResult]:
    """
    Perform hybrid search combining semantic and keyword matching.
    
    Args:
        query: Search query
        top_k: Number of results to return
        semantic_weight: Weight for semantic similarity (0-1)
        keyword_weight: Weight for keyword matching (0-1)
        
    Returns:
        List of QueryResult objects sorted by combined score
    """
    # Normalize weights
    total_weight = semantic_weight + keyword_weight
    semantic_weight = semantic_weight / total_weight
    keyword_weight = keyword_weight / total_weight
    
    logger.info(f"Performing hybrid search: semantic={semantic_weight:.2f}, keyword={keyword_weight:.2f}")
    
    # Step 1: Semantic search (get more results than needed for re-ranking)
    query_embedding = generate_embedding(query)
    semantic_results = query_vectors(
        query_embedding=query_embedding,
        top_k=top_k * 2  # Get more for re-ranking
    )
    
    # Step 2: Calculate keyword scores and combine
    hybrid_results = []
    for result in semantic_results:
        # Get keyword match score
        keyword_score = simple_keyword_match(query, result.text)
        
        # Combine scores
        combined_score = (semantic_weight * result.score) + (keyword_weight * keyword_score)
        
        # Create new result with combined score
        hybrid_result = QueryResult(
            chunk_id=result.chunk_id,
            paper_id=result.paper_id,
            chunk_index=result.chunk_index,
            text=result.text,
            score=combined_score,
            metadata={
                **result.metadata,
                "semantic_score": result.score,
                "keyword_score": keyword_score,
                "combined_score": combined_score
            }
        )
        hybrid_results.append(hybrid_result)
    
    # Step 3: Re-sort by combined score
    hybrid_results.sort(key=lambda x: x.score, reverse=True)
    
    # Step 4: Return top_k
    final_results = hybrid_results[:top_k]
    
    logger.info(f"Hybrid search returned {len(final_results)} results")
    return final_results

