"""Hybrid search combining semantic and keyword matching."""
from typing import List
from vectorstore.query import QueryResult, query_vectors
from processing.embeddings import generate_embedding
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def simple_keyword_match(query: str, text: str) -> float:
    """
    Improved keyword matching score with normalization.
    
    Args:
        query: Query string
        text: Text to match against
        
    Returns:
        Keyword match score (0-1)
    """
    if not query or not text:
        return 0.0
    
    # Normalize: lowercase, remove punctuation, split on whitespace
    import re
    query_normalized = re.sub(r'[^\w\s]', ' ', query.lower())
    text_normalized = re.sub(r'[^\w\s]', ' ', text.lower())
    
    query_words = set(word for word in query_normalized.split() if len(word) > 2)  # Ignore very short words
    text_words = set(word for word in text_normalized.split() if len(word) > 2)
    
    if not query_words:
        return 0.0
    
    # Calculate overlap (Jaccard similarity)
    intersection = query_words.intersection(text_words)
    union = query_words.union(text_words)
    
    if not union:
        return 0.0
    
    # Jaccard similarity: intersection / union
    jaccard_score = len(intersection) / len(union)
    
    # Also consider how many query terms matched (precision)
    precision_score = len(intersection) / len(query_words)
    
    # Combined score (weighted average)
    score = (0.6 * jaccard_score) + (0.4 * precision_score)
    
    return min(1.0, score)


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
    # Normalize weights (guard against zero division)
    total_weight = semantic_weight + keyword_weight
    if total_weight == 0:
        # Default to semantic-only if both weights are zero
        logger.warning("Both weights are zero, defaulting to semantic-only search")
        semantic_weight = 1.0
        keyword_weight = 0.0
        total_weight = 1.0
    else:
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
        # Safe metadata spreading: handle None metadata
        base_metadata = result.metadata if result.metadata is not None else {}
        hybrid_result = QueryResult(
            chunk_id=result.chunk_id,
            paper_id=result.paper_id,
            chunk_index=result.chunk_index,
            text=result.text,
            score=combined_score,
            metadata={
                **base_metadata,
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

