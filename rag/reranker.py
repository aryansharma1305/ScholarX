"""Result re-ranking for better relevance."""
from typing import List
from vectorstore.query import QueryResult
from rag.quality_scorer import calculate_quality_score
from utils.logger import get_logger

logger = get_logger(__name__)


def rerank_results(
    results: List[QueryResult],
    paper_metadata_map: dict = None,
    diversity_weight: float = 0.1
) -> List[QueryResult]:
    """
    Re-rank results considering multiple factors.
    
    Args:
        results: Initial search results
        paper_metadata_map: Dictionary mapping paper_id to metadata
        diversity_weight: Weight for diversity (avoid all from same paper)
        
    Returns:
        Re-ranked results
    """
    if not results:
        return results
    
    paper_metadata_map = paper_metadata_map or {}
    
    # Calculate re-ranking scores
    reranked = []
    seen_papers = set()
    
    for result in results:
        # Base score from search
        base_score = result.score
        
        # Quality score from paper metadata
        paper_meta = paper_metadata_map.get(result.paper_id, {})
        quality_score = calculate_quality_score(paper_meta)
        
        # Diversity bonus (prefer papers we haven't seen much)
        diversity_bonus = 0.0
        if result.paper_id not in seen_papers:
            diversity_bonus = 0.1
        seen_papers.add(result.paper_id)
        
        # Combine scores
        # 60% semantic/keyword score, 30% quality, 10% diversity
        final_score = (
            0.6 * base_score +
            0.3 * quality_score +
            diversity_weight * diversity_bonus
        )
        
        # Create new result with final score
        reranked_result = QueryResult(
            chunk_id=result.chunk_id,
            paper_id=result.paper_id,
            chunk_index=result.chunk_index,
            text=result.text,
            score=final_score,
            metadata={
                **result.metadata,
                "base_score": base_score,
                "quality_score": quality_score,
                "final_score": final_score
            }
        )
        reranked.append(reranked_result)
    
    # Sort by final score
    reranked.sort(key=lambda x: x.score, reverse=True)
    
    logger.info(f"Re-ranked {len(reranked)} results")
    return reranked


def ensure_diversity(results: List[QueryResult], max_per_paper: int = 2) -> List[QueryResult]:
    """
    Ensure diversity by limiting chunks per paper.
    
    Args:
        results: Search results
        max_per_paper: Maximum chunks per paper
        
    Returns:
        Diversified results
    """
    paper_counts = {}
    diversified = []
    
    for result in results:
        paper_id = result.paper_id
        count = paper_counts.get(paper_id, 0)
        
        if count < max_per_paper:
            diversified.append(result)
            paper_counts[paper_id] = count + 1
    
    logger.info(f"Diversified results: {len(diversified)} from {len(results)} original")
    return diversified

