"""Relevance ranking for search results."""
from typing import List, Dict, Optional
from processing.embeddings import generate_embedding
from vectorstore.query import query_vectors
from config.chroma_client import get_collection
from utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)


def calculate_relevance_score(
    query: str,
    paper: Dict,
    use_semantic: bool = True,
    use_keyword: bool = True
) -> Dict:
    """
    Calculate comprehensive relevance score for a paper.
    
    Args:
        query: Search query
        paper: Paper dictionary with title, abstract, etc.
        use_semantic: Use semantic similarity
        use_keyword: Use keyword matching
        
    Returns:
        Dictionary with relevance score, percentage, and breakdown
    """
    scores = {
        "semantic_score": 0.0,
        "keyword_score": 0.0,
        "title_score": 0.0,
        "abstract_score": 0.0,
        "combined_score": 0.0,
        "relevance_percent": "0%"
    }
    
    if not query:
        return scores
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Get paper text
    title = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()
    full_text = f"{title} {abstract}"
    
    # 1. Keyword matching (title + abstract)
    if use_keyword:
        title_words = set(title.split())
        abstract_words = set(abstract.split())
        
        # Title match (weighted higher)
        title_overlap = len(query_words & title_words)
        title_score = title_overlap / len(query_words) if query_words else 0.0
        
        # Abstract match
        abstract_overlap = len(query_words & abstract_words)
        abstract_score = abstract_overlap / len(query_words) if query_words else 0.0
        
        # Combined keyword score (title weighted 2x)
        keyword_score = (title_score * 0.6 + abstract_score * 0.4)
        scores["keyword_score"] = min(keyword_score, 1.0)
        scores["title_score"] = min(title_score, 1.0)
        scores["abstract_score"] = min(abstract_score, 1.0)
    
    # 2. Semantic similarity (if paper is in vector store)
    if use_semantic:
        try:
            collection = get_collection()
            paper_id = paper.get("paper_id")
            
            if paper_id:
                # Check if paper is in collection
                chunks = collection.get(
                    where={"paper_id": paper_id},
                    limit=1
                )
                
                if chunks.get("ids"):
                    # Paper is in vector store - use semantic search
                    query_embedding = generate_embedding(query)
                    results = query_vectors(
                        query_embedding=query_embedding,
                        top_k=5,
                        filter_metadata={"paper_id": paper_id}
                    )
                    
                    if results:
                        # Get max score from chunks
                        semantic_score = max([r.score for r in results])
                        scores["semantic_score"] = float(semantic_score)
                    else:
                        # Paper in store but no matching chunks - use embedding similarity
                        # Get paper text embedding
                        paper_text = f"{title} {abstract[:500]}"
                        paper_embedding = generate_embedding(paper_text)
                        query_embedding = generate_embedding(query)
                        
                        # Cosine similarity
                        similarity = np.dot(paper_embedding, query_embedding) / (
                            np.linalg.norm(paper_embedding) * np.linalg.norm(query_embedding)
                        )
                        scores["semantic_score"] = max(0.0, float(similarity))
                else:
                    # Paper not in vector store - calculate embedding similarity directly
                    paper_text = f"{title} {abstract[:500]}"
                    paper_embedding = generate_embedding(paper_text)
                    query_embedding = generate_embedding(query)
                    
                    # Cosine similarity
                    similarity = np.dot(paper_embedding, query_embedding) / (
                        np.linalg.norm(paper_embedding) * np.linalg.norm(query_embedding)
                    )
                    scores["semantic_score"] = max(0.0, float(similarity))
        except Exception as e:
            logger.warning(f"Error calculating semantic score: {e}")
            # Fallback to keyword only
            pass
    
    # 3. Combine scores
    # Weight: 50% semantic, 30% keyword, 20% title match
    if use_semantic and use_keyword:
        combined = (
            0.5 * scores["semantic_score"] +
            0.3 * scores["keyword_score"] +
            0.2 * scores["title_score"]
        )
    elif use_semantic:
        combined = scores["semantic_score"]
    elif use_keyword:
        combined = scores["keyword_score"]
    else:
        combined = 0.0
    
    scores["combined_score"] = min(combined, 1.0)
    scores["relevance_percent"] = f"{scores['combined_score'] * 100:.1f}%"
    
    return scores


def rank_papers_by_relevance(
    query: str,
    papers: List[Dict],
    use_semantic: bool = True,
    use_keyword: bool = True
) -> List[Dict]:
    """
    Rank papers by relevance to query.
    
    Args:
        query: Search query
        papers: List of paper dictionaries
        use_semantic: Use semantic similarity
        use_keyword: Use keyword matching
        
    Returns:
        List of papers with relevance scores, sorted by relevance
    """
    if not query:
        return papers
    
    logger.info(f"Ranking {len(papers)} papers by relevance to: {query}")
    
    # Calculate relevance for each paper
    ranked_papers = []
    for paper in papers:
        relevance = calculate_relevance_score(
            query=query,
            paper=paper,
            use_semantic=use_semantic,
            use_keyword=use_keyword
        )
        
        # Add relevance scores to paper
        paper_with_score = {
            **paper,
            "relevance_score": relevance["combined_score"],
            "relevance_percent": relevance["relevance_percent"],
            "relevance_breakdown": {
                "semantic": relevance["semantic_score"],
                "keyword": relevance["keyword_score"],
                "title_match": relevance["title_score"]
            }
        }
        ranked_papers.append(paper_with_score)
    
    # Sort by relevance score (descending)
    ranked_papers.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
    
    logger.info(f"Ranked papers - Top 3 scores: {[p.get('relevance_percent') for p in ranked_papers[:3]]}")
    
    return ranked_papers


def get_relevance_category(score: float) -> Dict[str, str]:
    """
    Get relevance category and color based on score.
    
    Args:
        score: Relevance score (0-1)
        
    Returns:
        Dictionary with category, color, and emoji
    """
    if score >= 0.8:
        return {
            "category": "Excellent Match",
            "color": "#28a745",  # Green
            "emoji": "🟢",
            "badge": "success"
        }
    elif score >= 0.6:
        return {
            "category": "Good Match",
            "color": "#17a2b8",  # Blue
            "emoji": "🔵",
            "badge": "info"
        }
    elif score >= 0.4:
        return {
            "category": "Moderate Match",
            "color": "#ffc107",  # Yellow
            "emoji": "🟡",
            "badge": "warning"
        }
    elif score >= 0.2:
        return {
            "category": "Weak Match",
            "color": "#fd7e14",  # Orange
            "emoji": "🟠",
            "badge": "warning"
        }
    else:
        return {
            "category": "Poor Match",
            "color": "#dc3545",  # Red
            "emoji": "🔴",
            "badge": "danger"
        }

