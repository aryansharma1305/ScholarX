"""Paper recommendation system based on user queries and reading patterns."""
from typing import List, Dict, Optional
from vectorstore.query import query_vectors
from config.chroma_client import get_collection
from processing.embeddings import generate_embedding
from rag.hybrid_search import hybrid_search
# from api.query_logger import get_query_history  # Will implement if needed
from utils.logger import get_logger
import json
from pathlib import Path
from datetime import datetime

logger = get_logger(__name__)

# User reading history storage
HISTORY_FILE = Path(__file__).parent.parent / "user_history.json"


def load_user_history() -> Dict:
    """Load user reading history."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load user history: {e}")
    return {"viewed_papers": [], "queries": [], "interests": []}


def save_user_history(history: Dict):
    """Save user reading history."""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save user history: {e}")


def record_paper_view(paper_id: str, paper_title: str = ""):
    """Record that a user viewed a paper."""
    history = load_user_history()
    if paper_id not in history["viewed_papers"]:
        history["viewed_papers"].append({
            "paper_id": paper_id,
            "title": paper_title,
            "timestamp": datetime.now().isoformat()
        })
        save_user_history(history)


def record_query(query: str):
    """Record a user query."""
    history = load_user_history()
    history["queries"].append({
        "query": query,
        "timestamp": datetime.now().isoformat()
    })
    # Extract interests from queries
    if len(history["queries"]) > 0:
        # Simple keyword extraction
        words = query.lower().split()
        for word in words:
            if len(word) > 4 and word not in ["what", "how", "when", "where", "which", "about"]:
                if word not in history["interests"]:
                    history["interests"].append(word)
    save_user_history(history)


def recommend_papers_based_on_history(limit: int = 10) -> List[Dict]:
    """
    Recommend papers based on user's reading history and queries.
    
    Returns:
        List of recommended papers with relevance scores
    """
    history = load_user_history()
    collection = get_collection()
    recommendations = []
    
    # Strategy 1: Based on viewed papers (find similar papers)
    for viewed in history["viewed_papers"][-5:]:  # Last 5 viewed
        paper_id = viewed.get("paper_id")
        try:
            # Get paper metadata
            chunks = collection.get(where={"paper_id": paper_id}, limit=1)
            if chunks.get("metadatas"):
                meta = chunks["metadatas"][0]
                title = meta.get("title", "")
                abstract = meta.get("abstract", "")
                
                # Find similar papers
                if abstract:
                    query_text = f"{title} {abstract[:200]}"
                    query_embedding = generate_embedding(query_text)
                    similar = query_vectors(query_embedding, top_k=5)
                    
                    for result in similar:
                        if result.paper_id != paper_id:
                            recommendations.append({
                                "paper_id": result.paper_id,
                                "score": result.score * 0.7,  # Weight for similarity
                                "reason": f"Similar to '{title[:50]}...'",
                                "type": "similarity"
                            })
        except Exception as e:
            logger.warning(f"Error processing viewed paper {paper_id}: {e}")
    
    # Strategy 2: Based on query history (find papers matching interests)
    if history["queries"]:
        # Combine recent queries
        recent_queries = [q["query"] for q in history["queries"][-3:]]
        combined_query = " ".join(recent_queries)
        
        query_embedding = generate_embedding(combined_query)
        query_results = query_vectors(query_embedding, top_k=limit * 2)
        
        for result in query_results:
            recommendations.append({
                "paper_id": result.paper_id,
                "score": result.score * 0.8,  # Weight for query match
                "reason": f"Matches your recent queries",
                "type": "query_match"
            })
    
    # Strategy 3: Based on interests (keyword-based)
    if history["interests"]:
        interest_query = " ".join(history["interests"][-5:])
        query_embedding = generate_embedding(interest_query)
        interest_results = query_vectors(query_embedding, top_k=limit)
        
        for result in interest_results:
            recommendations.append({
                "paper_id": result.paper_id,
                "score": result.score * 0.6,  # Weight for interest match
                "reason": f"Matches your interests: {', '.join(history['interests'][-3:])}",
                "type": "interest_match"
            })
    
    # Deduplicate and rank
    paper_scores = {}
    for rec in recommendations:
        pid = rec["paper_id"]
        if pid not in paper_scores or rec["score"] > paper_scores[pid]["score"]:
            paper_scores[pid] = rec
    
    # Get full paper metadata
    final_recommendations = []
    for pid, rec in sorted(paper_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:limit]:
        try:
            chunks = collection.get(where={"paper_id": pid}, limit=1)
            if chunks.get("metadatas"):
                meta = chunks["metadatas"][0]
                final_recommendations.append({
                    "paper_id": pid,
                    "title": meta.get("title", "Unknown"),
                    "authors": meta.get("authors", "Unknown"),
                    "year": meta.get("year"),
                    "abstract": meta.get("abstract", "")[:200],
                    "relevance_score": rec["score"],
                    "recommendation_reason": rec["reason"],
                    "recommendation_type": rec["type"],
                    "source": meta.get("source", "unknown")
                })
        except Exception as e:
            logger.warning(f"Error getting metadata for {pid}: {e}")
    
    return final_recommendations


def recommend_papers_for_query(query: str, limit: int = 10) -> List[Dict]:
    """
    Recommend papers for a specific query using collaborative filtering.
    
    Args:
        query: User query
        limit: Number of recommendations
        
    Returns:
        List of recommended papers
    """
    # Use hybrid search for better results
    results = hybrid_search(query, top_k=limit * 2, semantic_weight=0.7, keyword_weight=0.3)
    
    collection = get_collection()
    recommendations = []
    
    for result in results[:limit]:
        try:
            chunks = collection.get(where={"paper_id": result.paper_id}, limit=1)
            if chunks.get("metadatas"):
                meta = chunks["metadatas"][0]
                recommendations.append({
                    "paper_id": result.paper_id,
                    "title": meta.get("title", "Unknown"),
                    "authors": meta.get("authors", "Unknown"),
                    "year": meta.get("year"),
                    "abstract": meta.get("abstract", "")[:200],
                    "relevance_score": result.score,
                    "semantic_score": result.metadata.get("semantic_score", 0) if result.metadata else 0,
                    "keyword_score": result.metadata.get("keyword_score", 0) if result.metadata else 0,
                    "source": meta.get("source", "unknown")
                })
        except Exception as e:
            logger.warning(f"Error getting metadata: {e}")
    
    return recommendations


def get_trending_topics(days: int = 30) -> List[Dict]:
    """
    Identify trending topics based on recent queries and paper views.
    
    Args:
        days: Number of days to look back
        
    Returns:
        List of trending topics with scores
    """
    history = load_user_history()
    # Use user history queries instead of query_logs
    
    # Extract topics from queries
    topic_counts = {}
    for query in history["queries"]:
        words = query["query"].lower().split()
        for word in words:
            if len(word) > 4:
                topic_counts[word] = topic_counts.get(word, 0) + 1
    
    # Sort by frequency
    trending = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return [
        {"topic": topic, "frequency": count, "trend_score": count / max(topic_counts.values()) if topic_counts else 0}
        for topic, count in trending
    ]

