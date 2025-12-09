"""Similarity checking and plagiarism detection."""
from typing import Dict, List
from processing.embeddings import generate_embedding
from vectorstore.query import query_vectors
from config.chroma_client import get_collection
from utils.logger import get_logger

logger = get_logger(__name__)


def compare_papers(paper_id1: str, paper_id2: str) -> Dict:
    """
    Compare two papers for similarity.
    
    Returns:
        Similarity score and similar chunks
    """
    collection = get_collection()
    
    # Get representative chunks from each paper
    chunks1 = collection.get(
        where={"paper_id": paper_id1},
        limit=5
    )
    chunks2 = collection.get(
        where={"paper_id": paper_id2},
        limit=5
    )
    
    if not chunks1.get("ids") or not chunks2.get("ids"):
        return {
            "paper1": paper_id1,
            "paper2": paper_id2,
            "similarity": 0.0,
            "error": "One or both papers not found"
        }
    
    # Get text from first chunks
    text1 = " ".join([doc[:500] for doc in chunks1.get("documents", [])[:3]])
    text2 = " ".join([doc[:500] for doc in chunks2.get("documents", [])[:3]])
    
    # Generate embeddings
    from processing.embeddings import generate_embedding
    emb1 = generate_embedding(text1)
    emb2 = generate_embedding(text2)
    
    # Calculate cosine similarity
    import numpy as np
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    return {
        "paper1": paper_id1,
        "paper2": paper_id2,
        "similarity": float(similarity),
        "similarity_percent": f"{similarity * 100:.2f}%"
    }


def check_text_similarity(text: str, threshold: float = 0.8) -> List[Dict]:
    """
    Check if text is similar to any paper in collection.
    
    Useful for plagiarism detection or finding related work.
    
    Args:
        text: Text to check
        threshold: Minimum similarity threshold
        
    Returns:
        List of similar papers with scores
    """
    # Generate embedding for input text
    query_embedding = generate_embedding(text)
    
    # Search for similar chunks
    similar_chunks = query_vectors(query_embedding, top_k=20)
    
    # Group by paper and calculate average similarity
    paper_scores = {}
    for chunk in similar_chunks:
        pid = chunk.paper_id
        if pid not in paper_scores:
            paper_scores[pid] = {
                "paper_id": pid,
                "scores": [],
                "chunks": []
            }
        paper_scores[pid]["scores"].append(chunk.score)
        paper_scores[pid]["chunks"].append({
            "chunk_index": chunk.chunk_index,
            "text": chunk.text[:200],
            "score": chunk.score
        })
    
    # Calculate average scores
    results = []
    for pid, data in paper_scores.items():
        avg_score = sum(data["scores"]) / len(data["scores"])
        if avg_score >= threshold:
            results.append({
                "paper_id": pid,
                "similarity_score": avg_score,
                "similarity_percent": f"{avg_score * 100:.2f}%",
                "matching_chunks": len(data["chunks"]),
                "top_chunks": data["chunks"][:3]
            })
    
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return results

