"""Citation graph and related papers."""
from typing import List, Dict, Optional
from config.chroma_client import get_collection
from rag.search_enhanced import get_related_papers
from processing.embeddings import generate_embedding
from vectorstore.query import query_vectors
from utils.logger import get_logger

logger = get_logger(__name__)


def get_citation_info(paper_id: str) -> Dict:
    """
    Get citation information for a paper.
    
    Returns:
        Dictionary with outgoing and incoming citations
    """
    collection = get_collection()
    
    # Get paper metadata
    paper_chunks = collection.get(
        where={"paper_id": paper_id},
        limit=1
    )
    
    if not paper_chunks.get("ids"):
        return {
            "paper_id": paper_id,
            "outgoing_citations": [],
            "incoming_citations": [],
            "related_papers": []
        }
    
    metadata = paper_chunks["metadatas"][0] if paper_chunks.get("metadatas") else {}
    
    # Extract references from text (simple approach)
    # In a real system, you'd parse the references section
    outgoing_citations = []
    
    # Get related papers (papers that cite similar content)
    related = get_related_papers(paper_id, limit=10)
    
    # For incoming citations, we'd need to search for papers that mention this paper
    # This is a simplified version
    incoming_citations = []
    
    return {
        "paper_id": paper_id,
        "title": metadata.get("title", "Unknown"),
        "outgoing_citations": outgoing_citations,
        "incoming_citations": incoming_citations,
        "related_papers": [
            {
                "paper_id": rp["paper_id"],
                "similarity_score": rp["score"]
            }
            for rp in related
        ]
    }


def find_citing_papers(paper_id: str, limit: int = 10) -> List[Dict]:
    """
    Find papers that might cite this paper (by similarity).
    
    This is a simplified version - in production, you'd parse actual citations.
    """
    # Get paper embedding
    collection = get_collection()
    paper_chunks = collection.get(
        where={"paper_id": paper_id},
        limit=1
    )
    
    if not paper_chunks.get("ids"):
        return []
    
    # Use title/abstract for finding similar papers
    metadata = paper_chunks["metadatas"][0] if paper_chunks.get("metadatas") else {}
    title = metadata.get("title", "")
    abstract = metadata.get("abstract", "")
    
    # Generate embedding for title + abstract
    query_text = f"{title} {abstract[:500]}"
    query_embedding = generate_embedding(query_text)
    
    # Search for similar papers
    similar_chunks = query_vectors(query_embedding, top_k=limit * 3)
    
    # Get unique papers
    papers = {}
    for chunk in similar_chunks:
        pid = chunk.paper_id
        if pid != paper_id and pid not in papers:
            papers[pid] = {
                "paper_id": pid,
                "similarity_score": chunk.score,
                "title": chunk.metadata.get("title", "Unknown")
            }
    
    results = list(papers.values())[:limit]
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return results



