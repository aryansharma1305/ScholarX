"""Citation-based ranking and paper metrics."""
from typing import List, Dict
from collections import defaultdict
from config.chroma_client import get_collection
from api.citations import find_citing_papers
from utils.logger import get_logger

logger = get_logger(__name__)


def calculate_citation_metrics() -> Dict:
    """
    Calculate citation-based metrics for papers.
    
    Returns:
        Dictionary with papers ranked by citations
    """
    collection = get_collection()
    count = collection.count()
    
    if count == 0:
        return {}
    
    all_data = collection.get(limit=count)
    
    # Get all unique papers
    papers = {}
    for metadata in all_data.get("metadatas", []):
        pid = metadata.get("paper_id", "unknown")
        if pid not in papers:
            papers[pid] = {
                "paper_id": pid,
                "title": metadata.get("title", "Unknown"),
                "year": metadata.get("year"),
                "incoming_citations": 0,
                "outgoing_citations": 0,  # Would need to parse references
                "citation_score": 0
            }
    
    # Calculate incoming citations (simplified - based on similarity)
    for paper_id in papers.keys():
        citing = find_citing_papers(paper_id, limit=20)
        papers[paper_id]["incoming_citations"] = len(citing)
        # Citation score = number of papers that cite it
        papers[paper_id]["citation_score"] = len(citing)
    
    # Rank by citation score
    ranked = sorted(
        papers.values(),
        key=lambda x: x["citation_score"],
        reverse=True
    )
    
    return {
        "total_papers": len(papers),
        "ranked_papers": ranked[:50]  # Top 50
    }


def get_paper_rank(paper_id: str) -> Dict:
    """
    Get ranking information for a specific paper.
    
    Returns:
        Paper rank, citation metrics, etc.
    """
    metrics = calculate_citation_metrics()
    
    paper_rank = None
    for i, paper in enumerate(metrics.get("ranked_papers", []), 1):
        if paper["paper_id"] == paper_id:
            paper_rank = {
                "paper_id": paper_id,
                "rank": i,
                "citation_score": paper["citation_score"],
                "incoming_citations": paper["incoming_citations"],
                "title": paper["title"]
            }
            break
    
    if not paper_rank:
        # Paper not in top rankings
        paper_rank = {
            "paper_id": paper_id,
            "rank": "Not in top 50",
            "citation_score": 0,
            "incoming_citations": 0
        }
    
    return paper_rank



