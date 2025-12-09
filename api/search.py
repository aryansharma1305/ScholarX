"""Google Scholar-like search interface."""
from typing import List, Dict, Optional
from vectorstore.query import QueryResult, query_vectors
from config.chroma_client import get_collection
from processing.embeddings import generate_embedding
from rag.search_enhanced import search_by_author, search_by_year
from utils.logger import get_logger

logger = get_logger(__name__)


def search_papers(
    query: Optional[str] = None,
    author: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 10
) -> Dict:
    """
    Google Scholar-like search interface.
    
    Args:
        query: Topic/keyword search query
        author: Author name filter
        year: Publication year filter
        limit: Maximum results
        
    Returns:
        Dictionary with papers, total count, and metadata
    """
    collection = get_collection()
    papers = {}
    
    # If author filter is specified
    if author:
        author_results = search_by_author(author, limit=limit * 2)
        for paper in author_results:
            papers[paper["paper_id"]] = paper
    
    # If year filter is specified
    if year:
        year_results = search_by_year(year, limit=limit * 2)
        for paper in year_results:
            pid = paper["paper_id"]
            if pid not in papers:
                papers[pid] = paper
            elif author and author.lower() in paper.get("authors", "").lower():
                papers[pid] = paper
    
    # If query is specified, do semantic search
    if query:
        query_embedding = generate_embedding(query)
        query_results = query_vectors(query_embedding, top_k=limit * 3)
        
        # Get unique papers from results
        for result in query_results:
            pid = result.paper_id
            if pid not in papers:
                # Get full paper metadata
                paper_chunks = collection.get(
                    where={"paper_id": pid},
                    limit=1
                )
                if paper_chunks.get("metadatas"):
                    meta = paper_chunks["metadatas"][0]
                    papers[pid] = {
                        "paper_id": pid,
                        "title": meta.get("title", "Unknown"),
                        "authors": meta.get("authors", "Unknown"),
                        "year": meta.get("year"),
                        "abstract": meta.get("abstract", "")[:200],
                        "score": result.score,
                        "source": meta.get("source", "unknown")
                    }
    
    # If no filters, get all papers
    if not query and not author and not year:
        all_data = collection.get(limit=limit * 10)
        for metadata in all_data.get("metadatas", []):
            pid = metadata.get("paper_id", "unknown")
            if pid not in papers:
                papers[pid] = {
                    "paper_id": pid,
                    "title": metadata.get("title", "Unknown"),
                    "authors": metadata.get("authors", "Unknown"),
                    "year": metadata.get("year"),
                    "abstract": metadata.get("abstract", "")[:200],
                    "source": metadata.get("source", "unknown")
                }
    
    # Apply filters
    filtered_papers = []
    for pid, paper in papers.items():
        # Apply author filter if specified
        if author and author.lower() not in paper.get("authors", "").lower():
            continue
        
        # Apply year filter if specified
        if year and paper.get("year") != year:
            continue
        
        filtered_papers.append(paper)
    
    # Sort by score if available, else by year (newest first)
    filtered_papers.sort(
        key=lambda x: (x.get("score", 0), x.get("year", 0) or 0),
        reverse=True
    )
    
    results = filtered_papers[:limit]
    
    logger.info(f"Search returned {len(results)} papers (query={query}, author={author}, year={year})")
    
    return {
        "papers": results,
        "total": len(results),
        "query": query,
        "filters": {
            "author": author,
            "year": year
        }
    }


def compare_papers_side_by_side(paper_ids: List[str]) -> Dict:
    """
    Compare multiple papers side-by-side (like Google Scholar).
    
    Args:
        paper_ids: List of paper IDs to compare
        
    Returns:
        Comparison data with metadata, citations, and similarities
    """
    from api.main_api import api
    
    collection = get_collection()
    comparison = {
        "papers": [],
        "comparison_metrics": {}
    }
    
    for paper_id in paper_ids:
        paper = api.get_paper(paper_id)
        if paper:
            # Get additional metrics
            citations = api.get_citations(paper_id)
            ranking = api.get_paper_ranking(paper_id)
            
            comparison["papers"].append({
                "paper_id": paper_id,
                "title": paper.get("title", "Unknown"),
                "authors": paper.get("authors", "Unknown"),
                "year": paper.get("year"),
                "abstract": paper.get("abstract", "")[:300],
                "chunk_count": paper.get("chunk_count", 0),
                "related_papers_count": len(citations.get("related_papers", [])),
                "citation_score": ranking.get("citation_score", 0),
                "rank": ranking.get("rank", "N/A")
            })
    
    # Calculate similarities between papers
    if len(comparison["papers"]) >= 2:
        similarities = []
        for i in range(len(comparison["papers"])):
            for j in range(i + 1, len(comparison["papers"])):
                pid1 = comparison["papers"][i]["paper_id"]
                pid2 = comparison["papers"][j]["paper_id"]
                
                sim_result = api.compare_two_papers(pid1, pid2)
                similarities.append({
                    "paper1": pid1,
                    "paper2": pid2,
                    "similarity": sim_result.get("similarity", 0.0),
                    "similarity_percent": sim_result.get("similarity_percent", "0%")
                })
        
        comparison["comparison_metrics"]["similarities"] = similarities
    
    return comparison


def get_paper_details_scholar_style(paper_id: str) -> Dict:
    """
    Get paper details in Google Scholar format.
    
    Returns:
        Paper details with citations, related papers, etc.
    """
    from api.main_api import api
    
    paper = api.get_paper(paper_id)
    if not paper:
        return {}
    
    citations = api.get_citations(paper_id)
    related = api.get_related_papers(paper_id, limit=10)
    ranking = api.get_paper_ranking(paper_id)
    summary = api.generate_summary(paper_id, use_llm=False)
    
    return {
        "paper_id": paper_id,
        "title": paper.get("title", "Unknown"),
        "authors": paper.get("authors", "Unknown"),
        "year": paper.get("year"),
        "abstract": paper.get("abstract", ""),
        "pdf_url": paper.get("pdf_url", ""),
        "source": paper.get("source", "unknown"),
        "doi": paper.get("doi", ""),
        "arxiv_id": paper.get("arxiv_id", ""),
        "metrics": {
            "chunks": paper.get("chunk_count", 0),
            "related_papers": len(related),
            "citation_score": ranking.get("citation_score", 0),
            "rank": ranking.get("rank", "N/A")
        },
        "citations": {
            "related": related,
            "total_related": len(related)
        },
        "summary": {
            "short": summary.get("short", ""),
            "bullets": summary.get("bullets", [])
        },
        "top_chunks": paper.get("top_chunks", [])[:5]
    }

