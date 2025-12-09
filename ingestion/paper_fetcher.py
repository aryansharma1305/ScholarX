"""Fetch papers from various APIs based on topic."""
import requests
import feedparser
from typing import List, Dict, Optional
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def search_semantic_scholar(query: str, limit: int = 5) -> List[Dict]:
    """
    Search for papers on Semantic Scholar.
    
    Args:
        query: Search query/topic
        limit: Maximum number of results
        
    Returns:
        List of paper dictionaries with metadata and PDF URLs
    """
    try:
        url = f"{settings.semantic_scholar_base_url}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,abstract,year,openAccessPdf,paperId,url"
        }
        
        logger.info(f"Searching Semantic Scholar for: {query}")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        papers = data.get("data", [])
        
        results = []
        for paper in papers:
            authors = [author.get("name", "") for author in paper.get("authors", []) if author.get("name")]
            pdf_url = None
            if paper.get("openAccessPdf"):
                pdf_url = paper["openAccessPdf"].get("url")
            
            if pdf_url:  # Only include papers with PDFs
                results.append({
                    "paper_id": paper.get("paperId"),
                    "title": paper.get("title", "Untitled Paper"),
                    "authors": authors,
                    "authors_string": ", ".join(authors) if authors else "Unknown Authors",
                    "abstract": paper.get("abstract"),
                    "year": paper.get("year"),
                    "pdf_url": pdf_url,
                    "url": paper.get("url"),
                    "source": "semantic_scholar"
                })
        
        logger.info(f"Found {len(results)} papers with PDFs from Semantic Scholar")
        return results
        
    except Exception as e:
        logger.error(f"Error searching Semantic Scholar: {e}")
        return []


def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search for papers on ArXiv.
    
    Args:
        query: Search query/topic
        max_results: Maximum number of results
        
    Returns:
        List of paper dictionaries with metadata and PDF URLs
    """
    try:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        logger.info(f"Searching ArXiv for: {query}")
        response = requests.get(settings.arxiv_base_url, params=params, timeout=15)
        response.raise_for_status()
        
        feed = feedparser.parse(response.content)
        
        results = []
        for entry in feed.entries:
            # ArXiv always has PDFs
            pdf_url = None
            for link in entry.links:
                if link.rel == "alternate" and link.type == "application/pdf":
                    pdf_url = link.href
                    break
            
            if not pdf_url:
                # Fallback: construct PDF URL from ArXiv ID
                arxiv_id = entry.id.split("/")[-1]
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Extract authors
            authors = [author.name for author in entry.get("authors", [])]
            
            results.append({
                "paper_id": entry.id.split("/")[-1],  # ArXiv ID
                "title": entry.title,
                "authors": authors,
                "authors_string": ", ".join(authors) if authors else "Unknown Authors",
                "abstract": entry.get("summary", ""),
                "year": entry.published_parsed.tm_year if entry.published_parsed else None,
                "pdf_url": pdf_url,
                "url": entry.link,
                "source": "arxiv"
            })
        
        logger.info(f"Found {len(results)} papers from ArXiv")
        return results
        
    except Exception as e:
        logger.error(f"Error searching ArXiv: {e}")
        return []


def fetch_papers_by_topic(topic: str, max_papers: int = None) -> List[Dict]:
    """
    Fetch papers from multiple sources based on a topic.
    
    Args:
        topic: Topic or query string
        max_papers: Maximum total papers to fetch
        
    Returns:
        List of paper dictionaries with PDF URLs
    """
    max_papers = max_papers or settings.max_papers_per_query
    
    logger.info(f"Fetching papers for topic: {topic}")
    
    all_papers = []
    
    # Try Semantic Scholar first (usually has better metadata)
    semantic_papers = search_semantic_scholar(topic, limit=max_papers)
    all_papers.extend(semantic_papers)
    
    # If we need more, try ArXiv
    if len(all_papers) < max_papers:
        remaining = max_papers - len(all_papers)
        arxiv_papers = search_arxiv(topic, max_results=remaining)
        all_papers.extend(arxiv_papers)
    
    # Remove duplicates (by title similarity)
    unique_papers = []
    seen_titles = set()
    for paper in all_papers:
        title_lower = paper["title"].lower()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_papers.append(paper)
    
    logger.info(f"Fetched {len(unique_papers)} unique papers for topic: {topic}")
    return unique_papers[:max_papers]

