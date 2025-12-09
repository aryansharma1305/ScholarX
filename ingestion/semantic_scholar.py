"""Semantic Scholar API integration."""
import requests
from typing import Dict, Optional
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def fetch_paper_metadata(paper_id: str) -> Dict:
    """
    Fetch paper metadata from Semantic Scholar API.
    
    Args:
        paper_id: Semantic Scholar paper ID
        
    Returns:
        Dictionary with paper metadata (title, authors, abstract, pdf_url, etc.)
    """
    try:
        url = f"{settings.semantic_scholar_base_url}/paper/{paper_id}"
        params = {
            "fields": "title,authors,abstract,year,openAccessPdf,url"
        }
        
        logger.info(f"Fetching paper metadata for ID: {paper_id}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract authors
        authors = []
        if data.get("authors"):
            authors = [author.get("name", "") for author in data["authors"] if author.get("name")]
        
        # Get PDF URL
        pdf_url = None
        if data.get("openAccessPdf"):
            pdf_url = data["openAccessPdf"].get("url")
        
        result = {
            "paper_id": paper_id,
            "title": data.get("title", "Untitled Paper"),
            "authors": authors,
            "authors_string": ", ".join(authors) if authors else "Unknown Authors",
            "abstract": data.get("abstract"),
            "year": data.get("year"),
            "pdf_url": pdf_url,
            "url": data.get("url"),
        }
        
        logger.info(f"Successfully fetched metadata for: {result['title']}")
        return result
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch from Semantic Scholar: {e}")
        raise ValueError(f"Failed to fetch paper from Semantic Scholar: {e}")
    except Exception as e:
        logger.error(f"Error processing Semantic Scholar response: {e}")
        raise ValueError(f"Error processing Semantic Scholar response: {e}")


def search_papers(query: str, limit: int = 10) -> list:
    """
    Search for papers on Semantic Scholar.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of paper metadata dictionaries
    """
    try:
        url = f"{settings.semantic_scholar_base_url}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,abstract,year,openAccessPdf,paperId"
        }
        
        logger.info(f"Searching Semantic Scholar for: {query}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        papers = data.get("data", [])
        
        results = []
        for paper in papers:
            authors = [author.get("name", "") for author in paper.get("authors", []) if author.get("name")]
            pdf_url = None
            if paper.get("openAccessPdf"):
                pdf_url = paper["openAccessPdf"].get("url")
            
            results.append({
                "paper_id": paper.get("paperId"),
                "title": paper.get("title", "Untitled Paper"),
                "authors": authors,
                "authors_string": ", ".join(authors) if authors else "Unknown Authors",
                "abstract": paper.get("abstract"),
                "year": paper.get("year"),
                "pdf_url": pdf_url,
            })
        
        logger.info(f"Found {len(results)} papers")
        return results
        
    except requests.RequestException as e:
        logger.error(f"Failed to search Semantic Scholar: {e}")
        raise ValueError(f"Failed to search Semantic Scholar: {e}")

