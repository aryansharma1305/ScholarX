"""Semantic Scholar API integration."""
import requests
import time
from typing import Dict, Optional
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Semantic Scholar API key (optional but recommended)
SEMANTIC_SCHOLAR_API_KEY = settings.semantic_scholar_api_key if settings.semantic_scholar_api_key else None

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


def fetch_paper_metadata(paper_id: str) -> Dict:
    """
    Fetch paper metadata from Semantic Scholar API.
    
    Args:
        paper_id: Semantic Scholar paper ID
        
    Returns:
        Dictionary with paper metadata (title, authors, abstract, pdf_url, etc.)
    """
    url = f"{settings.semantic_scholar_base_url}/paper/{paper_id}"
    params = {
        "fields": "title,authors,abstract,year,openAccessPdf,url"
    }
    
    # Prepare headers with API key if available
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    else:
        logger.warning("No Semantic Scholar API key found. Using unauthenticated requests (may hit rate limits)")
    
    # Retry logic with exponential backoff
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Fetching paper metadata for ID: {paper_id} (attempt {attempt + 1}/{MAX_RETRIES})")
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", RETRY_DELAY * (2 ** attempt)))
                logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            
            response.raise_for_status()
            break  # Success, exit retry loop
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed")
                raise
    
    try:
        
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
    url = f"{settings.semantic_scholar_base_url}/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,abstract,year,openAccessPdf,paperId"
    }
    
    # Prepare headers with API key if available
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    else:
        logger.warning("No Semantic Scholar API key found. Using unauthenticated requests (may hit rate limits)")
    
    # Retry logic with exponential backoff
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Searching Semantic Scholar for: {query} (attempt {attempt + 1}/{MAX_RETRIES})")
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", RETRY_DELAY * (2 ** attempt)))
                logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            
            response.raise_for_status()
            break  # Success, exit retry loop
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed")
                raise
    
    # Process successful response
    try:
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

