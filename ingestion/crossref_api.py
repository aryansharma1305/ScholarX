"""Crossref API integration for metadata retrieval."""
import requests
import time
from typing import List, Dict, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

CROSSREF_BASE_URL = "https://api.crossref.org"


def search_crossref(
    query: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    doi: Optional[str] = None,
    year: Optional[int] = None,
    rows: int = 10,
    offset: int = 0,
    filter_dict: Optional[Dict] = None
) -> Dict:
    """
    Search Crossref for works.
    
    Args:
        query: General search query
        title: Title search
        author: Author name search
        doi: DOI lookup
        year: Publication year
        rows: Number of results (max 1000)
        offset: Pagination offset
        filter_dict: Additional filters (e.g., {"has-full-text": "true"})
        
    Returns:
        Dictionary with search results
    """
    try:
        if doi:
            # Direct DOI lookup
            url = f"{CROSSREF_BASE_URL}/works/{doi}"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 404:
                logger.warning(f"DOI not found: {doi}")
                return {"total": 0, "items": []}
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok" and data.get("message"):
                return {
                    "total": 1,
                    "items": [data["message"]]
                }
            return {"total": 0, "items": []}
        
        # Search endpoint
        url = f"{CROSSREF_BASE_URL}/works"
        params = {
            "rows": min(rows, 1000),
            "offset": offset
        }
        
        # Build query
        if query:
            params["query"] = query
        if title:
            params["query.title"] = title
        if author:
            params["query.author"] = author
        if year:
            params["filter"] = f"from-pub-date:{year}"
        
        # Add filters
        if filter_dict:
            filter_parts = []
            for key, value in filter_dict.items():
                filter_parts.append(f"{key}:{value}")
            if params.get("filter"):
                params["filter"] += "," + ",".join(filter_parts)
            else:
                params["filter"] = ",".join(filter_parts)
        
        # Add polite pool header
        headers = {
            "User-Agent": "ScholarX/1.0 (mailto:your-email@example.com)"
        }
        
        logger.info(f"Searching Crossref: {params}")
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("status") != "ok":
            logger.error(f"Crossref API error: {data}")
            return {"total": 0, "items": []}
        
        message = data.get("message", {})
        items = message.get("items", [])
        total = message.get("total-results", 0)
        
        results = []
        for item in items:
            # Extract authors
            authors = []
            if item.get("author"):
                for author in item["author"]:
                    name_parts = []
                    if author.get("given"):
                        name_parts.append(author["given"])
                    if author.get("family"):
                        name_parts.append(author["family"])
                    authors.append(" ".join(name_parts))
            
            # Get PDF URL
            pdf_url = None
            if item.get("link"):
                for link in item["link"]:
                    if link.get("content-type") == "application/pdf":
                        pdf_url = link.get("URL")
                        break
            
            # Get DOI
            doi_value = item.get("DOI", "")
            
            # Get publication date
            pub_date = item.get("published-print") or item.get("published-online") or item.get("created")
            year_value = None
            if pub_date and pub_date.get("date-parts"):
                year_value = pub_date["date-parts"][0][0] if pub_date["date-parts"][0] else None
            
            results.append({
                "paper_id": doi_value,
                "title": " ".join(item.get("title", [])),
                "authors": authors,
                "authors_string": ", ".join(authors) if authors else "Unknown",
                "abstract": None,  # Crossref abstracts may be copyrighted
                "year": year_value,
                "pdf_url": pdf_url,
                "url": item.get("URL"),
                "doi": doi_value,
                "journal": item.get("container-title", [""])[0] if item.get("container-title") else None,
                "publisher": item.get("publisher"),
                "citation_count": item.get("is-referenced-by-count", 0),
                "source": "crossref"
            })
        
        logger.info(f"Found {len(results)} papers from Crossref (total: {total})")
        return {
            "total": total,
            "items": results
        }
        
    except Exception as e:
        logger.error(f"Crossref API error: {e}")
        return {"total": 0, "items": []}


def get_crossref_by_doi(doi: str) -> Optional[Dict]:
    """Get paper metadata by DOI."""
    result = search_crossref(doi=doi)
    if result.get("items"):
        return result["items"][0]
    return None

