"""OpenAlex API integration for scholarly metadata."""
import requests
import time
from typing import List, Dict, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org"


def search_openalex(
    query: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    year: Optional[int] = None,
    per_page: int = 10,
    page: int = 1,
    filter_dict: Optional[Dict] = None
) -> Dict:
    """
    Search OpenAlex for works.
    
    Args:
        query: General search query
        title: Title search
        author: Author name search
        year: Publication year
        per_page: Results per page (max 200)
        page: Page number
        filter_dict: Additional filters
        
    Returns:
        Dictionary with search results
    """
    try:
        url = f"{OPENALEX_BASE_URL}/works"
        params = {
            "per-page": min(per_page, 200),
            "page": page
        }
        
        # Build search query
        if query:
            params["search"] = query
        elif title:
            params["search"] = f"title:{title}"
        elif author:
            params["search"] = f"author:{author}"
        
        # Add filters
        filters = []
        if year:
            filters.append(f"publication_year:{year}")
        if filter_dict:
            for key, value in filter_dict.items():
                filters.append(f"{key}:{value}")
        
        if filters:
            params["filter"] = ",".join(filters)
        
        logger.info(f"Searching OpenAlex: {params}")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        for item in data.get("results", []):
            # Extract authors
            authors = []
            if item.get("authorships"):
                for authorship in item["authorships"]:
                    author = authorship.get("author", {})
                    if author:
                        authors.append(author.get("display_name", "Unknown"))
            
            # Get PDF URL
            pdf_url = None
            if item.get("open_access") and item["open_access"].get("is_oa"):
                pdf_url = item["open_access"].get("oa_url")
            
            # Get primary location PDF
            if not pdf_url and item.get("primary_location"):
                loc = item["primary_location"]
                if loc.get("pdf_url"):
                    pdf_url = loc["pdf_url"]
            
            # Get concepts (fields/topics)
            concepts = []
            if item.get("concepts"):
                for concept in item["concepts"][:5]:  # Top 5
                    concepts.append(concept.get("display_name", ""))
            
            results.append({
                "paper_id": item.get("id", "").split("/")[-1] if item.get("id") else None,
                "title": item.get("title", "Unknown"),
                "authors": authors,
                "authors_string": ", ".join(authors) if authors else "Unknown",
                "abstract": item.get("abstract", ""),
                "year": item.get("publication_year"),
                "pdf_url": pdf_url,
                "url": item.get("doi") or item.get("id"),
                "doi": item.get("doi"),
                "citation_count": item.get("cited_by_count", 0),
                "concepts": concepts,
                "open_access": item.get("open_access", {}).get("is_oa", False),
                "venue": item.get("primary_location", {}).get("source", {}).get("display_name"),
                "source": "openalex"
            })
        
        meta = data.get("meta", {})
        total = meta.get("count", 0)
        
        logger.info(f"Found {len(results)} papers from OpenAlex (total: {total})")
        return {
            "total": total,
            "page": page,
            "per_page": per_page,
            "items": results
        }
        
    except Exception as e:
        logger.error(f"OpenAlex API error: {e}")
        return {"total": 0, "items": []}


def get_openalex_work(work_id: str) -> Optional[Dict]:
    """
    Get a single work by OpenAlex ID or DOI.
    
    Args:
        work_id: OpenAlex ID (W123456789) or DOI (10.1234/example)
        
    Returns:
        Work metadata dictionary
    """
    try:
        if work_id.startswith("10."):
            # DOI lookup
            url = f"{OPENALEX_BASE_URL}/works/doi:{work_id}"
        elif work_id.startswith("W"):
            # OpenAlex ID
            url = f"{OPENALEX_BASE_URL}/works/{work_id}"
        else:
            # Assume OpenAlex ID
            url = f"{OPENALEX_BASE_URL}/works/W{work_id}"
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        item = response.json()
        
        # Extract same format as search
        authors = []
        if item.get("authorships"):
            for authorship in item["authorships"]:
                author = authorship.get("author", {})
                if author:
                    authors.append(author.get("display_name", "Unknown"))
        
        pdf_url = None
        if item.get("open_access") and item["open_access"].get("is_oa"):
            pdf_url = item["open_access"].get("oa_url")
        
        concepts = []
        if item.get("concepts"):
            for concept in item["concepts"][:5]:
                concepts.append(concept.get("display_name", ""))
        
        return {
            "paper_id": item.get("id", "").split("/")[-1] if item.get("id") else None,
            "title": item.get("title", "Unknown"),
            "authors": authors,
            "authors_string": ", ".join(authors) if authors else "Unknown",
            "abstract": item.get("abstract", ""),
            "year": item.get("publication_year"),
            "pdf_url": pdf_url,
            "url": item.get("doi") or item.get("id"),
            "doi": item.get("doi"),
            "citation_count": item.get("cited_by_count", 0),
            "concepts": concepts,
            "open_access": item.get("open_access", {}).get("is_oa", False),
            "venue": item.get("primary_location", {}).get("source", {}).get("display_name"),
            "source": "openalex"
        }
        
    except Exception as e:
        logger.error(f"Error fetching OpenAlex work: {e}")
        return None



