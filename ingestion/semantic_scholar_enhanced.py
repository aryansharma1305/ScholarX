"""Enhanced Semantic Scholar API integration with full API capabilities."""
import requests
import time
from typing import List, Dict, Optional
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def paper_autocomplete(query: str, limit: int = 10) -> List[Dict]:
    """
    Suggest paper query completions for interactive search.
    
    Args:
        query: Partial query string (max 100 chars)
        limit: Maximum number of suggestions
        
    Returns:
        List of paper suggestions with minimal info
    """
    try:
        url = f"{settings.semantic_scholar_base_url}/paper/autocomplete"
        params = {
            "query": query[:100]  # Truncate to 100 chars
        }
        
        logger.info(f"Getting autocomplete suggestions for: {query[:50]}")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 429:
            logger.warning("Semantic Scholar rate limit (autocomplete)")
            return []
        
        response.raise_for_status()
        data = response.json()
        
        matches = data.get("matches", [])[:limit]
        results = []
        for match in matches:
            results.append({
                "paper_id": match.get("paperId"),
                "title": match.get("title", "Unknown"),
                "year": match.get("year"),
                "authors": match.get("authors", [])
            })
        
        logger.info(f"Found {len(results)} autocomplete suggestions")
        return results
        
    except Exception as e:
        logger.error(f"Autocomplete error: {e}")
        return []


def batch_get_papers(paper_ids: List[str], fields: str = "title,authors,abstract,year,openAccessPdf") -> List[Dict]:
    """
    Get details for multiple papers at once (up to 500).
    
    Args:
        paper_ids: List of paper IDs (supports various formats)
        fields: Comma-separated list of fields to return
        
    Returns:
        List of paper dictionaries
    """
    try:
        url = f"{settings.semantic_scholar_base_url}/paper/batch"
        params = {"fields": fields}
        
        # Split into batches of 500 (API limit)
        all_results = []
        for i in range(0, len(paper_ids), 500):
            batch = paper_ids[i:i+500]
            
            logger.info(f"Fetching batch {i//500 + 1} ({len(batch)} papers)")
            response = requests.post(url, params=params, json={"ids": batch}, timeout=30)
            
            if response.status_code == 429:
                logger.warning("Rate limited, waiting 5 seconds...")
                time.sleep(5)
                continue
            
            response.raise_for_status()
            batch_results = response.json()
            all_results.extend(batch_results)
            
            # Be respectful of rate limits
            time.sleep(0.1)
        
        logger.info(f"Fetched {len(all_results)} papers via batch API")
        return all_results
        
    except Exception as e:
        logger.error(f"Batch fetch error: {e}")
        return []


def search_papers_enhanced(
    query: str,
    limit: int = 10,
    fields: str = "title,authors,abstract,year,openAccessPdf,paperId,url,citationCount,referenceCount",
    year: Optional[str] = None,
    fields_of_study: Optional[List[str]] = None,
    open_access_only: bool = False,
    min_citation_count: Optional[int] = None,
    venue: Optional[str] = None
) -> Dict:
    """
    Enhanced paper search with filters.
    
    Args:
        query: Search query
        limit: Max results (<= 100)
        fields: Fields to return
        year: Year filter (e.g., "2020" or "2016-2020")
        fields_of_study: List of fields (e.g., ["Computer Science", "Physics"])
        open_access_only: Only open access papers
        min_citation_count: Minimum citations
        venue: Venue filter
        
    Returns:
        Dictionary with total, offset, next, and data
    """
    try:
        url = f"{settings.semantic_scholar_base_url}/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),  # API max is 100
            "fields": fields
        }
        
        # Add filters
        if year:
            params["year"] = year
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if open_access_only:
            params["openAccessPdf"] = ""
        if min_citation_count:
            params["minCitationCount"] = str(min_citation_count)
        if venue:
            params["venue"] = venue
        
        logger.info(f"Searching Semantic Scholar: {query} (filters: {params})")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 429:
            logger.warning("Semantic Scholar rate limit reached")
            return {"total": 0, "offset": 0, "next": 0, "data": []}
        
        response.raise_for_status()
        data = response.json()
        
        # Process results
        papers = data.get("data", [])
        results = []
        for paper in papers:
            authors = [author.get("name", "") for author in paper.get("authors", []) if author.get("name")]
            pdf_url = None
            if paper.get("openAccessPdf"):
                pdf_url = paper["openAccessPdf"].get("url")
            
            results.append({
                "paper_id": paper.get("paperId"),
                "title": paper.get("title", "Untitled"),
                "authors": authors,
                "authors_string": ", ".join(authors) if authors else "Unknown",
                "abstract": paper.get("abstract"),
                "year": paper.get("year"),
                "pdf_url": pdf_url,
                "url": paper.get("url"),
                "citation_count": paper.get("citationCount", 0),
                "reference_count": paper.get("referenceCount", 0),
                "source": "semantic_scholar"
            })
        
        logger.info(f"Found {len(results)} papers (total: {data.get('total', 0)})")
        return {
            "total": data.get("total", 0),
            "offset": data.get("offset", 0),
            "next": data.get("next", 0),
            "data": results
        }
        
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        return {"total": 0, "offset": 0, "next": 0, "data": []}


def get_paper_details(paper_id: str, fields: str = "title,authors,abstract,year,openAccessPdf,citationCount,referenceCount,citations,references") -> Optional[Dict]:
    """
    Get detailed information about a paper.
    
    Args:
        paper_id: Paper ID (supports various formats)
        fields: Fields to return
        
    Returns:
        Paper details dictionary
    """
    try:
        url = f"{settings.semantic_scholar_base_url}/paper/{paper_id}"
        params = {"fields": fields}
        
        logger.info(f"Fetching paper details: {paper_id}")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 429:
            logger.warning("Rate limited")
            return None
        
        if response.status_code == 404:
            logger.warning(f"Paper not found: {paper_id}")
            return None
        
        response.raise_for_status()
        data = response.json()
        
        # Extract citations and references
        citations = []
        for citation in data.get("citations", [])[:100]:  # Limit to 100
            citing_paper = citation.get("citingPaper", {})
            citations.append({
                "paper_id": citing_paper.get("paperId"),
                "title": citing_paper.get("title"),
                "year": citing_paper.get("year")
            })
        
        references = []
        for ref in data.get("references", [])[:100]:  # Limit to 100
            cited_paper = ref.get("citedPaper", {})
            references.append({
                "paper_id": cited_paper.get("paperId"),
                "title": cited_paper.get("title"),
                "year": cited_paper.get("year")
            })
        
        authors = [author.get("name", "") for author in data.get("authors", []) if author.get("name")]
        
        result = {
            "paper_id": data.get("paperId"),
            "title": data.get("title", "Unknown"),
            "authors": authors,
            "authors_string": ", ".join(authors) if authors else "Unknown",
            "abstract": data.get("abstract"),
            "year": data.get("year"),
            "pdf_url": data.get("openAccessPdf", {}).get("url") if data.get("openAccessPdf") else None,
            "url": data.get("url"),
            "citation_count": data.get("citationCount", 0),
            "reference_count": data.get("referenceCount", 0),
            "venue": data.get("venue"),
            "fields_of_study": data.get("fieldsOfStudy", []),
            "citations": citations,
            "references": references,
            "source": "semantic_scholar"
        }
        
        logger.info(f"Fetched details for: {result['title'][:50]}")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching paper details: {e}")
        return None


def get_paper_citations(paper_id: str, limit: int = 100, offset: int = 0) -> Dict:
    """
    Get papers that cite this paper.
    
    Args:
        paper_id: Paper ID
        limit: Max results (<= 1000)
        offset: Pagination offset
        
    Returns:
        Dictionary with citations data
    """
    try:
        url = f"{settings.semantic_scholar_base_url}/paper/{paper_id}/citations"
        params = {
            "limit": min(limit, 1000),
            "offset": offset,
            "fields": "title,year,authors,abstract"
        }
        
        logger.info(f"Fetching citations for: {paper_id}")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 429:
            logger.warning("Rate limited")
            return {"offset": 0, "next": 0, "data": []}
        
        response.raise_for_status()
        data = response.json()
        
        citations = []
        for citation in data.get("data", []):
            citing_paper = citation.get("citingPaper", {})
            citations.append({
                "paper_id": citing_paper.get("paperId"),
                "title": citing_paper.get("title"),
                "year": citing_paper.get("year"),
                "authors": [a.get("name", "") for a in citing_paper.get("authors", [])],
                "abstract": citing_paper.get("abstract"),
                "is_influential": citation.get("isInfluential", False)
            })
        
        return {
            "offset": data.get("offset", 0),
            "next": data.get("next", 0),
            "data": citations
        }
        
    except Exception as e:
        logger.error(f"Error fetching citations: {e}")
        return {"offset": 0, "next": 0, "data": []}


def get_paper_references(paper_id: str, limit: int = 100, offset: int = 0) -> Dict:
    """
    Get papers referenced by this paper.
    
    Args:
        paper_id: Paper ID
        limit: Max results (<= 1000)
        offset: Pagination offset
        
    Returns:
        Dictionary with references data
    """
    try:
        url = f"{settings.semantic_scholar_base_url}/paper/{paper_id}/references"
        params = {
            "limit": min(limit, 1000),
            "offset": offset,
            "fields": "title,year,authors,abstract"
        }
        
        logger.info(f"Fetching references for: {paper_id}")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 429:
            logger.warning("Rate limited")
            return {"offset": 0, "next": 0, "data": []}
        
        response.raise_for_status()
        data = response.json()
        
        references = []
        for ref in data.get("data", []):
            cited_paper = ref.get("citedPaper", {})
            references.append({
                "paper_id": cited_paper.get("paperId"),
                "title": cited_paper.get("title"),
                "year": cited_paper.get("year"),
                "authors": [a.get("name", "") for a in cited_paper.get("authors", [])],
                "abstract": cited_paper.get("abstract"),
                "is_influential": ref.get("isInfluential", False)
            })
        
        return {
            "offset": data.get("offset", 0),
            "next": data.get("next", 0),
            "data": references
        }
        
    except Exception as e:
        logger.error(f"Error fetching references: {e}")
        return {"offset": 0, "next": 0, "data": []}


def search_authors(query: str, limit: int = 10) -> Dict:
    """
    Search for authors by name.
    
    Args:
        query: Author name search query
        limit: Max results (<= 1000)
        
    Returns:
        Dictionary with authors data
    """
    try:
        url = f"{settings.semantic_scholar_base_url}/author/search"
        params = {
            "query": query,
            "limit": min(limit, 1000),
            "fields": "name,paperCount,citationCount,hIndex,affiliations"
        }
        
        logger.info(f"Searching authors: {query}")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 429:
            logger.warning("Rate limited")
            return {"total": 0, "offset": 0, "next": 0, "data": []}
        
        response.raise_for_status()
        data = response.json()
        
        authors = []
        for author in data.get("data", []):
            authors.append({
                "author_id": author.get("authorId"),
                "name": author.get("name"),
                "paper_count": author.get("paperCount", 0),
                "citation_count": author.get("citationCount", 0),
                "h_index": author.get("hIndex", 0),
                "affiliations": author.get("affiliations", [])
            })
        
        return {
            "total": data.get("total", 0),
            "offset": data.get("offset", 0),
            "next": data.get("next", 0),
            "data": authors
        }
        
    except Exception as e:
        logger.error(f"Author search error: {e}")
        return {"total": 0, "offset": 0, "next": 0, "data": []}


def search_snippets(query: str, limit: int = 10, paper_ids: Optional[List[str]] = None) -> List[Dict]:
    """
    Search for text snippets within papers.
    
    Args:
        query: Text query
        limit: Max results (<= 1000)
        paper_ids: Optional list of paper IDs to search within
        
    Returns:
        List of snippet matches
    """
    try:
        url = f"{settings.semantic_scholar_base_url}/snippet/search"
        params = {
            "query": query,
            "limit": min(limit, 1000)
        }
        
        if paper_ids:
            params["paperIds"] = ",".join(paper_ids[:100])  # Max 100 IDs
        
        logger.info(f"Searching snippets: {query[:50]}")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 429:
            logger.warning("Rate limited")
            return []
        
        response.raise_for_status()
        data = response.json()
        
        snippets = []
        for item in data.get("data", []):
            snippet_data = item.get("snippet", {})
            paper_data = item.get("paper", {})
            
            snippets.append({
                "text": snippet_data.get("text"),
                "score": item.get("score", 0),
                "paper_id": paper_data.get("corpusId"),
                "paper_title": paper_data.get("title"),
                "section": snippet_data.get("section"),
                "snippet_kind": snippet_data.get("snippetKind")
            })
        
        logger.info(f"Found {len(snippets)} snippet matches")
        return snippets
        
    except Exception as e:
        logger.error(f"Snippet search error: {e}")
        return []

