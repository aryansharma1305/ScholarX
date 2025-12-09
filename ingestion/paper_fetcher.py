"""Fetch papers from various APIs based on topic."""
import requests
import feedparser
from typing import List, Dict, Optional
from config.settings import settings
from utils.logger import get_logger
from ingestion.semantic_scholar_enhanced import search_papers_enhanced, paper_autocomplete
from ingestion.crossref_api import search_crossref
from ingestion.openalex_api import search_openalex

logger = get_logger(__name__)


def search_semantic_scholar(
    query: str, 
    limit: int = 5,
    year: Optional[str] = None,
    fields_of_study: Optional[List[str]] = None,
    open_access_only: bool = False,
    min_citation_count: Optional[int] = None
) -> List[Dict]:
    """
    Search for papers on Semantic Scholar.
    
    Args:
        query: Search query/topic
        limit: Maximum number of results
        
    Returns:
        List of paper dictionaries with metadata and PDF URLs
    """
    try:
        # Use enhanced search with filters
        search_result = search_papers_enhanced(
            query=query,
            limit=limit,
            year=year,
            fields_of_study=fields_of_study,
            open_access_only=open_access_only,
            min_citation_count=min_citation_count
        )
        
        results = search_result.get("data", [])
        
        # Filter to only papers with PDFs if open_access_only is False
        if not open_access_only:
            results = [p for p in results if p.get("pdf_url")]
        
        logger.info(f"Found {len(results)} papers from Semantic Scholar (total available: {search_result.get('total', 0)})")
        return results
        
    except requests.exceptions.Timeout:
        logger.warning("Semantic Scholar request timed out. Using ArXiv as fallback.")
        return []
    except requests.exceptions.RequestException as e:
        logger.warning(f"Semantic Scholar request failed: {e}. Using ArXiv as fallback.")
        return []
    except Exception as e:
        logger.error(f"Unexpected error searching Semantic Scholar: {e}")
        return []


def search_arxiv(
    query: str, 
    max_results: int = 5,
    field: Optional[str] = None,
    sort_by: str = "relevance",
    sort_order: str = "descending"
) -> List[Dict]:
    """
    Search for papers on ArXiv with enhanced capabilities.
    
    Args:
        query: Search query/topic (supports field prefixes: ti:, au:, abs:, etc.)
        max_results: Maximum number of results
        field: Field to search (ti, au, abs, co, jr, cat, rn, all)
        sort_by: Sort by "relevance", "lastUpdatedDate", or "submittedDate"
        sort_order: "ascending" or "descending"
        
    Returns:
        List of paper dictionaries with metadata and PDF URLs
    """
    try:
        from ingestion.arxiv_enhanced import search_arxiv_enhanced
        
        result = search_arxiv_enhanced(
            query=query,
            max_results=max_results,
            field=field,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        return result.get("entries", [])
        
    except Exception as e:
        logger.error(f"Error searching ArXiv: {e}")
        return []


def fetch_papers_by_topic(
    topic: str, 
    max_papers: int = None,
    sources: Optional[List[str]] = None
) -> List[Dict]:
    """
    Fetch papers from multiple sources based on a topic.
    
    Args:
        topic: Topic or query string
        max_papers: Maximum total papers to fetch
        sources: List of sources to use (default: all available)
        
    Returns:
        List of paper dictionaries with PDF URLs
    """
    max_papers = max_papers or settings.max_papers_per_query
    sources = sources or ["arxiv", "semantic_scholar", "crossref", "openalex"]
    
    logger.info(f"Fetching papers for topic: {topic} from {sources}")
    
    all_papers = []
    papers_per_source = max_papers // len(sources) + 1
    
    # Try Semantic Scholar
    if "semantic_scholar" in sources:
        try:
            semantic_papers = search_semantic_scholar(topic, limit=papers_per_source)
            if semantic_papers:
                all_papers.extend(semantic_papers)
                logger.info(f"Got {len(semantic_papers)} papers from Semantic Scholar")
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed: {e}")
    
    # Try ArXiv
    if "arxiv" in sources:
        remaining = max_papers - len(all_papers)
        if remaining > 0:
            try:
                arxiv_papers = search_arxiv(topic, max_results=min(remaining, papers_per_source))
                if arxiv_papers:
                    all_papers.extend(arxiv_papers)
                    logger.info(f"Got {len(arxiv_papers)} papers from ArXiv")
            except Exception as e:
                logger.warning(f"ArXiv search failed: {e}")
    
    # Try Crossref
    if "crossref" in sources:
        remaining = max_papers - len(all_papers)
        if remaining > 0:
            try:
                crossref_result = search_crossref(query=topic, rows=min(remaining, papers_per_source))
                crossref_papers = crossref_result.get("items", [])
                if crossref_papers:
                    all_papers.extend(crossref_papers)
                    logger.info(f"Got {len(crossref_papers)} papers from Crossref")
            except Exception as e:
                logger.warning(f"Crossref search failed: {e}")
    
    # Try OpenAlex
    if "openalex" in sources:
        remaining = max_papers - len(all_papers)
        if remaining > 0:
            try:
                openalex_result = search_openalex(query=topic, per_page=min(remaining, papers_per_source))
                openalex_papers = openalex_result.get("items", [])
                if openalex_papers:
                    all_papers.extend(openalex_papers)
                    logger.info(f"Got {len(openalex_papers)} papers from OpenAlex")
            except Exception as e:
                logger.warning(f"OpenAlex search failed: {e}")
    
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

