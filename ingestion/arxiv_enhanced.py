"""Enhanced ArXiv API integration with full query capabilities."""
import requests
import feedparser
from typing import List, Dict, Optional
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def search_arxiv_enhanced(
    query: Optional[str] = None,
    id_list: Optional[List[str]] = None,
    start: int = 0,
    max_results: int = 10,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    field: Optional[str] = None,
    submitted_date_start: Optional[str] = None,
    submitted_date_end: Optional[str] = None
) -> Dict:
    """
    Enhanced ArXiv search with full API capabilities.
    
    Args:
        query: Search query (can use field prefixes: ti:, au:, abs:, etc.)
        id_list: List of ArXiv IDs to fetch
        start: Starting index for paging (0-based)
        max_results: Maximum results (max 2000 per call, 30000 total)
        sort_by: "relevance", "lastUpdatedDate", or "submittedDate"
        sort_order: "ascending" or "descending"
        field: Field to search in (ti, au, abs, co, jr, cat, rn, all)
        submitted_date_start: Start date (YYYYMMDDHHMM format in GMT)
        submitted_date_end: End date (YYYYMMDDHHMM format in GMT)
        
    Returns:
        Dictionary with feed metadata and entries
    """
    try:
        params = {
            "start": start,
            "max_results": min(max_results, 2000),  # API limit per call
            "sortBy": sort_by,
            "sortOrder": sort_order
        }
        
        # Build search query
        if query:
            if field:
                # Use field prefix (ti:, au:, abs:, etc.)
                search_query = f"{field}:{query}"
            else:
                # Default to all: if no field specified
                search_query = f"all:{query}"
            
            # Add date filter if provided
            if submitted_date_start or submitted_date_end:
                date_filter = f"submittedDate:[{submitted_date_start or '*'} TO {submitted_date_end or '*'}]"
                search_query = f"{search_query} AND {date_filter}"
            
            params["search_query"] = search_query
        
        # Add id_list if provided
        if id_list:
            params["id_list"] = ",".join(id_list)
        
        logger.info(f"Searching ArXiv with params: {params}")
        response = requests.get(settings.arxiv_base_url, params=params, timeout=30)
        response.raise_for_status()
        
        feed = feedparser.parse(response.content)
        
        # Check for errors
        if feed.bozo and feed.bozo_exception:
            logger.error(f"ArXiv API error: {feed.bozo_exception}")
            return {"total": 0, "entries": []}
        
        # Extract feed metadata
        total_results = int(feed.feed.get("opensearch_totalresults", 0))
        start_index = int(feed.feed.get("opensearch_startindex", 0))
        items_per_page = int(feed.feed.get("opensearch_itemsperpage", 0))
        
        # Parse entries
        entries = []
        for entry in feed.entries:
            # Extract ArXiv ID
            arxiv_id = entry.id.split("/")[-1]
            
            # Get PDF URL
            pdf_url = None
            for link in entry.links:
                if link.rel == "alternate" and link.type == "application/pdf":
                    pdf_url = link.href
                    break
            
            if not pdf_url:
                # Fallback: construct PDF URL
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Extract authors
            authors = [author.name for author in entry.get("authors", [])]
            
            # Extract categories
            categories = []
            primary_category = None
            for tag in entry.get("tags", []):
                if tag.get("term"):
                    categories.append(tag.get("term"))
                    if tag.get("scheme") == "http://arxiv.org/schemas/atom":
                        primary_category = tag.get("term")
            
            # Extract ArXiv extension elements
            arxiv_comment = None
            arxiv_journal_ref = None
            arxiv_doi = None
            arxiv_affiliations = []
            
            # Check for arxiv namespace elements
            if hasattr(entry, 'arxiv_comment'):
                arxiv_comment = entry.arxiv_comment
            if hasattr(entry, 'arxiv_journal_ref'):
                arxiv_journal_ref = entry.arxiv_journal_ref
            if hasattr(entry, 'arxiv_doi'):
                arxiv_doi = entry.arxiv_doi
            
            # Extract affiliations from authors
            for author in entry.get("authors", []):
                if hasattr(author, 'arxiv_affiliation'):
                    arxiv_affiliations.append({
                        "name": author.name,
                        "affiliation": author.arxiv_affiliation
                    })
            
            entries.append({
                "paper_id": arxiv_id,
                "title": entry.title,
                "authors": authors,
                "authors_string": ", ".join(authors) if authors else "Unknown Authors",
                "abstract": entry.get("summary", ""),
                "year": entry.published_parsed.tm_year if entry.published_parsed else None,
                "published": entry.published,
                "updated": entry.updated,
                "pdf_url": pdf_url,
                "url": entry.link,
                "categories": categories,
                "primary_category": primary_category,
                "comment": arxiv_comment,
                "journal_ref": arxiv_journal_ref,
                "doi": arxiv_doi,
                "affiliations": arxiv_affiliations,
                "source": "arxiv"
            })
        
        logger.info(f"Found {len(entries)} papers (total: {total_results})")
        
        return {
            "total": total_results,
            "start": start_index,
            "items_per_page": items_per_page,
            "entries": entries
        }
        
    except Exception as e:
        logger.error(f"Error searching ArXiv: {e}")
        return {"total": 0, "entries": []}


def search_arxiv_by_field(
    field: str,
    term: str,
    max_results: int = 10,
    sort_by: str = "relevance"
) -> List[Dict]:
    """
    Search ArXiv by specific field.
    
    Args:
        field: Field to search (ti, au, abs, co, jr, cat, rn, all)
        term: Search term
        max_results: Maximum results
        sort_by: Sort order
        
    Returns:
        List of paper dictionaries
    """
    result = search_arxiv_enhanced(
        query=term,
        field=field,
        max_results=max_results,
        sort_by=sort_by
    )
    return result.get("entries", [])


def search_arxiv_by_author(author_name: str, max_results: int = 10) -> List[Dict]:
    """Search ArXiv by author name."""
    return search_arxiv_by_field("au", author_name, max_results)


def search_arxiv_by_title(title_keywords: str, max_results: int = 10) -> List[Dict]:
    """Search ArXiv by title keywords."""
    return search_arxiv_by_field("ti", title_keywords, max_results)


def search_arxiv_by_category(category: str, max_results: int = 10) -> List[Dict]:
    """Search ArXiv by subject category (e.g., cs.AI, math.CO)."""
    return search_arxiv_by_field("cat", category, max_results)


def get_arxiv_papers_by_id(arxiv_ids: List[str]) -> List[Dict]:
    """
    Get multiple ArXiv papers by their IDs.
    
    Args:
        arxiv_ids: List of ArXiv IDs (e.g., ["1706.03762", "2106.15928"])
        
    Returns:
        List of paper dictionaries
    """
    result = search_arxiv_enhanced(id_list=arxiv_ids, max_results=len(arxiv_ids))
    return result.get("entries", [])


def search_arxiv_with_boolean(
    query_parts: List[Dict],
    max_results: int = 10
) -> List[Dict]:
    """
    Search ArXiv with complex Boolean queries.
    
    Args:
        query_parts: List of query parts with 'field', 'term', 'operator'
                     Example: [
                         {"field": "au", "term": "Einstein", "operator": None},
                         {"field": "ti", "term": "relativity", "operator": "AND"}
                     ]
        max_results: Maximum results
        
    Returns:
        List of paper dictionaries
    """
    # Build Boolean query
    query_parts_str = []
    for part in query_parts:
        field = part.get("field", "all")
        term = part.get("term", "")
        operator = part.get("operator", "")
        
        if operator:
            query_parts_str.append(operator)
        
        query_parts_str.append(f"{field}:{term}")
    
    query = " ".join(query_parts_str)
    
    result = search_arxiv_enhanced(query=query, max_results=max_results)
    return result.get("entries", [])



