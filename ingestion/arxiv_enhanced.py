"""Enhanced ArXiv API integration with full query capabilities."""
import requests
import feedparser
import time
import arxiv
from typing import List, Dict, Optional
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)
MAX_RETRIES = 3
BASE_RETRY_DELAY_SECONDS = 1.0


def _retry_delay(response: requests.Response, attempt: int) -> float:
    """Compute retry delay using Retry-After header when available."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), 0.5)
        except ValueError:
            pass
    return BASE_RETRY_DELAY_SECONDS * (2 ** attempt)


def _request_arxiv_with_retries(params: Dict, timeout: int = 30, max_retries: int = MAX_RETRIES) -> requests.Response:
    """Make ArXiv API request with retries for transient/rate-limit failures."""
    transient_codes = {429, 500, 502, 503, 504}

    for attempt in range(max_retries):
        try:
            response = requests.get(settings.arxiv_base_url, params=params, timeout=timeout)
            if response.status_code in transient_codes and attempt < max_retries - 1:
                delay = _retry_delay(response, attempt)
                logger.warning(
                    "ArXiv request got status %s (attempt %s/%s). Retrying in %.1fs.",
                    response.status_code,
                    attempt + 1,
                    max_retries,
                    delay
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                delay = BASE_RETRY_DELAY_SECONDS * (2 ** attempt)
                logger.warning(
                    "ArXiv request failed (attempt %s/%s): %s. Retrying in %.1fs.",
                    attempt + 1,
                    max_retries,
                    e,
                    delay
                )
                time.sleep(delay)
                continue
            raise

    raise RuntimeError("ArXiv request failed after retries")


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
        else:
            params["search_query"] = ""

        # Add id_list if provided
        id_list_args = []
        if id_list:
            id_list_args = id_list
            if not params["search_query"]:
                # The arxiv package requires a query if id_list is empty, but id_list handles it
                params["search_query"] = ""

        # Map sort
        sort_crit = arxiv.SortCriterion.Relevance
        if sort_by == "lastUpdatedDate":
            sort_crit = arxiv.SortCriterion.LastUpdatedDate
        elif sort_by == "submittedDate":
            sort_crit = arxiv.SortCriterion.SubmittedDate

        sort_dir = arxiv.SortOrder.Descending
        if sort_order == "ascending":
            sort_dir = arxiv.SortOrder.Ascending

        logger.info(f"Searching ArXiv via python package: query='{params['search_query']}', ids={id_list_args}, max={params['max_results']}")

        client = arxiv.Client()
        search = arxiv.Search(
            query=params["search_query"],
            id_list=id_list_args,
            max_results=params["max_results"] + start,
            sort_by=sort_crit,
            sort_order=sort_dir
        )

        # Parse entries
        entries = []
        try:
            # We have to skip 'start' items
            results_iter = client.results(search)
            # Drain up to 'start' items
            for _ in range(start):
                try:
                    next(results_iter)
                except StopIteration:
                    break

            # Collect up to max_results items
            count = 0
            for r in results_iter:
                if count >= params["max_results"]:
                    break
                count += 1

                arxiv_id = r.get_short_id()
                pdf_url = r.pdf_url
                authors = [a.name for a in r.authors]
                categories = r.categories
                primary_category = r.primary_category

                entries.append({
                    "paper_id": arxiv_id,
                    "title": r.title,
                    "authors": authors,
                    "authors_string": ", ".join(authors) if authors else "Unknown Authors",
                    "abstract": r.summary,
                    "year": r.published.year if r.published else None,
                    "published": r.published.isoformat() if r.published else None,
                    "updated": r.updated.isoformat() if r.updated else None,
                    "pdf_url": pdf_url,
                    "url": r.entry_id,
                    "categories": categories,
                    "primary_category": primary_category,
                    "comment": r.comment,
                    "journal_ref": r.journal_ref,
                    "doi": r.doi,
                    "affiliations": [],
                    "source": "arxiv"
                })
        except Exception as e:
            logger.error(f"ArXiv API error: {e}")
            return {"total": 0, "entries": [], "error": str(e)}

        logger.info(f"Found {len(entries)} papers")

        return {
            "total": len(entries),
            "start": start,
            "items_per_page": params["max_results"],
            "entries": entries,
            "error": None
        }

    except Exception as e:
        logger.error(f"Error searching ArXiv: {e}")
        return {"total": 0, "entries": [], "error": str(e)}


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
