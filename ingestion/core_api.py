"""CORE API integration for open-access paper discovery."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import requests

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _build_headers() -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if settings.core_api_key:
        headers["Authorization"] = f"Bearer {settings.core_api_key}"
    return headers


def _format_core_error(error: Exception) -> str:
    message = str(error)
    if "401" in message or "403" in message:
        return "CORE API authentication failed. Check CORE_API_KEY in .env and restart the app."
    if "429" in message:
        return "CORE API rate limited (429). Wait a few seconds and retry with fewer expanded queries."
    return message


def _authors_to_list(authors: Any) -> List[str]:
    if not isinstance(authors, list):
        return []
    normalized = []
    for author in authors:
        if isinstance(author, dict) and author.get("name"):
            normalized.append(str(author["name"]))
        elif isinstance(author, str) and author.strip():
            normalized.append(author.strip())
    return normalized


def _field_of_study(raw_value: Any) -> List[str]:
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value if item]
    if isinstance(raw_value, str) and raw_value.strip():
        return [raw_value.strip()]
    return []


def _first_pdf_link(item: Dict[str, Any]) -> Optional[str]:
    download_url = item.get("downloadUrl")
    if isinstance(download_url, str) and download_url.strip():
        return download_url.strip()

    links = item.get("links")
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            url = str(link.get("url") or "").strip()
            if link.get("type") == "download" and url:
                return url
            if re.search(r"\.pdf($|[?#])", url.lower()):
                return url

    source_urls = item.get("sourceFulltextUrls")
    if isinstance(source_urls, list):
        for url in source_urls:
            url = str(url or "").strip()
            if re.search(r"\.pdf($|[?#])", url.lower()):
                return url
    return None


def _display_url(item: Dict[str, Any]) -> Optional[str]:
    links = item.get("links")
    if isinstance(links, list):
        for link_type in ("display", "reader", "download"):
            for link in links:
                if isinstance(link, dict) and link.get("type") == link_type and link.get("url"):
                    return str(link["url"])

    core_id = item.get("id")
    if core_id:
        return f"https://core.ac.uk/works/{core_id}"
    return item.get("downloadUrl")


def _reference_count(item: Dict[str, Any]) -> int:
    references_count = item.get("referencesCount")
    if isinstance(references_count, int):
        return references_count
    references = item.get("references")
    if isinstance(references, list):
        return len(references)
    return 0


def _normalize_core_work(item: Dict[str, Any]) -> Dict[str, Any]:
    authors = _authors_to_list(item.get("authors"))
    core_id = item.get("id")
    return {
        "paper_id": f"core:{core_id}" if core_id else None,
        "title": item.get("title") or "Untitled",
        "authors": authors,
        "authors_string": ", ".join(authors) if authors else "Unknown",
        "abstract": item.get("abstract") or "",
        "year": item.get("yearPublished"),
        "published": item.get("publishedDate"),
        "pdf_url": _first_pdf_link(item),
        "url": _display_url(item),
        "doi": item.get("doi"),
        "citation_count": item.get("citationCount") or 0,
        "reference_count": _reference_count(item),
        "publisher": item.get("publisher"),
        "concepts": _field_of_study(item.get("fieldOfStudy")),
        "open_access": True,
        "source": "core",
    }


def search_core(
    query: str,
    *,
    limit: int = 10,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Search CORE works and normalize results into the app's paper schema.

    CORE requires a free API key for reliable access. The key should be stored
    in CORE_API_KEY, not committed into source code.
    """
    if not settings.core_api_key:
        return {
            "total": 0,
            "items": [],
            "error": "CORE_API_KEY is missing. Add it to .env and restart the app.",
        }

    try:
        url = f"{settings.core_base_url.rstrip('/')}/search/works/"
        params = {
            "q": query,
            "limit": max(1, min(limit, 100)),
            "offset": max(offset, 0),
        }
        logger.info("Searching CORE: %s", query)
        response = requests.get(url, params=params, headers=_build_headers(), timeout=20)
        response.raise_for_status()
        payload = response.json()
        results = [_normalize_core_work(item) for item in payload.get("results", [])]
        return {
            "total": payload.get("totalHits", 0),
            "limit": payload.get("limit", params["limit"]),
            "offset": payload.get("offset", offset),
            "items": results,
            "error": None,
        }
    except Exception as exc:
        logger.error("CORE API error: %s", exc)
        return {
            "total": 0,
            "items": [],
            "error": _format_core_error(exc),
        }
