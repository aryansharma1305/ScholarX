"""Literature-review-oriented paper discovery and ranking."""
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from api.relevance_ranking import calculate_relevance_score, rank_papers_by_relevance
from ingestion.arxiv_enhanced import search_arxiv_enhanced
from ingestion.core_api import search_core
from ingestion.crossref_api import search_crossref
from ingestion.openalex_api import search_openalex
from ingestion.semantic_scholar_enhanced import search_papers_enhanced
from utils.logger import get_logger

logger = get_logger(__name__)

REVIEW_TERMS = (
    "survey",
    "review",
    "systematic review",
    "literature review",
    "meta-analysis",
    "state of the art",
    "state-of-the-art",
    "taxonomy",
    "overview",
)

SOURCE_WEIGHT = {
    "semantic_scholar": 1.0,
    "openalex": 0.9,
    "arxiv": 0.78,
    "crossref": 0.82,
    "core": 0.88,
    "multi_source": 1.0,
}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _normalize_identifier(value: Any) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", str(value or "").strip().lower())


def normalize_title(title: Any) -> str:
    """Normalize title for cross-source deduplication."""
    text = _clean_text(title).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _paper_key(paper: Dict[str, Any]) -> Tuple[str, str]:
    doi = _normalize_identifier(paper.get("doi"))
    if doi:
        return ("doi", doi)

    title = normalize_title(paper.get("title"))
    year = _as_int(paper.get("year"), default=0)
    if title:
        return ("title", f"{title}|{year or ''}")

    paper_id = _normalize_identifier(paper.get("paper_id"))
    if paper_id:
        return ("paper_id", paper_id)

    return ("unknown", str(id(paper)))


def _source_list(paper: Dict[str, Any]) -> List[str]:
    sources = paper.get("sources")
    if isinstance(sources, list):
        return [str(source) for source in sources if source]
    source = paper.get("source")
    return [str(source)] if source else []


def _merge_values(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    sources = sorted(set(_source_list(existing) + _source_list(incoming)))
    merged["sources"] = sources
    if len(sources) > 1:
        merged["source"] = "multi_source"
    elif sources:
        merged["source"] = sources[0]

    for field in (
        "title",
        "authors",
        "authors_string",
        "abstract",
        "year",
        "pdf_url",
        "url",
        "doi",
        "venue",
        "journal",
        "publisher",
        "paper_id",
    ):
        if not merged.get(field) and incoming.get(field):
            merged[field] = incoming[field]

    merged["citation_count"] = max(
        _as_int(merged.get("citation_count")),
        _as_int(incoming.get("citation_count")),
    )
    merged["reference_count"] = max(
        _as_int(merged.get("reference_count")),
        _as_int(incoming.get("reference_count")),
    )

    concepts = []
    for candidate in (merged.get("concepts"), incoming.get("concepts")):
        if isinstance(candidate, list):
            concepts.extend(str(item) for item in candidate if item)
    if concepts:
        merged["concepts"] = sorted(set(concepts))

    return merged


def deduplicate_papers(papers: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate papers by DOI first, then normalized title/year."""
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for raw_paper in papers:
        if not isinstance(raw_paper, dict):
            continue
        paper = dict(raw_paper)
        paper.setdefault("sources", _source_list(paper))
        key = _paper_key(paper)
        if key in deduped:
            deduped[key] = _merge_values(deduped[key], paper)
        else:
            deduped[key] = paper

    return list(deduped.values())


def expand_review_query(query: str, field: Optional[str] = None, max_queries: int = 6) -> List[str]:
    """Create practical query variants for literature-review discovery."""
    base = _clean_text(query)
    if not base:
        return []

    field_suffix = f" {field.strip()}" if field and field.strip().lower() not in base.lower() else ""
    candidates = [
        base,
        f"{base} survey",
        f"{base} literature review",
        f"{base} systematic review",
        f"{base} state of the art",
        f"{base} recent advances",
        f"{base}{field_suffix}".strip(),
        f"{base} applications challenges future directions",
    ]

    lowered = base.lower()
    if "rag" in lowered or "retrieval augmented" in lowered:
        candidates.extend([
            "retrieval augmented generation survey",
            "retrieval augmented generation evaluation challenges",
        ])
    if "federated" in lowered:
        candidates.append(f"{base} privacy security survey")
    if "blockchain" in lowered:
        candidates.append(f"{base} taxonomy applications")
    if "deep learning" in lowered or "machine learning" in lowered:
        candidates.append(f"{base} benchmark comparison")

    seen = set()
    expanded = []
    for candidate in candidates:
        normalized = _clean_text(candidate)
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            expanded.append(normalized)
        if len(expanded) >= max_queries:
            break
    return expanded


def _review_term_score(paper: Dict[str, Any]) -> float:
    title = normalize_title(paper.get("title"))
    abstract = normalize_title(paper.get("abstract"))
    haystack = f"{title} {abstract[:1200]}"
    if not haystack.strip():
        return 0.0
    matches = sum(1 for term in REVIEW_TERMS if term.replace("-", " ") in haystack)
    return min(matches / 2.0, 1.0)


def _citation_score(citation_count: int) -> float:
    if citation_count <= 0:
        return 0.0
    return min(math.log1p(citation_count) / math.log1p(1000), 1.0)


def _recency_score(year: int, current_year: int) -> float:
    if not year:
        return 0.25
    age = max(current_year - year, 0)
    if age <= 2:
        return 1.0
    if age <= 5:
        return 0.85
    if age <= 10:
        return 0.6
    if age <= 20:
        return 0.35
    return 0.2


def _source_score(paper: Dict[str, Any]) -> float:
    sources = _source_list(paper)
    if len(set(sources)) > 1:
        return 1.0
    return SOURCE_WEIGHT.get(str(paper.get("source", "")).lower(), 0.7)


def _has_pdf(paper: Dict[str, Any]) -> bool:
    if paper.get("pdf_url") or paper.get("pdfUrl"):
        return True
    url = str(paper.get("url") or "").lower()
    return ".pdf" in url or "arxiv.org/abs/" in url or paper.get("source") == "arxiv"


def score_review_paper(
    query: str,
    paper: Dict[str, Any],
    *,
    current_year: Optional[int] = None,
) -> Dict[str, Any]:
    """Score a paper for literature-review usefulness."""
    current_year = current_year or datetime.now().year
    relevance = calculate_relevance_score(
        query=query,
        paper=paper,
        use_semantic=False,
        use_keyword=True,
    )

    citations = _as_int(paper.get("citation_count"))
    year = _as_int(paper.get("year"))
    citation_component = _citation_score(citations)
    recency_component = _recency_score(year, current_year)
    review_component = _review_term_score(paper)
    source_component = _source_score(paper)
    pdf_component = 1.0 if _has_pdf(paper) else 0.0
    venue_component = 1.0 if paper.get("venue") or paper.get("journal") else 0.45

    score = (
        0.34 * relevance["combined_score"]
        + 0.20 * citation_component
        + 0.15 * recency_component
        + 0.13 * review_component
        + 0.07 * source_component
        + 0.06 * pdf_component
        + 0.05 * venue_component
    )
    score = min(max(score, 0.0), 1.0)

    reasons = []
    if relevance["combined_score"] >= 0.45:
        reasons.append("strong topic match")
    if citations >= 100:
        reasons.append(f"highly cited ({citations})")
    elif citations >= 25:
        reasons.append(f"solid citation signal ({citations})")
    if year and current_year - year <= 3:
        reasons.append("recent")
    elif citations >= 500 and year and current_year - year > 8:
        reasons.append("foundational")
    if review_component >= 0.5:
        reasons.append("review/survey language")
    if len(set(_source_list(paper))) > 1:
        reasons.append("found in multiple sources")
    if _has_pdf(paper):
        reasons.append("PDF available")

    if score >= 0.72:
        label = "Review-ready"
    elif score >= 0.55:
        label = "Strong candidate"
    elif score >= 0.38:
        label = "Useful background"
    else:
        label = "Needs manual screening"

    return {
        "review_score": score,
        "review_percent": f"{score * 100:.1f}%",
        "review_breakdown": {
            "topic": relevance["combined_score"],
            "citations": citation_component,
            "recency": recency_component,
            "review_signal": review_component,
            "source": source_component,
            "pdf": pdf_component,
            "venue": venue_component,
        },
        "quality_label": label,
        "quality_reasons": reasons or ["limited metadata; manually screen"],
    }


def add_quality_labels(query: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach lightweight quality labels to normal search results."""
    labelled = []
    for paper in papers:
        scored = score_review_paper(query=query, paper=paper)
        labelled.append({**paper, **scored})
    return labelled


def rank_review_papers(query: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank papers for literature review quality."""
    ranked = add_quality_labels(query, papers)
    ranked.sort(
        key=lambda paper: (
            paper.get("review_score", 0.0),
            _as_int(paper.get("citation_count")),
            _as_int(paper.get("year")),
        ),
        reverse=True,
    )
    return ranked


def _year_range_string(year_start: Optional[int], year_end: Optional[int]) -> Optional[str]:
    if year_start and year_end:
        return f"{year_start}-{year_end}"
    if year_start:
        return f"{year_start}-"
    if year_end:
        return f"-{year_end}"
    return None


def _filter_papers(
    papers: List[Dict[str, Any]],
    *,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    min_citations: int = 0,
    pdf_only: bool = False,
) -> List[Dict[str, Any]]:
    filtered = []
    for paper in papers:
        year = _as_int(paper.get("year"))
        citations = _as_int(paper.get("citation_count"))
        if year_start and year and year < year_start:
            continue
        if year_end and year and year > year_end:
            continue
        if min_citations and citations < min_citations:
            continue
        if pdf_only and not _has_pdf(paper):
            continue
        filtered.append(paper)
    return filtered


def search_literature_review(
    query: str,
    *,
    field: Optional[str] = None,
    max_results: int = 30,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    min_citations: int = 0,
    pdf_only: bool = False,
    max_expanded_queries: int = 5,
) -> Dict[str, Any]:
    """
    Search multiple scholarly sources with review-oriented query expansion,
    deduplication, and quality ranking.
    """
    expanded_queries = expand_review_query(query, field=field, max_queries=max_expanded_queries)
    if not expanded_queries:
        return {
            "papers": [],
            "expanded_queries": [],
            "warnings": ["Enter a topic before searching."],
            "source_counts": {},
            "total_before_dedup": 0,
            "total_after_dedup": 0,
        }

    per_query = max(4, min(12, math.ceil(max_results / max(len(expanded_queries), 1))))
    year_range = _year_range_string(year_start, year_end)
    all_papers: List[Dict[str, Any]] = []
    warnings: List[str] = []
    source_counts: Dict[str, int] = {}

    for query_index, expanded_query in enumerate(expanded_queries):
        try:
            semantic = search_papers_enhanced(
                query=expanded_query,
                limit=per_query,
                year=year_range,
                min_citation_count=min_citations if min_citations > 0 else None,
            )
            if semantic.get("error"):
                warnings.append(f"Semantic Scholar ({expanded_query}): {semantic['error']}")
            semantic_papers = semantic.get("data", [])
            all_papers.extend(semantic_papers)
            source_counts["semantic_scholar"] = source_counts.get("semantic_scholar", 0) + len(semantic_papers)
        except Exception as exc:
            warnings.append(f"Semantic Scholar ({expanded_query}) failed: {exc}")

        try:
            arxiv = search_arxiv_enhanced(
                query=expanded_query,
                max_results=per_query,
                sort_by="relevance",
                sort_order="descending",
            )
            if arxiv.get("error"):
                warnings.append(f"ArXiv ({expanded_query}): {arxiv['error']}")
            arxiv_papers = arxiv.get("entries", [])
            all_papers.extend(arxiv_papers)
            source_counts["arxiv"] = source_counts.get("arxiv", 0) + len(arxiv_papers)
        except Exception as exc:
            warnings.append(f"ArXiv ({expanded_query}) failed: {exc}")

        try:
            openalex = search_openalex(
                query=expanded_query,
                per_page=per_query,
            )
            openalex_papers = openalex.get("items", [])
            all_papers.extend(openalex_papers)
            source_counts["openalex"] = source_counts.get("openalex", 0) + len(openalex_papers)
        except Exception as exc:
            warnings.append(f"OpenAlex ({expanded_query}) failed: {exc}")

        try:
            crossref = search_crossref(
                query=expanded_query,
                rows=per_query,
            )
            crossref_papers = crossref.get("items", [])
            all_papers.extend(crossref_papers)
            source_counts["crossref"] = source_counts.get("crossref", 0) + len(crossref_papers)
        except Exception as exc:
            warnings.append(f"Crossref ({expanded_query}) failed: {exc}")

        # CORE has valuable open-access PDFs, but keep expanded-review calls conservative.
        if query_index < 2:
            try:
                core = search_core(
                    query=expanded_query,
                    limit=per_query,
                )
                if core.get("error"):
                    warnings.append(f"CORE ({expanded_query}): {core['error']}")
                core_papers = core.get("items", [])
                all_papers.extend(core_papers)
                source_counts["core"] = source_counts.get("core", 0) + len(core_papers)
            except Exception as exc:
                warnings.append(f"CORE ({expanded_query}) failed: {exc}")

    total_before_dedup = len(all_papers)
    deduped = deduplicate_papers(all_papers)
    filtered = _filter_papers(
        deduped,
        year_start=year_start,
        year_end=year_end,
        min_citations=min_citations,
        pdf_only=pdf_only,
    )
    ranked = rank_review_papers(query, filtered)[:max_results]

    return {
        "query": query,
        "papers": ranked,
        "expanded_queries": expanded_queries,
        "warnings": warnings,
        "source_counts": source_counts,
        "total_before_dedup": total_before_dedup,
        "total_after_dedup": len(deduped),
    }


def review_papers_to_dataframe(papers: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build an export-ready table for selected review candidates."""
    rows = []
    for paper in papers:
        rows.append({
            "title": paper.get("title"),
            "authors": paper.get("authors_string") or paper.get("authors"),
            "year": paper.get("year"),
            "source": ", ".join(_source_list(paper)) or paper.get("source"),
            "citations": _as_int(paper.get("citation_count")),
            "review_score": paper.get("review_percent"),
            "quality_label": paper.get("quality_label"),
            "quality_reasons": "; ".join(paper.get("quality_reasons", [])),
            "doi": paper.get("doi"),
            "url": paper.get("url"),
            "pdf_url": paper.get("pdf_url"),
        })
    return pd.DataFrame(rows)


def review_papers_to_markdown(papers: List[Dict[str, Any]]) -> str:
    """Create a review-ready Markdown bibliography list."""
    lines = ["# Review-ready papers", ""]
    for index, paper in enumerate(papers, 1):
        title = paper.get("title", "Untitled")
        authors = paper.get("authors_string") or paper.get("authors") or "Unknown authors"
        year = paper.get("year") or "n.d."
        citations = _as_int(paper.get("citation_count"))
        label = paper.get("quality_label", "Candidate")
        score = paper.get("review_percent", "N/A")
        reasons = "; ".join(paper.get("quality_reasons", []))
        url = paper.get("url") or paper.get("pdf_url") or ""
        lines.extend([
            f"## {index}. {title}",
            f"- Authors: {authors}",
            f"- Year: {year}",
            f"- Quality: {label} ({score})",
            f"- Citations: {citations}",
            f"- Why selected: {reasons or 'manual screening needed'}",
            f"- Link: {url}" if url else "- Link: unavailable",
            "",
        ])
    return "\n".join(lines)
