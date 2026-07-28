import traceback
import logging

class MockST:
    def __init__(self):
        self.session_state = {}
    def error(self, e): print(f"ST.ERROR: {e}")
    def exception(self, e): traceback.print_exc()
    def warning(self, w): print(f"ST.WARNING: {w}")
    def info(self, i): print(f"ST.INFO: {i}")

st = MockST()

query = 'natural lagnuauge proecessing for rag'
max_results = 10
arxiv_field = 'all'
arxiv_category = ''
arxiv_sort_by = 'relevance'
year_range = ''
fields_of_study = []
open_access_only = False
min_citations = 0
year_filter = None
pdf_only = True
source = "All Sources"

from ingestion.arxiv_enhanced import search_arxiv_enhanced
from ingestion.arxiv_enhanced import search_arxiv_enhanced as search_arxiv
from ingestion.semantic_scholar_enhanced import search_papers_enhanced
from ingestion.crossref_api import search_crossref
from ingestion.openalex_api import search_openalex
from ingestion.core_api import search_core
from api.literature_review_search import deduplicate_papers, add_quality_labels
from api.relevance_ranking import rank_papers_by_relevance

papers = []
try:
    if source in ["Both", "ArXiv", "All Sources"]:
        arxiv_result = search_arxiv_enhanced(
            query=query, max_results=max_results,
            field=None, sort_by=arxiv_sort_by, sort_order="descending"
        )
        arxiv_error = arxiv_result.get("error")
        if arxiv_error:
            st.warning(f"ArXiv API issue: {arxiv_error}")
        arxiv_papers = arxiv_result.get("entries", [])
        if arxiv_error and not arxiv_papers:
            arxiv_papers = search_arxiv(query, max_results=max_results)
        papers.extend(arxiv_papers)
        if arxiv_result.get("total", 0) > 0:
            st.info(f"📊 ArXiv found {arxiv_result.get('total', 0)} total papers")

    print(f'Papers after ArXiv: {len(papers)}')

    if source in ["Both", "Semantic Scholar", "All Sources"]:
        semantic_result = search_papers_enhanced(
            query=query, limit=max_results,
            year=year_range if year_range else None,
            fields_of_study=fields_of_study if fields_of_study else None,
            open_access_only=open_access_only,
            min_citation_count=min_citations if min_citations > 0 else None
        )
        semantic_error = semantic_result.get("error")
        if semantic_error:
            st.warning(f"Semantic Scholar API issue: {semantic_error}")
        papers.extend(semantic_result.get("data", []))
        if semantic_result.get("total", 0) > 0:
            st.info(f"📊 Semantic Scholar found {semantic_result.get('total', 0)} total papers")

    print(f'Papers after S2: {len(papers)}')

    def _resolve_pdf_url(paper):
        for key in ('pdf_url', 'openAccessPdf', 'pdfUrl'):
            v = paper.get(key)
            if v and isinstance(v, str) and v.startswith('http'):
                return v
        return None

    if pdf_only:
        papers = [p for p in papers if _resolve_pdf_url(p)]

    print(f'Papers after PDF filter: {len(papers)}')

    unique_papers = deduplicate_papers(papers)

    print(f'Papers after dedup: {len(unique_papers)}')

    if query and unique_papers:
        unique_papers = rank_papers_by_relevance(
            query=query, papers=unique_papers,
            use_semantic=True, use_keyword=True
        )
        unique_papers = add_quality_labels(query, unique_papers)

    st.session_state["search_results"] = unique_papers
    st.session_state["search_results_query"] = query

    print("Success. Final results:", len(st.session_state["search_results"]))
except Exception as e:
    st.error(f"Search error: {e}")
    st.exception(e)
