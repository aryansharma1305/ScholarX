"""Comprehensive Streamlit App for ScholarX - Full Featured RAG Pipeline."""
import streamlit as st
import sys
import json
import time
import os
import hashlib
import re
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import io
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).parent))

from main import query_rag, search_papers
from api.main_api import api
from manage_papers import get_statistics
from ingestion.ingest_pipeline import ingest_pdf_from_url, ingest_pdf_from_file
from ingestion.paper_fetcher import search_arxiv, search_semantic_scholar
from ingestion.arxiv_enhanced import (
    search_arxiv_enhanced, search_arxiv_by_author, search_arxiv_by_title,
    search_arxiv_by_category, get_arxiv_papers_by_id
)
from ingestion.semantic_scholar_enhanced import (
    search_papers_enhanced, paper_autocomplete, get_paper_details,
    get_paper_citations, get_paper_references, search_authors, get_author_papers,
    search_snippets
)
from ingestion.crossref_api import search_crossref
from ingestion.openalex_api import search_openalex
from ingestion.core_api import search_core
from config.settings import settings
from config.chroma_client import get_collection
from api.relevance_ranking import rank_papers_by_relevance, get_relevance_category
from api.literature_review_search import (
    add_quality_labels,
    deduplicate_papers,
    review_papers_to_dataframe,
    review_papers_to_markdown,
    search_literature_review,
)
from api.visualization import (
    visualize_citation_network, 
    get_influential_papers, 
    get_research_communities,
    get_citation_statistics,
    build_citation_graph
)
import networkx as nx

CURRENT_YEAR = datetime.now().year

# Page config
st.set_page_config(
    page_title="ScholarX - Research Paper RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_papers' not in st.session_state:
    st.session_state.selected_papers = []
if 'library_papers' not in st.session_state:
    st.session_state.library_papers = []
if 'processing_tasks' not in st.session_state:
    st.session_state.processing_tasks = {}
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'search_results_query' not in st.session_state:
    st.session_state.search_results_query = ""
if 'review_search_meta' not in st.session_state:
    st.session_state.review_search_meta = {}
if 'graph_thread_id' not in st.session_state:
    st.session_state.graph_thread_id = str(uuid.uuid4())


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .paper-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .citation {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 3px solid #1f77b4;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .stat-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .badge-arxiv { background-color: #b31b1b; color: white; }
    .badge-semantic { background-color: #1857b0; color: white; }
    .badge-processed { background-color: #28a745; color: white; }
    .badge-processing { background-color: #ffc107; color: black; }
    .badge-pending { background-color: #6c757d; color: white; }
</style>
""", unsafe_allow_html=True)


def get_library_papers(force_refresh: bool = False):
    """Get all papers in the library (cached in session_state to avoid per-card DB calls)."""
    cache_key = "_library_papers_cache"
    if not force_refresh and cache_key in st.session_state:
        return st.session_state[cache_key]
    try:
        collection = get_collection()
        all_data = collection.get(limit=10000)

        papers = {}
        # Count chunks directly from the single all_data fetch
        chunk_counts: Dict[str, int] = {}
        for metadata in all_data.get("metadatas", []):
            pid = metadata.get("paper_id", "unknown")
            chunk_counts[pid] = chunk_counts.get(pid, 0) + 1

        for metadata in all_data.get("metadatas", []):
            paper_id = metadata.get("paper_id", "unknown")
            if paper_id not in papers:
                papers[paper_id] = {
                    "paper_id": paper_id,
                    "title": metadata.get("title", "Unknown"),
                    "authors": metadata.get("authors", "Unknown"),
                    "year": metadata.get("year"),
                    "abstract": metadata.get("abstract", ""),
                    "source": metadata.get("source", "unknown"),
                    "pdf_url": metadata.get("pdf_url", ""),
                    "processed": True,
                    "chunk_count": chunk_counts.get(paper_id, 0)
                }

        result = list(papers.values())
        st.session_state[cache_key] = result
        return result
    except Exception as e:
        st.error(f"Error loading library: {e}")
        return []


def remove_paper_from_library(paper_id: str) -> bool:
    """Remove all chunks for a paper from the vector store."""
    try:
        collection = get_collection()
        existing = collection.get(where={"paper_id": paper_id})
        chunk_ids = existing.get("ids", []) if existing else []
        if not chunk_ids:
            return False
        collection.delete(ids=chunk_ids)
        # Invalidate library cache after removal
        st.session_state.pop("_library_papers_cache", None)
        return True
    except Exception as e:
        st.error(f"Error removing paper {paper_id}: {e}")
        return False


def _paper_widget_id(paper: Dict) -> str:
    """Create a stable unique ID for UI widgets even when paper_id is missing."""
    base_id = str(
        paper.get("paper_id")
        or paper.get("doi")
        or paper.get("url")
        or f"{paper.get('title', '')}|{paper.get('year', '')}|{paper.get('source', '')}"
    )
    return hashlib.md5(base_id.encode("utf-8")).hexdigest()[:12]


def _resolve_pdf_url(paper: Dict) -> Optional[str]:
    """
    Resolve a direct PDF URL from a paper record.

    Returns None when only a landing page is available.
    """
    direct_candidates = [
        paper.get("pdf_url"),
        paper.get("pdfUrl")
    ]

    for candidate in direct_candidates:
        if candidate and isinstance(candidate, str):
            return candidate

    # ArXiv link conversion
    url = str(paper.get("url") or "")
    if "arxiv.org/abs/" in url:
        arxiv_id = url.split("arxiv.org/abs/")[-1].split("?")[0].strip()
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    if "arxiv.org/pdf/" in url and url.endswith(".pdf"):
        return url

    if paper.get("source") == "arxiv":
        arxiv_id = str(paper.get("paper_id") or paper.get("arxiv_id") or "").strip()
        if arxiv_id:
            arxiv_id = arxiv_id.replace("arXiv:", "").replace("arxiv:", "")
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    # Generic URL fallback only when it clearly points to a PDF file
    if url and re.search(r"\.pdf($|[?#])", url.lower()):
        return url

    return None


def display_paper_card_with_ranking(paper: Dict, query: str = "", rank: int = 0, show_add_button: bool = True):
    """Display a paper card with relevance ranking."""
    # Ensure get_relevance_category is available
    try:
        from api.relevance_ranking import get_relevance_category
    except (ImportError, NameError):
        # Fallback function if import fails
        def get_relevance_category(score: float) -> Dict[str, str]:
            if score >= 0.8:
                return {"category": "Excellent Match", "color": "#28a745", "emoji": "🟢", "badge": "success"}
            elif score >= 0.6:
                return {"category": "Good Match", "color": "#17a2b8", "emoji": "🔵", "badge": "info"}
            elif score >= 0.4:
                return {"category": "Moderate Match", "color": "#ffc107", "emoji": "🟡", "badge": "warning"}
            elif score >= 0.2:
                return {"category": "Weak Match", "color": "#fd7e14", "emoji": "🟠", "badge": "warning"}
            else:
                return {"category": "Poor Match", "color": "#dc3545", "emoji": "🔴", "badge": "danger"}
    
    source = paper.get("source", "unknown")
    sources = paper.get("sources") if isinstance(paper.get("sources"), list) else []
    
    # Get relevance info
    relevance_score = paper.get("relevance_score", 0.0)
    relevance_percent = paper.get("relevance_percent", "N/A")
    relevance_info = get_relevance_category(relevance_score) if relevance_score > 0 and query else None
    
    # Display relevance badge using Streamlit components
    if relevance_info and query:
        col1, col2 = st.columns([1, 15])
        with col1:
            st.markdown(f"## {relevance_info['emoji']}")
        with col2:
            st.markdown(f"**#{rank} - {relevance_percent} Match** - {relevance_info['category']}")
    
    # Display paper card using Streamlit native components
    st.markdown(f"### {paper.get('title', 'Unknown')}")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Authors:** {paper.get('authors_string', paper.get('authors', 'Unknown'))}")
        st.markdown(f"**Year:** {paper.get('year', 'N/A')}")
    with col2:
        # Source badge
        source_label = ", ".join(str(item).upper() for item in sources) if sources else str(source).upper()
        st.markdown(f"**Source:** {source_label}")
        if paper.get("citation_count") is not None:
            st.markdown(f"**Citations:** {paper.get('citation_count', 0)}")
    
    # Show relevance progress bar
    if relevance_info and query:
        st.progress(relevance_score, text=f"Relevance Score: {relevance_percent}")
        
        # Show breakdown
        if paper.get("relevance_breakdown"):
            with st.expander("📊 Relevance Breakdown", expanded=False):
                breakdown = paper["relevance_breakdown"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Semantic", f"{breakdown.get('semantic', 0) * 100:.1f}%")
                with col2:
                    st.metric("Keyword", f"{breakdown.get('keyword', 0) * 100:.1f}%")
                with col3:
                    st.metric("Title Match", f"{breakdown.get('title_match', 0) * 100:.1f}%")

    if paper.get("quality_label"):
        quality_label = paper.get("quality_label")
        review_percent = paper.get("review_percent")
        quality_text = f"**Quality:** {quality_label}"
        if review_percent:
            quality_text += f" ({review_percent})"
        st.markdown(quality_text)
        reasons = paper.get("quality_reasons") or []
        if reasons:
            st.caption("Why: " + "; ".join(str(reason) for reason in reasons))
        if paper.get("review_score") is not None:
            try:
                st.progress(float(paper.get("review_score", 0.0)), text=f"Review readiness: {review_percent}")
            except (TypeError, ValueError):
                pass
        if paper.get("review_breakdown"):
            with st.expander("Review Quality Breakdown", expanded=False):
                breakdown = paper["review_breakdown"]
                cols = st.columns(4)
                cols[0].metric("Topic", f"{breakdown.get('topic', 0) * 100:.0f}%")
                cols[1].metric("Citations", f"{breakdown.get('citations', 0) * 100:.0f}%")
                cols[2].metric("Recency", f"{breakdown.get('recency', 0) * 100:.0f}%")
                cols[3].metric("Review Signal", f"{breakdown.get('review_signal', 0) * 100:.0f}%")
    
    if paper.get("abstract"):
        with st.expander("📄 Abstract"):
            st.write(paper["abstract"])
    
    widget_id = _paper_widget_id(paper)
    pdf_url = _resolve_pdf_url(paper)

    col1, col2, col3 = st.columns(3)
    with col1:
        if pdf_url:
            st.link_button("📥 Download PDF", pdf_url)
        else:
            st.info("📄 No PDF available")
    with col2:
        if show_add_button:
            # Use cached library data (not a fresh DB call per card)
            paper_id = str(paper.get("paper_id") or "")
            paper_title = str(paper.get("title") or "").strip().lower()
            library = get_library_papers()  # uses session_state cache
            lib_ids = {str(p.get("paper_id") or "") for p in library}
            lib_titles = {str(p.get("title") or "").strip().lower() for p in library}
            is_in_library = (
                (paper_id and paper_id in lib_ids)
                or (paper_title and paper_title in lib_titles)
            )

            # Check if PDF URL is available (use _resolve_pdf_url, not just pdf_url field)
            has_pdf = bool(pdf_url)

            if is_in_library:
                st.success("✅ Already in Library")
            elif has_pdf:
                if st.button("➕ Add to Library", key=f"add_card_{widget_id}", type="primary"):
                    # Invalidate cache before processing so updated state is shown
                    st.session_state.pop("_library_papers_cache", None)
                    process_paper_for_rag(paper)
            else:
                st.warning("⚠️ No PDF — cannot add to library")
    with col3:
        if st.button("📊 View Details", key=f"view_card_{widget_id}"):
            st.session_state[f"view_paper_{widget_id}"] = paper
            st.rerun()

    st.divider()


def display_paper_card(paper: Dict, show_add_button: bool = True):
    """Display a paper card with all metadata."""
    display_paper_card_with_ranking(paper, query="", rank=0, show_add_button=show_add_button)
    
    # Paper details view
    for key in st.session_state.keys():
        if key.startswith("view_paper_"):
            paper = st.session_state[key]
            with st.expander(f"📄 Full Details: {paper.get('title', 'Unknown')}", expanded=True):
                st.markdown(f"**Title:** {paper.get('title', 'Unknown')}")
                st.markdown(f"**Authors:** {paper.get('authors', 'Unknown')}")
                st.markdown(f"**Year:** {paper.get('year', 'N/A')}")
                st.markdown(f"**Source:** {paper.get('source', 'unknown')}")
                
                if paper.get("abstract"):
                    st.markdown("**Abstract:**")
                    st.write(paper["abstract"])
                
                # Get enhanced details from Semantic Scholar if available
                detail_widget_id = _paper_widget_id(paper)
                if paper.get("paper_id") and paper.get("source") == "semantic_scholar":
                    if st.button("🔍 Get Full Details", key=f"details_{detail_widget_id}"):
                        with st.spinner("Fetching detailed information..."):
                            try:
                                details = get_paper_details(paper.get("paper_id"))
                                if details:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Citations", details.get("citation_count", 0))
                                        st.metric("References", details.get("reference_count", 0))
                                    with col2:
                                        if details.get("venue"):
                                            st.write(f"**Venue:** {details['venue']}")
                                        if details.get("fields_of_study"):
                                            st.write(f"**Fields:** {', '.join(details['fields_of_study'])}")
                                    
                                    # Citations
                                    if details.get("citations"):
                                        with st.expander(f"📚 Citations ({len(details['citations'])} shown)"):
                                            for citation in details["citations"][:10]:
                                                st.write(f"• {citation.get('title', 'Unknown')} ({citation.get('year', 'N/A')})")
                                    
                                    # References
                                    if details.get("references"):
                                        with st.expander(f"📖 References ({len(details['references'])} shown)"):
                                            for ref in details["references"][:10]:
                                                st.write(f"• {ref.get('title', 'Unknown')} ({ref.get('year', 'N/A')})")
                            except Exception as e:
                                st.error(f"Error fetching details: {e}")
                
                if st.button("❌ Close", key=f"close_{detail_widget_id}"):
                    del st.session_state[key]
                    st.rerun()


def process_paper_for_rag(paper: Dict):
    """Process a paper for RAG."""
    # Resolve direct PDF URL only (landing pages are not ingestible)
    pdf_url = _resolve_pdf_url(paper)
    
    if not pdf_url:
        st.error("❌ No downloadable PDF URL found for this paper.")
        st.info("💡 Use ArXiv or an open-access source where a direct PDF link is available, then try again.")
        return
    
    task_id = str(paper.get("paper_id") or _paper_widget_id(paper))
    
    st.session_state.processing_tasks[task_id] = {
        "status": "processing",
        "paper": paper,
        "start_time": datetime.now()
    }
    
    try:
        # Create a placeholder for status messages
        status_container = st.empty()
        status_container.info(f"🔄 Processing: {paper.get('title', 'Paper')[:60]}...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📥 Downloading PDF...")
        progress_bar.progress(10)
        
        status_text.text("📄 Extracting text from PDF...")
        progress_bar.progress(30)
        
        status_text.text("✂️ Chunking content...")
        progress_bar.progress(50)
        
        status_text.text("🧮 Generating embeddings...")
        progress_bar.progress(70)
        
        # Ingest the paper
        paper_id = ingest_pdf_from_url(
            pdf_url=pdf_url,
            paper_id=paper.get("paper_id"),
            metadata={
                "title": paper.get("title", ""),
                "authors": paper.get("authors_string", paper.get("authors", "")),
                "abstract": paper.get("abstract", ""),
                "year": paper.get("year"),
                "source": paper.get("source", "api"),
            }
        )
        
        status_text.text("💾 Storing in vector database...")
        progress_bar.progress(90)
        
        status_text.text("✅ Complete!")
        progress_bar.progress(100)
        
        st.session_state.processing_tasks[task_id]["status"] = "completed"
        # Invalidate library cache so the card immediately shows "Already in Library"
        st.session_state.pop("_library_papers_cache", None)

        # Clear status messages
        status_container.empty()
        status_text.empty()
        progress_bar.empty()

        # Show success message
        st.success(f"✅ **Paper added to library successfully!**\n\n**Title:** {paper.get('title', 'Unknown')}\n**ID:** {paper_id}")
        time.sleep(1)
        st.rerun()
            
    except Exception as e:
        st.session_state.processing_tasks[task_id]["status"] = "failed"
        st.error(f"❌ **Failed to add paper to library:**\n\n{str(e)}")
        st.exception(e)  # Show full traceback for debugging


# Main App
st.markdown('<div class="main-header">📚 ScholarX - Research Paper RAG System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("📊 Collection Stats")
    if st.button("🔄 Refresh Stats"):
        st.rerun()
    
    try:
        stats_output = []
        f = io.StringIO()
        with redirect_stdout(f):
            get_statistics()
        stats_text = f.getvalue()
        st.text(stats_text[:500])
    except:
        st.info("No statistics available")
    
    st.divider()
    
    st.subheader("🔧 RAG Settings")
    max_papers = st.slider("Papers per query", 1, 10, 5)
    settings.max_papers_per_query = max_papers
    
    top_k = st.slider("Top K chunks", 3, 15, 5)
    use_enhanced = st.checkbox("Enhanced features", True)
    
    st.divider()
    
    st.subheader("💾 Chat History")
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.session_state.graph_thread_id = str(uuid.uuid4())
        st.rerun()
    
    if st.session_state.chat_history:
        st.write(f"{len(st.session_state.chat_history)} conversations")

# Main Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Home", "🔍 Search Papers", "📚 My Library", "💬 RAG Chat", 
    "🔎 Advanced Search", "📤 Upload Paper", "📊 Analysis"
])

# Tab 1: Home
with tab1:
    st.markdown("### Welcome to ScholarX!")
    st.markdown("Your personal research assistant powered by RAG (Retrieval-Augmented Generation)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Papers in Library", len(get_library_papers()))
    with col2:
        try:
            collection = get_collection()
            st.metric("Total Chunks", collection.count())
        except:
            st.metric("Total Chunks", 0)
    with col3:
        st.metric("Chat Sessions", len(st.session_state.chat_history))
    
    st.divider()
    
    st.markdown("### 🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔍 Search Papers", use_container_width=True):
            st.switch_page("pages/2_🔍_Search_Papers.py") if Path("pages").exists() else None
    with col2:
        if st.button("💬 Start Chat", use_container_width=True):
            st.switch_page("pages/4_💬_RAG_Chat.py") if Path("pages").exists() else None
    with col3:
        if st.button("📤 Upload PDF", use_container_width=True):
            st.switch_page("pages/6_📤_Upload_Paper.py") if Path("pages").exists() else None
    with col4:
        if st.button("📊 View Library", use_container_width=True):
            st.switch_page("pages/3_📚_My_Library.py") if Path("pages").exists() else None
    
    st.divider()
    
    st.markdown("### 📈 Recent Activity")
    if st.session_state.processing_tasks:
        st.write("Processing Tasks:")
        for task_id, task in list(st.session_state.processing_tasks.items())[-5:]:
            status = task["status"]
            badge = "🟢" if status == "completed" else "🟡" if status == "processing" else "🔴"
            st.write(f"{badge} {task['paper'].get('title', 'Unknown')[:50]}... - {status}")

# Tab 2: Search Papers
with tab2:
    st.markdown("### 🔍 Search Research Papers")
    st.markdown("Search across ArXiv, Semantic Scholar, and more")
    
    search_type = st.radio(
        "Search by:",
        ["Keywords", "Literature Review", "Author", "Year", "Field"],
        horizontal=True,
        key="search_type_radio"
    )

    # Clear stale results when the user switches search modes
    if st.session_state.get("_last_search_type") != search_type:
        st.session_state["_last_search_type"] = search_type
        st.session_state["search_results"] = []
        st.session_state["search_results_query"] = ""
        st.session_state["review_search_meta"] = {}

    
    if search_type == "Keywords":
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Enter search query:", placeholder="e.g., transformer architecture, attention mechanism", key="search_query")
        with col2:
            source = st.selectbox("Source", ["Both", "ArXiv", "Semantic Scholar", "Crossref", "OpenAlex", "CORE", "All Sources"])
        
        # ArXiv query builder helper
        if source in ["Both", "ArXiv", "All Sources"]:
            with st.expander("🔧 ArXiv Query Builder"):
                st.markdown("**Field Prefixes:**")
                st.code("ti:title  au:author  abs:abstract  cat:category  all:everything")
                st.markdown("**Boolean Operators:**")
                st.code("AND  OR  ANDNOT")
                st.markdown("**Examples:**")
                st.code("au:Einstein AND ti:relativity\ncat:cs.AI AND abs:neural\nall:transformer ANDNOT cat:math")
                
                use_advanced = st.checkbox("Use advanced query syntax", False, key="use_advanced_query")
                if use_advanced:
                    # Use same key as main text_input so they stay in sync (no accumulation)
                    query = st.text_area(
                        "Enter ArXiv query (advanced):",
                        value=st.session_state.get("search_query", ""),
                        key="search_query",
                        height=80
                    )
                    st.caption("💡 You can use field prefixes (ti:, au:, abs:) and Boolean operators (AND, OR, ANDNOT)")
        
        # Enhanced filters
        with st.expander("🔧 Advanced Filters"):
            col1, col2 = st.columns(2)
            with col1:
                # ArXiv filters
                arxiv_field = st.selectbox(
                    "ArXiv Search Field",
                    ["all", "ti (title)", "au (author)", "abs (abstract)", "cat (category)"],
                    key="arxiv_field"
                )
                arxiv_category = st.text_input("ArXiv Category (e.g., cs.AI, math.CO)", placeholder="Optional")
                arxiv_sort_by = st.selectbox("Sort by", ["relevance", "lastUpdatedDate", "submittedDate"], key="arxiv_sort_by")
            
            with col2:
                # Semantic Scholar filters
                if source in ["Both", "Semantic Scholar", "All Sources"]:
                    year_range = st.text_input("Year range (e.g., 2020-2024)", placeholder="2020-2024")
                    fields_of_study = st.multiselect(
                        "Fields of Study",
                        ["Computer Science", "Physics", "Mathematics", "Biology", "Medicine", "Engineering"],
                        default=[]
                    )
                    min_citations = st.number_input("Min citations", min_value=0, value=0)
                    open_access_only = st.checkbox("Open access only", False)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            year_filter = st.number_input("Year (optional)", min_value=1900, max_value=CURRENT_YEAR, value=None, key="year_filter")
        with col2:
            pdf_only = st.checkbox("PDF available only", True)
        with col3:
            max_results = st.slider("Max results", 5, 50, 10)
        
        # Autocomplete suggestions — only fires when query is stable (not on every keystroke)
        if query and len(query) > 2:
            last_query = st.session_state.get("_last_autocomplete_query", "")
            if query != last_query:
                # Query just changed — store it, skip API call this render
                st.session_state["_last_autocomplete_query"] = query
            else:
                # Query is stable — safe to call API
                try:
                    suggestions = paper_autocomplete(query, limit=5)
                    if suggestions:
                        st.caption("💡 Suggestions:")
                        for sug in suggestions:
                            if st.button(f"📄 {sug.get('title', 'Unknown')[:60]}...", key=f"sug_{sug.get('paper_id')}"):
                                st.session_state["search_query"] = sug.get("title", query)
                                st.rerun()
                except Exception:
                    pass
        
        if st.button("🔍 Search", type="primary", key="main_search_btn"):
            if query:
                with st.spinner("Searching papers..."):
                    try:
                        papers = []

                        # Enhanced ArXiv search
                        if source in ["Both", "ArXiv", "All Sources"]:
                            try:
                                field_map = {
                                    "all": None, "ti (title)": "ti", "au (author)": "au",
                                    "abs (abstract)": "abs", "cat (category)": "cat"
                                }
                                arxiv_field_value = field_map.get(arxiv_field, None)
                                arxiv_query = query.strip()  # strip to avoid whitespace accumulation
                                if arxiv_category and arxiv_category.strip():
                                    arxiv_query = f"{arxiv_query} AND cat:{arxiv_category.strip()}"
                                arxiv_result = search_arxiv_enhanced(
                                    query=arxiv_query, max_results=max_results,
                                    field=arxiv_field_value, sort_by=arxiv_sort_by, sort_order="descending"
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
                            except Exception as e:
                                st.warning(f"ArXiv search failed: {e}")
                                papers.extend(search_arxiv(query, max_results=max_results))

                        # Enhanced Semantic Scholar search
                        if source in ["Both", "Semantic Scholar", "All Sources"]:
                            try:
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
                            except Exception as e:
                                st.warning(f"Semantic Scholar search failed: {e}")

                        # Crossref search
                        if source in ["Crossref", "All Sources"]:
                            try:
                                crossref_result = search_crossref(query=query, rows=min(max_results, 100))
                                crossref_papers = crossref_result.get("items", [])
                                if crossref_papers:
                                    papers.extend(crossref_papers)
                                    st.info(f"📊 Crossref found {crossref_result.get('total', 0)} total papers")
                            except Exception as e:
                                st.warning(f"Crossref search failed: {e}")

                        # OpenAlex search
                        if source in ["OpenAlex", "All Sources"]:
                            try:
                                openalex_result = search_openalex(query=query, per_page=min(max_results, 200))
                                openalex_papers = openalex_result.get("items", [])
                                if openalex_papers:
                                    papers.extend(openalex_papers)
                                    st.info(f"📊 OpenAlex found {openalex_result.get('total', 0)} total papers")
                            except Exception as e:
                                st.warning(f"OpenAlex search failed: {e}")

                        # CORE search
                        if source in ["CORE", "All Sources"]:
                            try:
                                core_result = search_core(query=query, limit=min(max_results, 100))
                                core_error = core_result.get("error")
                                if core_error:
                                    st.warning(f"CORE API issue: {core_error}")
                                core_papers = core_result.get("items", [])
                                if core_papers:
                                    papers.extend(core_papers)
                                    st.info(f"📊 CORE found {core_result.get('total', 0)} total papers")
                            except Exception as e:
                                st.warning(f"CORE search failed: {e}")

                        # Filter by year
                        if year_filter:
                            papers = [p for p in papers if p.get("year") == year_filter]

                        # Filter PDF only — use _resolve_pdf_url which handles all URL fields
                        if pdf_only:
                            papers = [p for p in papers if _resolve_pdf_url(p)]

                        # Remove duplicates across DOI/title/source records
                        unique_papers = deduplicate_papers(papers)

                        # Rank papers by relevance
                        if query and unique_papers:
                            unique_papers = rank_papers_by_relevance(
                                query=query, papers=unique_papers,
                                use_semantic=True, use_keyword=True
                            )
                            unique_papers = add_quality_labels(query, unique_papers)

                        # ✅ KEY FIX: persist results in session_state so they survive reruns
                        st.session_state["search_results"] = unique_papers
                        st.session_state["search_results_query"] = query
                        st.session_state["search_performed"] = True

                    except Exception as e:
                        st.error(f"Search error: {e}")
                        st.exception(e)

        # ✅ Display results from session_state (persists across button-click reruns)
        if st.session_state.get("search_results"):
            unique_papers = st.session_state["search_results"]
            saved_query = st.session_state.get("search_results_query", "")

            col_info, col_sort, col_clear = st.columns([3, 2, 1])
            with col_info:
                st.success(f"Found {len(unique_papers)} unique papers! (Ranked by relevance)")
            with col_sort:
                sort_results_by = st.selectbox(
                    "Sort by",
                    ["Relevance (Best Match)", "Year (newest)", "Year (oldest)", "Citations"],
                    key="sort_results"
                )
            with col_clear:
                if st.button("🗑️ Clear", key="clear_search_results"):
                    st.session_state.pop("search_results", None)
                    st.session_state.pop("search_results_query", None)
                    st.rerun()

            # Sort a copy so session_state order is preserved
            display_papers = list(unique_papers)
            if sort_results_by == "Year (newest)":
                display_papers.sort(key=lambda x: x.get("year") or 0, reverse=True)
            elif sort_results_by == "Year (oldest)":
                display_papers.sort(key=lambda x: x.get("year") or 0)
            elif sort_results_by == "Citations":
                display_papers.sort(key=lambda x: x.get("citation_count", 0), reverse=True)

            for i, paper in enumerate(display_papers, 1):
                display_paper_card_with_ranking(paper, query=saved_query, rank=i)

        elif st.session_state.get("search_performed") and not st.session_state.get("search_results"):
            st.warning("⚠️ No papers found for this query.")
            st.info("Try adjusting your filters (e.g., uncheck 'PDF available only' or use fewer keywords).")

    elif search_type == "Literature Review":
        st.markdown("### Literature Review Search")
        st.caption("Expands your topic into review/survey-style queries, searches ArXiv, Semantic Scholar, Crossref, OpenAlex, and CORE, then deduplicates and ranks papers for review usefulness.")

        col1, col2 = st.columns([3, 1])
        with col1:
            review_query = st.text_input(
                "Enter review topic:",
                placeholder="e.g., retrieval augmented generation evaluation, federated learning privacy",
                key="review_query"
            )
        with col2:
            review_field = st.text_input("Field/domain (optional)", placeholder="e.g., Computer Science")

        with st.expander("Review Search Controls", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                use_year_window = st.checkbox("Limit publication years", value=False)
                if use_year_window:
                    review_year_start = st.number_input(
                        "From year",
                        min_value=1900,
                        max_value=CURRENT_YEAR,
                        value=max(1900, CURRENT_YEAR - 6),
                        key="review_year_start"
                    )
                    review_year_end = st.number_input(
                        "To year",
                        min_value=1900,
                        max_value=CURRENT_YEAR,
                        value=CURRENT_YEAR,
                        key="review_year_end"
                    )
                else:
                    review_year_start = None
                    review_year_end = None
            with col2:
                review_min_citations = st.number_input(
                    "Minimum citations",
                    min_value=0,
                    value=0,
                    step=5,
                    key="review_min_citations"
                )
                review_pdf_only = st.checkbox("PDF available only", value=False, key="review_pdf_only")
            with col3:
                review_max_results = st.slider("Review candidates", 10, 75, 30, key="review_max_results")
                review_expansion_depth = st.slider("Query expansion depth", 1, 8, 5, key="review_expansion_depth")

        if st.button("Build Review Paper List", type="primary", key="review_search_btn"):
            if review_query.strip():
                with st.spinner("Searching review candidates across ArXiv, Semantic Scholar, Crossref, OpenAlex, and CORE..."):
                    result = search_literature_review(
                        review_query.strip(),
                        field=review_field.strip() if review_field.strip() else None,
                        max_results=review_max_results,
                        year_start=review_year_start,
                        year_end=review_year_end,
                        min_citations=review_min_citations,
                        pdf_only=review_pdf_only,
                        max_expanded_queries=review_expansion_depth,
                    )
                    st.session_state["search_results"] = result.get("papers", [])
                    st.session_state["search_results_query"] = review_query.strip()
                    st.session_state["review_search_meta"] = result
            else:
                st.warning("Enter a review topic first.")

        review_results = st.session_state.get("search_results", [])
        review_meta = st.session_state.get("review_search_meta", {})

        if review_meta:
            warnings = review_meta.get("warnings", [])
            source_counts = review_meta.get("source_counts", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("Raw source results", review_meta.get("total_before_dedup", 0))
            col2.metric("After dedupe", review_meta.get("total_after_dedup", 0))
            col3.metric("Review candidates", len(review_results))

            if source_counts:
                st.caption(
                    "Sources: "
                    + ", ".join(f"{source}: {count}" for source, count in sorted(source_counts.items()))
                )
            if review_meta.get("expanded_queries"):
                with st.expander("Expanded Queries Used", expanded=False):
                    for expanded_query in review_meta["expanded_queries"]:
                        st.write(f"- {expanded_query}")
            if warnings:
                with st.expander("API Warnings", expanded=True):
                    for warning in warnings:
                        st.warning(warning)
                    if any("429" in warning or "rate limited" in warning.lower() for warning in warnings):
                        st.info("Semantic Scholar is stricter without an API key. Add SEMANTIC_SCHOLAR_API_KEY in .env and restart Streamlit for higher quota.")

        if review_results:
            col_info, col_sort, col_clear = st.columns([3, 2, 1])
            with col_info:
                st.success(f"Prepared {len(review_results)} review candidates.")
            with col_sort:
                review_sort = st.selectbox(
                    "Sort review list by",
                    ["Review Score", "Citations", "Year (newest)", "Title"],
                    key="review_sort_results"
                )
            with col_clear:
                if st.button("Clear", key="clear_review_results"):
                    st.session_state["search_results"] = []
                    st.session_state["search_results_query"] = ""
                    st.session_state["review_search_meta"] = {}
                    st.rerun()

            display_papers = list(review_results)
            if review_sort == "Citations":
                display_papers.sort(key=lambda item: item.get("citation_count", 0), reverse=True)
            elif review_sort == "Year (newest)":
                display_papers.sort(key=lambda item: item.get("year") or 0, reverse=True)
            elif review_sort == "Title":
                display_papers.sort(key=lambda item: item.get("title", ""))
            else:
                display_papers.sort(key=lambda item: item.get("review_score", 0.0), reverse=True)

            export_df = review_papers_to_dataframe(display_papers)
            export_slug = re.sub(r"[^a-z0-9]+", "_", st.session_state.get("search_results_query", "review").lower()).strip("_") or "review"
            col_csv, col_md = st.columns(2)
            with col_csv:
                st.download_button(
                    "Export Review List CSV",
                    data=export_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{export_slug}_review_papers.csv",
                    mime="text/csv",
                    key="review_csv_export"
                )
            with col_md:
                st.download_button(
                    "Export Review List Markdown",
                    data=review_papers_to_markdown(display_papers).encode("utf-8"),
                    file_name=f"{export_slug}_review_papers.md",
                    mime="text/markdown",
                    key="review_md_export"
                )

            saved_query = st.session_state.get("search_results_query", "")
            for index, paper in enumerate(display_papers, 1):
                display_paper_card_with_ranking(paper, query=saved_query, rank=index)
        elif review_meta:
            st.warning("No review candidates matched the filters. Try removing PDF-only/min-citation filters or reducing the year restriction.")


    elif search_type == "Author":
        author = st.text_input("Enter author name:", placeholder="e.g., Geoffrey Hinton")
        source_author = st.selectbox("Source", ["Both", "ArXiv", "Semantic Scholar"], key="author_source")
        if st.button("🔍 Search", type="primary"):
            if author:
                with st.spinner(f"Searching for papers by {author}..."):
                    try:
                        papers = []

                        # ArXiv author search (external API)
                        if source_author in ["Both", "ArXiv"]:
                            try:
                                arxiv_result = search_arxiv_enhanced(
                                    query=author,
                                    field="au",
                                    max_results=20,
                                    sort_by="relevance",
                                    sort_order="descending"
                                )
                                arxiv_error = arxiv_result.get("error")
                                if arxiv_error:
                                    st.warning(f"ArXiv API issue: {arxiv_error}")
                                arxiv_papers = arxiv_result.get("entries", [])
                                papers.extend(arxiv_papers)
                                st.info(f"📊 ArXiv found {len(arxiv_papers)} papers for author search")
                            except Exception as e:
                                st.warning(f"ArXiv author search failed: {e}")

                        # Semantic Scholar author-oriented search (external API)
                        if source_author in ["Both", "Semantic Scholar"]:
                            try:
                                author_query = " ".join(author.lower().split())
                                author_query_tokens = [token for token in author_query.split() if token]

                                def author_match_score(name: str) -> int:
                                    normalized_name = " ".join((name or "").lower().split())
                                    if not normalized_name:
                                        return 0
                                    if normalized_name == author_query:
                                        return 3
                                    if author_query and author_query in normalized_name:
                                        return 2
                                    if author_query_tokens and all(token in normalized_name for token in author_query_tokens):
                                        return 1
                                    return 0

                                author_result = search_authors(author, limit=10)
                                author_error = author_result.get("error")
                                if author_error:
                                    st.warning(f"Semantic Scholar author lookup issue: {author_error}")

                                author_profiles = author_result.get("data", [])
                                ranked_profiles = sorted(
                                    author_profiles,
                                    key=lambda profile: (
                                        author_match_score(profile.get("name", "")),
                                        profile.get("paper_count", 0),
                                        profile.get("citation_count", 0)
                                    ),
                                    reverse=True
                                )

                                selected_profiles = [p for p in ranked_profiles if author_match_score(p.get("name", "")) > 0][:3]
                                if not selected_profiles and ranked_profiles:
                                    selected_profiles = ranked_profiles[:1]

                                semantic_papers = []
                                profile_fetch_errors = []

                                for profile in selected_profiles:
                                    author_id = profile.get("author_id")
                                    if not author_id:
                                        continue

                                    profile_papers = get_author_papers(author_id=author_id, limit=20)
                                    if profile_papers.get("error"):
                                        profile_fetch_errors.append(
                                            f"{profile.get('name', 'Unknown')}: {profile_papers.get('error')}"
                                        )
                                        continue

                                    semantic_papers.extend(profile_papers.get("data", []))

                                if not semantic_papers:
                                    fallback_result = search_papers_enhanced(query=author, limit=20)
                                    fallback_error = fallback_result.get("error")
                                    if fallback_error:
                                        profile_fetch_errors.append(f"fallback paper search: {fallback_error}")
                                    semantic_papers = fallback_result.get("data", [])
                                    semantic_papers = [
                                        p for p in semantic_papers
                                        if author_query in (p.get("authors_string", "").lower())
                                    ]

                                for err in profile_fetch_errors:
                                    st.warning(f"Semantic Scholar API issue: {err}")

                                papers.extend(semantic_papers)
                                if selected_profiles:
                                    st.info(
                                        f"📊 Semantic Scholar matched {len(selected_profiles)} author profile(s) "
                                        f"and returned {len(semantic_papers)} papers"
                                    )
                                else:
                                    st.info(f"📊 Semantic Scholar returned {len(semantic_papers)} author-matched papers")
                            except Exception as e:
                                st.warning(f"Semantic Scholar author search failed: {e}")

                        # Deduplicate by title
                        seen_titles = set()
                        unique_papers = []
                        for paper in papers:
                            title_lower = paper.get("title", "").lower()
                            if title_lower and title_lower not in seen_titles:
                                seen_titles.add(title_lower)
                                unique_papers.append(paper)

                        papers = unique_papers
                        if papers:
                            st.success(f"Found {len(papers)} papers!")
                            for paper in papers:
                                display_paper_card(paper)
                        else:
                            st.warning("No papers found.")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    elif search_type == "Year":
        year = st.number_input("Enter year:", min_value=1900, max_value=CURRENT_YEAR, value=CURRENT_YEAR)
        source_year = st.selectbox("Source", ["Both", "ArXiv", "Semantic Scholar"], key="year_source")
        semantic_year_query = st.text_input(
            "Semantic Scholar topic/query for this year",
            value="machine learning",
            help="Semantic Scholar requires a query; year is applied as a filter."
        )
        
        if st.button("🔍 Search", type="primary"):
            with st.spinner(f"Searching papers from {year}..."):
                try:
                    papers = []
                    
                    # ArXiv search by year
                    if source_year in ["Both", "ArXiv"]:
                        arxiv_result = search_arxiv_enhanced(
                            query=None,
                            max_results=20,
                            submitted_date_start=f"{year}01010000",
                            submitted_date_end=f"{year}12312359",
                            sort_by="submittedDate",
                            sort_order="descending"
                        )
                        arxiv_error = arxiv_result.get("error")
                        if arxiv_error:
                            st.warning(f"ArXiv API issue: {arxiv_error}")
                        arxiv_entries = arxiv_result.get("entries", [])
                        papers.extend(arxiv_entries)
                        st.info(f"📊 ArXiv found {arxiv_result.get('total', 0)} total papers for {year}")
                    
                    # Semantic Scholar search by year
                    if source_year in ["Both", "Semantic Scholar"]:
                        semantic_result = search_papers_enhanced(
                            query=semantic_year_query.strip() if semantic_year_query.strip() else "machine learning",
                            limit=20,
                            year=str(year)
                        )
                        semantic_error = semantic_result.get("error")
                        if semantic_error:
                            st.warning(f"Semantic Scholar API issue: {semantic_error}")
                        semantic_papers = semantic_result.get("data", [])
                        papers.extend(semantic_papers)
                        st.info(f"📊 Semantic Scholar found {semantic_result.get('total', 0)} total papers for {year}")

                    # Remove duplicates
                    seen_titles = set()
                    unique_papers = []
                    for paper in papers:
                        title_lower = paper.get("title", "").lower()
                        if title_lower and title_lower not in seen_titles:
                            seen_titles.add(title_lower)
                            unique_papers.append(paper)
                    papers = unique_papers
                    
                    if papers:
                        st.success(f"Found {len(papers)} papers!")
                        for paper in papers:
                            display_paper_card(paper)
                    else:
                        st.warning(f"No papers found from {year}.")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    elif search_type == "Field":
        st.markdown("### Search by ArXiv Category")
        category = st.text_input("Enter category:", placeholder="e.g., cs.AI, math.CO, physics.quant-ph")
        
        if st.button("🔍 Search by Category", type="primary"):
            if category:
                with st.spinner(f"Searching ArXiv category {category}..."):
                    try:
                        papers = search_arxiv_by_category(category, max_results=20)
                        if papers:
                            st.success(f"Found {len(papers)} papers!")
                            for paper in papers:
                                display_paper_card(paper)
                        else:
                            st.warning(f"No papers found in category {category}.")
                    except Exception as e:
                        st.error(f"Error: {e}")

# Tab 3: My Library
with tab3:
    st.markdown("### 📚 My Paper Library")
    
    library_papers = get_library_papers()
    
    if library_papers:
        st.info(f"You have {len(library_papers)} papers in your library")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox("Sort by", ["Year (newest)", "Year (oldest)", "Title", "Author"])
        with col2:
            source_filter = st.selectbox("Filter by source", ["All", "ArXiv", "Semantic Scholar", "Uploaded"])
        with col3:
            search_library = st.text_input("🔍 Search in library", placeholder="Search titles, authors...")
        
        # Apply filters
        filtered = library_papers.copy()
        if source_filter != "All":
            filtered = [p for p in filtered if p.get("source", "").lower() == source_filter.lower()]
        if search_library:
            search_lower = search_library.lower()
            filtered = [p for p in filtered if 
                       search_lower in p.get("title", "").lower() or 
                       search_lower in p.get("authors", "").lower()]
        
        # Sort
        if "Year" in sort_by:
            filtered.sort(key=lambda x: x.get("year") or 0, reverse=("newest" in sort_by))
        elif sort_by == "Title":
            filtered.sort(key=lambda x: x.get("title", ""))
        elif sort_by == "Author":
            filtered.sort(key=lambda x: x.get("authors", ""))
        
        st.write(f"Showing {len(filtered)} papers")
        
        for paper in filtered:
            with st.expander(f"📄 {paper.get('title', 'Unknown')}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Authors:** {paper.get('authors', 'Unknown')}")
                    st.write(f"**Year:** {paper.get('year', 'N/A')}")
                    st.write(f"**Source:** {paper.get('source', 'unknown')}")
                    st.write(f"**Chunks:** {paper.get('chunk_count', 0)}")
                    if paper.get("abstract"):
                        st.write(f"**Abstract:** {paper.get('abstract')[:200]}...")
                with col2:
                    if paper.get("pdf_url"):
                        st.link_button("📥 PDF", paper["pdf_url"])
                    if st.button("🗑️ Remove", key=f"remove_{paper.get('paper_id')}"):
                        removed = remove_paper_from_library(paper.get("paper_id", ""))
                        if removed:
                            st.success("✅ Paper removed from library")
                            st.rerun()
                        else:
                            st.warning("Paper not found in library")
    else:
        st.info("Your library is empty. Search and add papers to get started!")

# Tab 4: RAG Chat
with tab4:
    st.markdown("### 💬 RAG Chat Interface")
    st.markdown("Ask questions about your research papers. Papers are fetched automatically if needed.")
    
    # Paper selection
    st.subheader("📚 Select Papers")
    paper_selection_mode = st.radio(
        "Chat with:",
        ["All Library", "Selected Papers", "Fetch New Papers"],
        horizontal=True
    )
    
    selected_paper_ids = []
    if paper_selection_mode == "Selected Papers":
        library = get_library_papers()
        if library:
            paper_options = {f"{p['title'][:50]}... ({p.get('year', 'N/A')})": p['paper_id'] for p in library}
            selected = st.multiselect("Choose papers:", list(paper_options.keys()))
            selected_paper_ids = [paper_options[s] for s in selected]
        else:
            st.warning("No papers in library. Add papers first or use 'Fetch New Papers' mode.")
    
    # Chat interface
    st.divider()
    st.subheader("💬 Chat")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat["query"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            if chat.get("citations"):
                with st.expander("📚 Citations"):
                    for j, citation in enumerate(chat["citations"][:5], 1):
                        st.write(f"[{j}] Paper: {citation.get('paper_id', 'Unknown')}")
    
    # Query input
    query = st.chat_input("Ask a question about research papers...")
    
    if query:
        # Add user message
        st.chat_message("user").write(query)
        
        # Generate response
        with st.chat_message("assistant"):
            graph_status = st.status("Starting research workflow...", expanded=True)

            def report_graph_progress(stage: str, message: str) -> None:
                graph_status.write(message)

            try:
                result = query_rag(
                    query=query,
                    top_k=top_k,
                    fetch_papers=(paper_selection_mode == "Fetch New Papers"),
                    use_enhanced=use_enhanced,
                    selected_paper_ids=(
                        selected_paper_ids
                        if paper_selection_mode == "Selected Papers"
                        else None
                    ),
                    thread_id=st.session_state.graph_thread_id,
                    progress_callback=report_graph_progress,
                )
                graph_status.update(
                    label="Research workflow complete",
                    state="complete",
                    expanded=False,
                )
                    
                # Display answer
                st.write(result["answer"])
                    
                # Display citations
                if result.get("citations"):
                    with st.expander("📚 Citations & Sources"):
                        unique_papers = {}
                        for citation in result["citations"]:
                            pid = citation["paper_id"]
                            if pid not in unique_papers:
                                unique_papers[pid] = {
                                    "paper_id": pid,
                                    "chunks": [],
                                    "scores": []
                                }
                            unique_papers[pid]["chunks"].append(citation.get("chunk_index"))
                            unique_papers[pid]["scores"].append(citation.get("score", 0))
                            
                        for pid, info in unique_papers.items():
                            avg_score = sum(info["scores"]) / len(info["scores"]) if info["scores"] else 0
                            st.markdown(f"""
                            <div class="citation">
                                <strong>Paper ID:</strong> {pid}<br>
                                <strong>Chunks cited:</strong> {len(info['chunks'])}<br>
                                <strong>Relevance:</strong> {avg_score:.2%}
                            </div>
                            """, unsafe_allow_html=True)

                workflow = result.get("workflow") or {}
                if workflow:
                    with st.expander("Workflow diagnostics", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Retrieval attempts", workflow.get("retrieval_attempts", 0))
                        col2.metric("External searches", workflow.get("external_searches", 0))
                        col3.metric("Answer retries", workflow.get("answer_retries", 0))
                        evidence_grade = workflow.get("evidence_grade") or {}
                        answer_grade = workflow.get("answer_grade") or {}
                        if evidence_grade:
                            st.write(
                                f"Evidence: {evidence_grade.get('reason', 'N/A')} "
                                f"({evidence_grade.get('method', 'unknown')})"
                            )
                        if answer_grade:
                            st.write(
                                f"Validation: {answer_grade.get('reason', 'N/A')} "
                                f"({answer_grade.get('method', 'unknown')})"
                            )
                        for warning in workflow.get("warnings", []):
                            st.warning(warning)
                    
                # Save to history
                st.session_state.chat_history.append({
                    "query": query,
                    "answer": result["answer"],
                    "citations": result.get("citations", []),
                    "workflow": workflow,
                    "timestamp": datetime.now().isoformat()
                })
                    
            except Exception as e:
                graph_status.update(
                    label="Research workflow failed",
                    state="error",
                    expanded=True,
                )
                st.error(f"Error: {str(e)}")
                st.exception(e)

# Tab 5: Advanced Search
with tab5:
    st.markdown("### 🔎 Advanced Search")
    
    search_mode = st.radio("Search Mode:", ["Semantic Search", "Snippet Search", "Batch Paper Lookup"], horizontal=True)
    
    if search_mode == "Semantic Search":
        st.markdown("Search your library using semantic similarity")
        
        semantic_query = st.text_input("Enter semantic search query:", placeholder="e.g., attention mechanisms in transformers")
        top_k_semantic = st.slider("Top K results", 5, 50, 10)
        
        if st.button("🔍 Search Semantically", type="primary"):
            if semantic_query:
                with st.spinner("Performing semantic search..."):
                    try:
                        from processing.embeddings import generate_embedding
                        from vectorstore.query import query_vectors
                        
                        query_embedding = generate_embedding(semantic_query)
                        results = query_vectors(query_embedding, top_k=top_k_semantic)
                        
                        if results:
                            st.success(f"Found {len(results)} relevant chunks!")
                            for i, result in enumerate(results, 1):
                                with st.expander(f"Result {i} - Score: {result.score:.2%} (Paper: {result.paper_id})"):
                                    st.write(f"**Chunk Index:** {result.chunk_index}")
                                    st.write(f"**Text:**")
                                    st.text(result.text[:500] + "..." if len(result.text) > 500 else result.text)
                        else:
                            st.warning("No results found.")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    elif search_mode == "Snippet Search":
        st.markdown("Search for text snippets within papers using Semantic Scholar")
        
        snippet_query = st.text_input("Enter snippet search query:", placeholder="e.g., The literature graph is a property graph")
        snippet_limit = st.slider("Max snippets", 5, 50, 10)
        
        paper_ids_input = st.text_input("Paper IDs (optional, comma-separated):", placeholder="Leave empty to search all papers")
        paper_ids = [pid.strip() for pid in paper_ids_input.split(",") if pid.strip()] if paper_ids_input else None
        
        if st.button("🔍 Search Snippets", type="primary"):
            if snippet_query:
                with st.spinner("Searching for text snippets..."):
                    try:
                        snippets = search_snippets(snippet_query, limit=snippet_limit, paper_ids=paper_ids)
                        
                        if snippets:
                            st.success(f"Found {len(snippets)} snippet matches!")
                            for i, snippet in enumerate(snippets, 1):
                                with st.expander(f"Snippet {i} - Score: {snippet.get('score', 0):.2f} - {snippet.get('paper_title', 'Unknown')}"):
                                    st.write(f"**Section:** {snippet.get('section', 'Unknown')}")
                                    st.write(f"**Kind:** {snippet.get('snippet_kind', 'Unknown')}")
                                    st.write(f"**Text:**")
                                    st.text(snippet.get("text", ""))
                        else:
                            st.warning("No snippets found.")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    elif search_mode == "Batch Paper Lookup":
        st.markdown("Get details for multiple papers at once (up to 500)")
        
        paper_ids_text = st.text_area(
            "Enter paper IDs (one per line or comma-separated):",
            placeholder="649def34f8be52c8b66281af98ae884c09aef38b\nARXIV:2106.15928\nDOI:10.18653/v1/N18-3011",
            height=100
        )
        
        fields = st.text_input("Fields to fetch (comma-separated):", value="title,authors,abstract,year,openAccessPdf,citationCount")
        
        if st.button("📥 Fetch Papers", type="primary"):
            if paper_ids_text:
                # Parse paper IDs
                paper_ids = []
                for line in paper_ids_text.split("\n"):
                    paper_ids.extend([pid.strip() for pid in line.split(",") if pid.strip()])
                
                paper_ids = paper_ids[:500]  # Limit to 500
                
                with st.spinner(f"Fetching {len(paper_ids)} papers..."):
                    try:
                        from ingestion.semantic_scholar_enhanced import batch_get_papers
                        papers = batch_get_papers(paper_ids, fields=fields)
                        
                        if papers:
                            st.success(f"Fetched {len(papers)} papers!")
                            for paper in papers:
                                display_paper_card({
                                    "paper_id": paper.get("paperId"),
                                    "title": paper.get("title", "Unknown"),
                                    "authors": ", ".join([a.get("name", "") for a in paper.get("authors", [])]),
                                    "year": paper.get("year"),
                                    "abstract": paper.get("abstract", ""),
                                    "pdf_url": paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None,
                                    "source": "semantic_scholar"
                                })
                        else:
                            st.warning("No papers found.")
                    except Exception as e:
                        st.error(f"Error: {e}")

# Tab 6: Upload Paper
with tab6:
    st.markdown("### 📤 Upload & Process Papers")
    
    upload_method = st.radio("Upload method:", ["PDF File", "PDF URL", "ArXiv ID"], horizontal=True)
    
    if upload_method == "PDF File":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file:
            uploaded_file_id = st.text_input(
                "Paper ID (optional)",
                placeholder="Leave empty for auto-generate",
                key="pdf_file_paper_id"
            )
            if st.button("📥 Process Uploaded PDF", type="primary"):
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_path = temp_file.name

                    with st.spinner("Processing uploaded PDF..."):
                        result_id = ingest_pdf_from_file(
                            temp_path,
                            paper_id=uploaded_file_id if uploaded_file_id else None,
                            metadata={
                                "title": uploaded_file.name.rsplit(".", 1)[0],
                                "source": "upload"
                            }
                        )
                    st.success(f"✅ Paper processed! ID: {result_id}")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
    
    elif upload_method == "PDF URL":
        pdf_url = st.text_input("Enter PDF URL:", placeholder="https://arxiv.org/pdf/...")
        paper_id = st.text_input("Paper ID (optional):", placeholder="Leave empty for auto-generate")
        
        if st.button("📥 Process PDF", type="primary"):
            if pdf_url:
                with st.spinner("Processing PDF..."):
                    try:
                        progress = st.progress(0)
                        status = st.empty()
                        
                        status.text("Downloading...")
                        progress.progress(25)
                        
                        status.text("Extracting text...")
                        progress.progress(50)
                        
                        status.text("Chunking...")
                        progress.progress(75)
                        
                        result_id = ingest_pdf_from_url(pdf_url, paper_id=paper_id if paper_id else None)
                        
                        progress.progress(100)
                        status.text("Complete!")
                        
                        st.success(f"✅ Paper processed! ID: {result_id}")
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
    
    elif upload_method == "ArXiv ID":
        arxiv_id = st.text_input("Enter ArXiv ID:", placeholder="e.g., 1706.03762")
        if st.button("📥 Fetch & Process", type="primary"):
            if arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                with st.spinner(f"Fetching and processing {arxiv_id}..."):
                    try:
                        result_id = ingest_pdf_from_url(pdf_url, paper_id=arxiv_id)
                        st.success(f"✅ Paper processed! ID: {result_id}")
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Processing failed: {e}")

# Tab 7: Analysis
with tab7:
    st.markdown("### 📊 Analysis Tools")
    st.info("💡 **Citation Network Visualization** is available below! Select 'Citation Graph' from the dropdown to visualize paper relationships.")
    
    analysis_type = st.selectbox("Choose analysis:", [
        "Paper Summaries",
        "Citation Rankings",
        "Author Statistics",
        "Topic Clustering",
        "Citation Graph"
    ])
    
    if analysis_type == "Paper Summaries":
        st.subheader("📄 Generate Paper Summaries")
        library = get_library_papers()
        if library:
            paper_options = {f"{p['title'][:50]}...": p['paper_id'] for p in library}
            selected_paper = st.selectbox("Select paper:", list(paper_options.keys()))
            if st.button("Generate Summary"):
                paper_id = paper_options[selected_paper]
                with st.spinner("Generating summary..."):
                    try:
                        summary = api.generate_summary(paper_id, use_llm=False)
                        st.write("**Short Summary:**")
                        st.write(summary.get("short", ""))
                        st.write("**Medium Summary:**")
                        st.write(summary.get("medium", ""))
                        if summary.get("bullets"):
                            st.write("**Key Points:**")
                            for bullet in summary["bullets"]:
                                st.write(f"• {bullet}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    elif analysis_type == "Citation Rankings":
        st.subheader("📊 Citation Rankings")
        if st.button("Calculate Rankings"):
            with st.spinner("Calculating citation metrics..."):
                try:
                    rankings = api.get_citation_rankings()
                    papers = rankings.get("ranked_papers", [])
                    if papers:
                        df = pd.DataFrame(papers[:20])
                        st.dataframe(df[["title", "citation_score", "incoming_citations", "year"]])
                    else:
                        st.info("No citation data available")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    elif analysis_type == "Author Statistics":
        st.subheader("👥 Author Statistics")
        if st.button("Get Statistics"):
            with st.spinner("Analyzing authors..."):
                try:
                    stats = api.get_author_statistics()
                    st.write(f"**Total Authors:** {stats.get('total_authors', 0)}")
                    if stats.get("top_authors"):
                        df = pd.DataFrame(stats["top_authors"][:20])
                        st.dataframe(df)
                except Exception as e:
                    st.error(f"Error: {e}")
    
    elif analysis_type == "Topic Clustering":
        st.subheader("🎯 Topic Clustering")
        num_clusters = st.slider("Number of clusters", 2, 10, 5)
        if st.button("Cluster Papers"):
            with st.spinner("Clustering papers..."):
                try:
                    clusters = api.cluster_topics(num_clusters=num_clusters)
                    if clusters:
                        for cluster_id, cluster_info in clusters.items():
                            with st.expander(f"📁 {cluster_info['topic']} ({cluster_info['paper_count']} papers)"):
                                for paper in cluster_info["papers"]:
                                    st.write(f"• {paper['title']}")
                    else:
                        st.info("Not enough papers for clustering")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    elif analysis_type == "Citation Graph":
        st.subheader("🕸️ Citation Network Visualization")
        st.markdown("Visualize citation relationships between papers in your library")
        
        # Get papers from library
        library = get_library_papers()
        
        if not library:
            st.warning("No papers in library. Add papers first to visualize citation network.")
        else:
            # Paper selection
            col1, col2 = st.columns([3, 1])
            with col1:
                paper_options = {f"{p.get('title', 'Unknown')[:60]}... ({p.get('paper_id', '')[:8]})": p.get('paper_id') 
                                for p in library}
                selected_papers = st.multiselect(
                    "Select papers to visualize (select 1-10 papers):",
                    options=list(paper_options.keys()),
                    default=list(paper_options.keys())[:min(5, len(paper_options))]
                )
            
            with col2:
                max_depth = st.slider("Max Depth", 1, 3, 2, help="How many levels of citations to show")
                max_nodes = st.slider("Max Nodes", 20, 100, 50, help="Maximum papers in graph")
                layout_type = st.selectbox("Layout", ["spring", "circular", "kamada_kawai"], 
                                         help="Graph layout algorithm")
            
            if selected_papers:
                paper_ids = [paper_options[p] for p in selected_papers]
                
                if st.button("🕸️ Generate Citation Network", type="primary"):
                    with st.spinner("Building citation network..."):
                        try:
                            # Generate visualization
                            fig = visualize_citation_network(
                                paper_ids=paper_ids,
                                max_depth=max_depth,
                                max_nodes=max_nodes,
                                layout=layout_type
                            )
                            
                            # Display graph
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Build graph for statistics
                            graph = build_citation_graph(paper_ids, max_depth=max_depth, max_nodes=max_nodes)
                            
                            # Show statistics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Papers", len(graph.nodes))
                            with col2:
                                st.metric("Citations", len(graph.edges))
                            with col3:
                                density = nx.density(graph) if len(graph.nodes) > 1 else 0
                                st.metric("Network Density", f"{density:.3f}")
                            
                            # Tabs for additional insights
                            tab1, tab2, tab3 = st.tabs(["📊 Influential Papers", "👥 Research Communities", "📈 Network Statistics"])
                            
                            with tab1:
                                st.markdown("### Most Influential Papers (PageRank)")
                                influential = get_influential_papers(graph, top_k=10)
                                if influential:
                                    for i, paper in enumerate(influential, 1):
                                        with st.expander(f"#{i} - {paper['title'][:60]}... (Score: {paper['influence_score']})"):
                                            st.write(f"**Authors:** {paper.get('authors', 'Unknown')}")
                                            st.write(f"**Year:** {paper.get('year', 'N/A')}")
                                            st.write(f"**Influence Score:** {paper['influence_score']}")
                                            st.write(f"**Citations:** {paper.get('citation_count', 0)}")
                                else:
                                    st.info("Not enough data for influence analysis")
                            
                            with tab2:
                                st.markdown("### Research Communities")
                                communities = get_research_communities(graph)
                                if communities.get("num_communities", 0) > 0:
                                    st.success(f"Found {communities['num_communities']} research communities")
                                    for comm in communities.get("communities", []):
                                        with st.expander(f"Community {comm['community_id'] + 1} ({comm['size']} papers)"):
                                            for paper in comm.get("papers", [])[:10]:
                                                st.write(f"• {paper['title'][:60]}...")
                                                st.caption(f"Authors: {paper.get('authors', 'Unknown')}")
                                else:
                                    st.info("Not enough connections to detect communities")
                            
                            with tab3:
                                st.markdown("### Network Statistics")
                                stats = get_citation_statistics(graph)
                                if stats:
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Most Cited Papers:**")
                                        for paper in stats.get("most_cited", [])[:5]:
                                            st.write(f"• {paper['title'][:50]}... ({paper['in_degree']} citations)")
                                    
                                    with col2:
                                        st.write("**Most Citing Papers:**")
                                        for paper in stats.get("most_citing", [])[:5]:
                                            st.write(f"• {paper['title'][:50]}... ({paper['out_degree']} citations)")
                                    
                                    st.markdown("---")
                                    st.write(f"**Average In-Degree:** {stats.get('avg_in_degree', 0):.2f}")
                                    st.write(f"**Average Out-Degree:** {stats.get('avg_out_degree', 0):.2f}")
                                    st.write(f"**Network Density:** {stats.get('density', 0):.4f}")
                                else:
                                    st.info("No statistics available")
                            
                        except Exception as e:
                            st.error(f"Error generating citation network: {e}")
                            st.exception(e)
            else:
                st.info("Select at least one paper to visualize")

# Footer
st.divider()
st.markdown("---")
st.markdown("**ScholarX** - Research Paper RAG System | Built with Streamlit")
