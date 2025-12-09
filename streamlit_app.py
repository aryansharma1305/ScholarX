"""Comprehensive Streamlit App for ScholarX - Full Featured RAG Pipeline."""
import streamlit as st
import sys
import json
import time
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
from ingestion.ingest_pipeline import ingest_pdf_from_url
from ingestion.paper_fetcher import search_arxiv, search_semantic_scholar
from ingestion.arxiv_enhanced import (
    search_arxiv_enhanced, search_arxiv_by_author, search_arxiv_by_title,
    search_arxiv_by_category, get_arxiv_papers_by_id
)
from ingestion.semantic_scholar_enhanced import (
    search_papers_enhanced, paper_autocomplete, get_paper_details,
    get_paper_citations, get_paper_references, search_authors, search_snippets
)
from config.settings import settings
from config.chroma_client import get_collection

# Set to free mode
settings.embedding_provider = "sentence-transformers"
settings.llm_provider = "simple"

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


def get_library_papers():
    """Get all papers in the library."""
    try:
        collection = get_collection()
        all_data = collection.get(limit=10000)
        
        papers = {}
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
                    "chunk_count": 0
                }
        
        # Count chunks per paper
        for paper_id in papers:
            chunks = collection.get(where={"paper_id": paper_id}, limit=1)
            if chunks.get("ids"):
                papers[paper_id]["chunk_count"] = len(chunks["ids"])
        
        return list(papers.values())
    except Exception as e:
        st.error(f"Error loading library: {e}")
        return []


def display_paper_card_with_ranking(paper: Dict, query: str = "", rank: int = 0, show_add_button: bool = True):
    """Display a paper card with relevance ranking."""
    source = paper.get("source", "unknown")
    badge_class = "badge-arxiv" if source == "arxiv" else "badge-semantic"
    
    # Get relevance info
    relevance_score = paper.get("relevance_score", 0.0)
    relevance_percent = paper.get("relevance_percent", "N/A")
    relevance_info = get_relevance_category(relevance_score) if relevance_score > 0 and query else None
    
    # Build relevance badge HTML
    relevance_html = ""
    if relevance_info and query:
        relevance_html = f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; padding: 0.5rem; background: linear-gradient(90deg, {relevance_info['color']}20, transparent); border-left: 4px solid {relevance_info['color']}; border-radius: 4px;">
            <span style="font-size: 1.5rem;">{relevance_info['emoji']}</span>
            <div>
                <div style="font-weight: bold; color: {relevance_info['color']}; font-size: 1.1rem;">
                    #{rank} - {relevance_percent} Match
                </div>
                <div style="font-size: 0.9rem; color: #666;">
                    {relevance_info['category']}
                </div>
            </div>
        </div>
        """
    
    st.markdown(f"""
    <div class="paper-card">
        {relevance_html}
        <h3>{paper.get('title', 'Unknown')}</h3>
        <p><strong>Authors:</strong> {paper.get('authors_string', paper.get('authors', 'Unknown'))}</p>
        <p><strong>Year:</strong> {paper.get('year', 'N/A')}</p>
        <span class="badge {badge_class}">{source.upper()}</span>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    if paper.get("abstract"):
        with st.expander("📄 Abstract"):
            st.write(paper["abstract"])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if paper.get("pdf_url"):
            st.link_button("📥 Download PDF", paper["pdf_url"])
    with col2:
        if show_add_button:
            if st.button("➕ Add to Library", key=f"add_{paper.get('paper_id')}"):
                process_paper_for_rag(paper)
    with col3:
        if st.button("📊 View Details", key=f"view_{paper.get('paper_id')}"):
            st.session_state[f"view_paper_{paper.get('paper_id')}"] = paper
            st.rerun()
    
    st.divider()


def display_paper_card(paper: Dict, show_add_button: bool = True):
    """Display a paper card with all metadata."""
    display_paper_card_with_ranking(paper, query="", rank=0, show_add_button=show_add_button)
    
    if paper.get("abstract"):
        with st.expander("📄 Abstract"):
            st.write(paper["abstract"])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if paper.get("pdf_url"):
            st.link_button("📥 Download PDF", paper["pdf_url"])
    with col2:
        if show_add_button:
            if st.button("➕ Add to Library", key=f"add_{paper.get('paper_id')}"):
                process_paper_for_rag(paper)
    with col3:
        if st.button("📊 View Details", key=f"view_{paper.get('paper_id')}"):
            st.session_state[f"view_paper_{paper.get('paper_id')}"] = paper
            st.rerun()
    
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
                if paper.get("paper_id") and paper.get("source") == "semantic_scholar":
                    if st.button("🔍 Get Full Details", key=f"details_{paper.get('paper_id')}"):
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
                
                if st.button("❌ Close", key=f"close_{paper.get('paper_id')}"):
                    del st.session_state[key]
                    st.rerun()


def process_paper_for_rag(paper: Dict):
    """Process a paper for RAG."""
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        st.error("No PDF URL available")
        return
    
    task_id = paper.get("paper_id", "unknown")
    st.session_state.processing_tasks[task_id] = {
        "status": "processing",
        "paper": paper,
        "start_time": datetime.now()
    }
    
    try:
        with st.spinner(f"Processing {paper.get('title', 'paper')}..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Downloading PDF...")
            progress_bar.progress(20)
            
            status_text.text("Extracting text...")
            progress_bar.progress(40)
            
            status_text.text("Chunking content...")
            progress_bar.progress(60)
            
            status_text.text("Generating embeddings...")
            progress_bar.progress(80)
            
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
            
            status_text.text("Storing in vector database...")
            progress_bar.progress(100)
            
            st.session_state.processing_tasks[task_id]["status"] = "completed"
            st.success(f"✅ Paper processed successfully! (ID: {paper_id})")
            time.sleep(1)
            st.rerun()
            
    except Exception as e:
        st.session_state.processing_tasks[task_id]["status"] = "failed"
        st.error(f"❌ Processing failed: {e}")


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
    
    search_type = st.radio("Search by:", ["Keywords", "Author", "Year", "Field"], horizontal=True)
    
    if search_type == "Keywords":
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Enter search query:", placeholder="e.g., transformer architecture, attention mechanism", key="search_query")
        with col2:
            source = st.selectbox("Source", ["Both", "ArXiv", "Semantic Scholar"])
        
        # ArXiv query builder helper
        if source in ["Both", "ArXiv"]:
            with st.expander("🔧 ArXiv Query Builder"):
                st.markdown("**Field Prefixes:**")
                st.code("ti:title  au:author  abs:abstract  cat:category  all:everything")
                st.markdown("**Boolean Operators:**")
                st.code("AND  OR  ANDNOT")
                st.markdown("**Examples:**")
                st.code("au:Einstein AND ti:relativity\ncat:cs.AI AND abs:neural\nall:transformer ANDNOT cat:math")
                
                use_advanced = st.checkbox("Use advanced query syntax", False)
                if use_advanced:
                    query = st.text_area("Enter ArXiv query:", value=query, key="advanced_query")
                    st.caption("💡 You can use field prefixes and Boolean operators directly")
        
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
                sort_by = st.selectbox("Sort by", ["relevance", "lastUpdatedDate", "submittedDate"], key="sort_by")
            
            with col2:
                # Semantic Scholar filters
                if source in ["Both", "Semantic Scholar"]:
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
            year_filter = st.number_input("Year (optional)", min_value=1900, max_value=2025, value=None, key="year_filter")
        with col2:
            pdf_only = st.checkbox("PDF available only", True)
        with col3:
            max_results = st.slider("Max results", 5, 50, 10)
        
        # Autocomplete suggestions
        if query and len(query) > 2:
            with st.spinner("Getting suggestions..."):
                try:
                    suggestions = paper_autocomplete(query, limit=5)
                    if suggestions:
                        st.caption("💡 Suggestions:")
                        for sug in suggestions:
                            if st.button(f"📄 {sug.get('title', 'Unknown')[:60]}...", key=f"sug_{sug.get('paper_id')}"):
                                st.session_state["search_query"] = sug.get("title", query)
                                st.rerun()
                except:
                    pass
        
        if st.button("🔍 Search", type="primary"):
            if query:
                with st.spinner("Searching papers..."):
                    try:
                        papers = []
                        
                        # Enhanced ArXiv search
                        if source in ["Both", "ArXiv"]:
                            try:
                                # Determine field
                                field_map = {
                                    "all": None,
                                    "ti (title)": "ti",
                                    "au (author)": "au",
                                    "abs (abstract)": "abs",
                                    "cat (category)": "cat"
                                }
                                arxiv_field_value = field_map.get(arxiv_field, None)
                                
                                # Build query
                                arxiv_query = query
                                if arxiv_category:
                                    arxiv_query = f"{arxiv_query} AND cat:{arxiv_category}"
                                
                                arxiv_result = search_arxiv_enhanced(
                                    query=arxiv_query,
                                    max_results=max_results,
                                    field=arxiv_field_value,
                                    sort_by=sort_by,
                                    sort_order="descending"
                                )
                                arxiv_papers = arxiv_result.get("entries", [])
                                papers.extend(arxiv_papers)
                                
                                if arxiv_result.get("total", 0) > 0:
                                    st.info(f"📊 ArXiv found {arxiv_result.get('total', 0)} total papers")
                            except Exception as e:
                                st.warning(f"ArXiv search failed: {e}")
                                # Fallback to simple search
                                arxiv_papers = search_arxiv(query, max_results=max_results)
                                papers.extend(arxiv_papers)
                        
                        # Enhanced Semantic Scholar search
                        if source in ["Both", "Semantic Scholar"]:
                            try:
                                semantic_result = search_papers_enhanced(
                                    query=query,
                                    limit=max_results,
                                    year=year_range if year_range else None,
                                    fields_of_study=fields_of_study if fields_of_study else None,
                                    open_access_only=open_access_only,
                                    min_citation_count=min_citations if min_citations > 0 else None
                                )
                                semantic_papers = semantic_result.get("data", [])
                                papers.extend(semantic_papers)
                                
                                if semantic_result.get("total", 0) > 0:
                                    st.info(f"📊 Semantic Scholar found {semantic_result.get('total', 0)} total papers")
                            except Exception as e:
                                st.warning(f"Semantic Scholar search failed: {e}. Using ArXiv only.")
                        
                        # Filter by year
                        if year_filter:
                            papers = [p for p in papers if p.get("year") == year_filter]
                        
                        # Filter PDF only
                        if pdf_only:
                            papers = [p for p in papers if p.get("pdf_url")]
                        
                        # Remove duplicates
                        seen_titles = set()
                        unique_papers = []
                        for paper in papers:
                            title_lower = paper.get("title", "").lower()
                            if title_lower not in seen_titles:
                                seen_titles.add(title_lower)
                                unique_papers.append(paper)
                        
                        # Rank papers by relevance to query
                        if query and unique_papers:
                            unique_papers = rank_papers_by_relevance(
                                query=query,
                                papers=unique_papers,
                                use_semantic=True,
                                use_keyword=True
                            )
                        
                        if unique_papers:
                            st.success(f"Found {len(unique_papers)} unique papers! (Ranked by relevance)")
                            
                            # Show sorting option
                            col1, col2 = st.columns([3, 1])
                            with col2:
                                sort_by = st.selectbox("Sort by", ["Relevance (Best Match)", "Year (newest)", "Year (oldest)", "Citations"], key="sort_results")
                            
                            # Sort papers
                            if sort_by == "Year (newest)":
                                unique_papers.sort(key=lambda x: x.get("year") or 0, reverse=True)
                            elif sort_by == "Year (oldest)":
                                unique_papers.sort(key=lambda x: x.get("year") or 0)
                            elif sort_by == "Citations":
                                unique_papers.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
                            # Default: Relevance (already sorted)
                            
                            for i, paper in enumerate(unique_papers, 1):
                                # Display with relevance score
                                display_paper_card_with_ranking(paper, query=query, rank=i)
                        else:
                            st.warning("No papers found. Try different keywords.")
                    except Exception as e:
                        st.error(f"Search error: {e}")
                        st.exception(e)
    
    elif search_type == "Author":
        author = st.text_input("Enter author name:", placeholder="e.g., Geoffrey Hinton")
        if st.button("🔍 Search", type="primary"):
            if author:
                with st.spinner(f"Searching for papers by {author}..."):
                    try:
                        results = api.search(author=author, limit=20)
                        papers = results.get("papers", [])
                        if papers:
                            st.success(f"Found {len(papers)} papers!")
                            for paper in papers:
                                display_paper_card(paper)
                        else:
                            st.warning("No papers found.")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    elif search_type == "Year":
        year = st.number_input("Enter year:", min_value=1900, max_value=2025, value=2024)
        source_year = st.selectbox("Source", ["Both", "ArXiv", "Semantic Scholar"], key="year_source")
        
        if st.button("🔍 Search", type="primary"):
            with st.spinner(f"Searching papers from {year}..."):
                try:
                    papers = []
                    
                    # ArXiv search by year
                    if source_year in ["Both", "ArXiv"]:
                        arxiv_result = search_arxiv_enhanced(
                            query=f"submittedDate:[{year}01010000 TO {year}12312359]",
                            max_results=20,
                            sort_by="submittedDate",
                            sort_order="descending"
                        )
                        papers.extend(arxiv_result.get("entries", []))
                    
                    # Semantic Scholar search by year
                    if source_year in ["Both", "Semantic Scholar"]:
                        results = api.search(year=year, limit=20)
                        papers.extend(results.get("papers", []))
                    
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
                        # TODO: Implement removal
                        st.info("Removal feature coming soon")
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
            with st.spinner("Thinking..."):
                try:
                    if paper_selection_mode == "Selected Papers" and selected_paper_ids:
                        # Multi-document RAG
                        result = api.rag_multi_document(selected_paper_ids, query)
                    else:
                        # Standard RAG (fetches papers if needed)
                        result = query_rag(
                            query=query,
                            top_k=top_k,
                            fetch_papers=(paper_selection_mode == "Fetch New Papers"),
                            use_enhanced=use_enhanced
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
                    
                    # Save to history
                    st.session_state.chat_history.append({
                        "query": query,
                        "answer": result["answer"],
                        "citations": result.get("citations", []),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
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
            st.info("PDF upload feature - save file and process")
            # TODO: Implement file upload processing
    
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
        st.subheader("🕸️ Citation Graph")
        st.info("Citation graph visualization - coming soon!")
        st.write("This feature will visualize paper relationships using network graphs.")

# Footer
st.divider()
st.markdown("---")
st.markdown("**ScholarX** - Research Paper RAG System | Built with Streamlit")

