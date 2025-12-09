"""Enhanced Streamlit App - Better Than Google Scholar."""
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

from main import query_rag
from api.main_api import api
from manage_papers import get_statistics
from ingestion.ingest_pipeline import ingest_pdf_from_url
from ingestion.paper_fetcher import search_arxiv, search_semantic_scholar, fetch_papers_by_topic
from ingestion.arxiv_enhanced import search_arxiv_enhanced, search_arxiv_by_category
from ingestion.semantic_scholar_enhanced import (
    search_papers_enhanced, paper_autocomplete, get_paper_details,
    get_paper_citations, get_paper_references, search_authors
)
from ingestion.crossref_api import search_crossref, get_crossref_by_doi
from ingestion.openalex_api import search_openalex, get_openalex_work
from config.settings import settings
from config.chroma_client import get_collection

# Set to free mode
settings.embedding_provider = "sentence-transformers"
settings.llm_provider = "simple"

# Page config
st.set_page_config(
    page_title="ScholarX - Better Than Google Scholar",
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
if 'comparison_papers' not in st.session_state:
    st.session_state.comparison_papers = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = []

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .paper-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin: 1.5rem 0;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .paper-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .paper-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .paper-title a {
        color: #1f77b4;
        text-decoration: none;
    }
    .paper-title a:hover {
        text-decoration: underline;
    }
    .metadata-row {
        display: flex;
        gap: 1rem;
        margin: 0.5rem 0;
        flex-wrap: wrap;
    }
    .metadata-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.9rem;
        color: #666;
    }
    .tag-chip {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        margin: 0.25rem;
        border: 1px solid #90caf9;
    }
    .author-link {
        color: #1f77b4;
        text-decoration: none;
        cursor: pointer;
    }
    .author-link:hover {
        text-decoration: underline;
    }
    .abstract-preview {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 3px solid #28a745;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 0.1rem 0.2rem;
        border-radius: 0.2rem;
        font-weight: bold;
    }
    .action-buttons {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .comparison-card {
        background-color: #fff;
        border: 2px solid #1f77b4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def highlight_query_terms(text: str, query: str) -> str:
    """Highlight query terms in text."""
    if not query or not text:
        return text
    
    query_terms = query.lower().split()
    highlighted = text
    for term in query_terms:
        if len(term) > 2:  # Only highlight words longer than 2 chars
            import re
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(
                lambda m: f'<span class="highlight">{m.group()}</span>',
                highlighted
            )
    return highlighted


def display_enhanced_paper_card(paper: Dict, query: str = "", show_actions: bool = True):
    """Display enhanced paper card with all metadata."""
    paper_id = paper.get("paper_id", "unknown")
    title = paper.get("title", "Unknown Title")
    authors = paper.get("authors", [])
    authors_string = paper.get("authors_string", "Unknown Authors")
    abstract = paper.get("abstract", "")
    year = paper.get("year")
    source = paper.get("source", "unknown")
    pdf_url = paper.get("pdf_url")
    citation_count = paper.get("citation_count", 0)
    concepts = paper.get("concepts", [])
    doi = paper.get("doi")
    
    # Create card
    st.markdown(f"""
    <div class="paper-card">
        <div class="paper-title">
            <a href="{paper.get('url', '#')}" target="_blank">{title}</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Authors with clickable links
    col1, col2 = st.columns([3, 1])
    with col1:
        author_display = []
        for author in authors[:5]:  # Show first 5 authors
            if st.button(f"👤 {author}", key=f"author_{paper_id}_{author}", use_container_width=False):
                st.session_state[f"view_author_{author}"] = author
            author_display.append(author)
        if len(authors) > 5:
            st.caption(f"and {len(authors) - 5} more authors")
    
    with col2:
        if source == "arxiv":
            st.markdown('<span class="badge badge-arxiv">arXiv</span>', unsafe_allow_html=True)
        elif source == "semantic_scholar":
            st.markdown('<span class="badge badge-semantic">Semantic Scholar</span>', unsafe_allow_html=True)
        elif source == "crossref":
            st.markdown('<span class="badge" style="background-color: #ff6b35; color: white;">Crossref</span>', unsafe_allow_html=True)
        elif source == "openalex":
            st.markdown('<span class="badge" style="background-color: #6c5ce7; color: white;">OpenAlex</span>', unsafe_allow_html=True)
    
    # Metadata row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if year:
            st.markdown(f'📅 **Year:** {year}')
    with col2:
        if concepts:
            st.markdown(f'🏷️ **Field:** {concepts[0] if concepts else "N/A"}')
    with col3:
        st.markdown(f'📄 **Source:** {source.upper()}')
    with col4:
        if citation_count > 0:
            st.markdown(f'📈 **Citations:** {citation_count:,}')
    
    # Tags/Keywords
    if concepts:
        st.markdown("**Topics:**")
        tag_html = " ".join([f'<span class="tag-chip">{tag}</span>' for tag in concepts[:5]])
        st.markdown(tag_html, unsafe_allow_html=True)
    
    # Abstract preview with highlights
    if abstract:
        with st.expander("📄 Abstract Preview", expanded=False):
            highlighted_abstract = highlight_query_terms(abstract[:300] + "...", query)
            st.markdown(f'<div class="abstract-preview">{highlighted_abstract}</div>', unsafe_allow_html=True)
            if len(abstract) > 300:
                if st.button("📖 View Full Abstract", key=f"full_abs_{paper_id}"):
                    st.markdown(f"**Full Abstract:**\n\n{abstract}")
    
    # Action buttons
    if show_actions:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if pdf_url:
                st.link_button("📥 Download PDF", pdf_url, use_container_width=True)
            else:
                if st.button("🔍 Find PDF", key=f"find_pdf_{paper_id}", use_container_width=True):
                    st.info("Searching for open access PDF...")
                    # TODO: Implement Unpaywall API
        
        with col2:
            if st.button("➕ Add to Library", key=f"add_{paper_id}", use_container_width=True):
                st.session_state.library_papers.append(paper)
                st.success("Added to library!")
                time.sleep(1)
                st.rerun()
        
        with col3:
            if st.button("⚙️ Process for RAG", key=f"process_{paper_id}", use_container_width=True):
                process_paper_for_rag(paper)
        
        with col4:
            if st.button("🤖 Ask AI", key=f"ask_ai_{paper_id}", use_container_width=True):
                st.session_state[f"ask_ai_paper_{paper_id}"] = paper
        
        with col5:
            if st.button("📊 Compare", key=f"compare_{paper_id}", use_container_width=True):
                if paper_id not in [p.get("paper_id") for p in st.session_state.comparison_papers]:
                    st.session_state.comparison_papers.append(paper)
                    st.success(f"Added to comparison ({len(st.session_state.comparison_papers)}/3)")
    
    st.divider()


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


# Main App
st.markdown('<div class="main-header">📚 ScholarX - Better Than Google Scholar</div>', unsafe_allow_html=True)

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
    
    st.divider()
    
    st.subheader("📚 Comparison Mode")
    if st.session_state.comparison_papers:
        st.write(f"{len(st.session_state.comparison_papers)} papers selected")
        if st.button("🔄 Clear Comparison"):
            st.session_state.comparison_papers = []
            st.rerun()

# Main Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔍 Search", "📚 Library", "💬 RAG Chat", "📊 Compare Papers",
    "📤 Upload", "🔎 Advanced", "📈 Analytics"
])

# Tab 1: Enhanced Search
with tab1:
    st.markdown("### 🔍 Search Papers, Authors, or Topics")
    
    # Big search bar
    search_query = st.text_input(
        "🔍",
        placeholder="Search papers, authors or topics…",
        key="main_search",
        label_visibility="collapsed"
    )
    
    # Collapsible filters
    with st.expander("🔧 Advanced Filters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Date & Source")
            year_range = st.slider("Year Range", 1900, 2025, (2020, 2024))
            sources = st.multiselect(
                "Sources",
                ["ArXiv", "Semantic Scholar", "Crossref", "OpenAlex"],
                default=["ArXiv", "Semantic Scholar", "Crossref", "OpenAlex"]
            )
        
        with col2:
            st.subheader("📋 Content Filters")
            has_pdf = st.checkbox("Has PDF", True)
            open_access_only = st.checkbox("Open Access Only", False)
            min_citations = st.number_input("Min Citations", min_value=0, value=0)
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("👤 Author Filter")
            author_filter = st.text_input("Author Name", placeholder="e.g., Geoffrey Hinton")
        
        with col4:
            st.subheader("🏷️ Fields")
            fields = st.multiselect(
                "Fields of Study",
                ["Computer Science", "Physics", "Mathematics", "Biology", "Medicine", 
                 "Engineering", "NLP", "Computer Vision", "Machine Learning", "Robotics"],
                default=[]
            )
    
    # Search button
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if search_query:
            with st.spinner("Searching across all sources..."):
                try:
                    all_papers = []
                    
                    # Map source names
                    source_map = {
                        "ArXiv": "arxiv",
                        "Semantic Scholar": "semantic_scholar",
                        "Crossref": "crossref",
                        "OpenAlex": "openalex"
                    }
                    selected_sources = [source_map.get(s, s.lower()) for s in sources]
                    
                    # Search ArXiv
                    if "arxiv" in selected_sources:
                        try:
                            arxiv_result = search_arxiv_enhanced(
                                query=search_query,
                                max_results=20,
                                sort_by="relevance"
                            )
                            all_papers.extend(arxiv_result.get("entries", []))
                        except:
                            pass
                    
                    # Search Semantic Scholar
                    if "semantic_scholar" in selected_sources:
                        try:
                            semantic_result = search_papers_enhanced(
                                query=search_query,
                                limit=20,
                                open_access_only=open_access_only,
                                min_citation_count=min_citations if min_citations > 0 else None
                            )
                            all_papers.extend(semantic_result.get("data", []))
                        except:
                            pass
                    
                    # Search Crossref
                    if "crossref" in selected_sources:
                        try:
                            crossref_result = search_crossref(
                                query=search_query,
                                rows=20,
                                filter_dict={"has-full-text": "true"} if has_pdf else None
                            )
                            all_papers.extend(crossref_result.get("items", []))
                        except:
                            pass
                    
                    # Search OpenAlex
                    if "openalex" in selected_sources:
                        try:
                            openalex_result = search_openalex(
                                query=search_query,
                                per_page=20,
                                filter_dict={"is_oa": "true"} if open_access_only else None
                            )
                            all_papers.extend(openalex_result.get("items", []))
                        except:
                            pass
                    
                    # Apply filters
                    filtered = all_papers.copy()
                    
                    # Year filter
                    filtered = [p for p in filtered if not p.get("year") or 
                               (year_range[0] <= p.get("year", 0) <= year_range[1])]
                    
                    # PDF filter
                    if has_pdf:
                        filtered = [p for p in filtered if p.get("pdf_url")]
                    
                    # Author filter
                    if author_filter:
                        filtered = [p for p in filtered if 
                                   author_filter.lower() in p.get("authors_string", "").lower()]
                    
                    # Remove duplicates
                    seen_titles = set()
                    unique_papers = []
                    for paper in filtered:
                        title_lower = paper.get("title", "").lower()
                        if title_lower not in seen_titles:
                            seen_titles.add(title_lower)
                            unique_papers.append(paper)
                    
                    st.session_state.search_results = unique_papers
                    st.success(f"Found {len(unique_papers)} papers!")
                    
                except Exception as e:
                    st.error(f"Search error: {e}")
                    st.exception(e)
    
    # Display results
    if st.session_state.search_results:
        st.markdown(f"### 📄 Results ({len(st.session_state.search_results)} papers)")
        
        # Sort options
        col1, col2 = st.columns([3, 1])
        with col2:
            sort_option = st.selectbox("Sort by", ["Relevance", "Year (newest)", "Year (oldest)", "Citations"])
        
        # Sort papers
        sorted_papers = st.session_state.search_results.copy()
        if sort_option == "Year (newest)":
            sorted_papers.sort(key=lambda x: x.get("year") or 0, reverse=True)
        elif sort_option == "Year (oldest)":
            sorted_papers.sort(key=lambda x: x.get("year") or 0)
        elif sort_option == "Citations":
            sorted_papers.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
        
        # Display each paper
        for paper in sorted_papers:
            display_enhanced_paper_card(paper, query=search_query, show_actions=True)
    
    # Handle Ask AI for papers
    for key in list(st.session_state.keys()):
        if key.startswith("ask_ai_paper_"):
            paper = st.session_state[key]
            with st.expander(f"🤖 Ask AI About: {paper.get('title', 'Unknown')[:50]}...", expanded=True):
                ai_query = st.text_input("Your question:", key=f"ai_q_{key}")
                if st.button("Ask", key=f"ai_btn_{key}"):
                    if ai_query:
                        with st.spinner("Thinking..."):
                            try:
                                # Check if paper is processed
                                collection = get_collection()
                                paper_chunks = collection.get(
                                    where={"paper_id": paper.get("paper_id")},
                                    limit=1
                                )
                                
                                if paper_chunks.get("ids"):
                                    # Paper is in library, use RAG
                                    result = api.rag_multi_document(
                                        [paper.get("paper_id")],
                                        ai_query
                                    )
                                    st.write(result["answer"])
                                else:
                                    # Paper not processed, process it first
                                    st.info("Paper not in library. Processing first...")
                                    process_paper_for_rag(paper)
                                    time.sleep(2)
                                    result = api.rag_multi_document(
                                        [paper.get("paper_id")],
                                        ai_query
                                    )
                                    st.write(result["answer"])
                            except Exception as e:
                                st.error(f"Error: {e}")

# Tab 2: Library
with tab2:
    st.markdown("### 📚 My Paper Library")
    
    library_papers = get_library_papers()
    
    if library_papers:
        st.info(f"You have {len(library_papers)} papers in your library")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            search_library = st.text_input("🔍 Search in library", placeholder="Search titles, authors...")
        with col2:
            sort_lib = st.selectbox("Sort by", ["Year (newest)", "Year (oldest)", "Title", "Author"])
        
        # Apply filters
        filtered = library_papers.copy()
        if search_library:
            search_lower = search_library.lower()
            filtered = [p for p in filtered if 
                       search_lower in p.get("title", "").lower() or 
                       search_lower in p.get("authors", "").lower()]
        
        # Sort
        if "Year" in sort_lib:
            filtered.sort(key=lambda x: x.get("year") or 0, reverse=("newest" in sort_lib))
        elif sort_lib == "Title":
            filtered.sort(key=lambda x: x.get("title", ""))
        elif sort_lib == "Author":
            filtered.sort(key=lambda x: x.get("authors", ""))
        
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
                        st.info("Removal feature coming soon")
    else:
        st.info("Your library is empty. Search and add papers to get started!")

# Tab 3: RAG Chat
with tab3:
    st.markdown("### 💬 RAG Chat Interface")
    
    # Paper selection
    paper_selection_mode = st.radio(
        "Chat with:",
        ["All Library", "Selected Papers", "Fetch New Papers"],
        horizontal=True
    )
    
    selected_paper_ids = []
    if paper_selection_mode == "Selected Papers":
        library = get_library_papers()
        if library:
            paper_options = {f"{p['title'][:50]}...": p['paper_id'] for p in library}
            selected = st.multiselect("Choose papers:", list(paper_options.keys()))
            selected_paper_ids = [paper_options[s] for s in selected]
    
    # Chat interface
    st.divider()
    
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
        st.chat_message("user").write(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if paper_selection_mode == "Selected Papers" and selected_paper_ids:
                        result = api.rag_multi_document(selected_paper_ids, query)
                    else:
                        result = query_rag(
                            query=query,
                            top_k=top_k,
                            fetch_papers=(paper_selection_mode == "Fetch New Papers"),
                            use_enhanced=use_enhanced
                        )
                    
                    st.write(result["answer"])
                    
                    if result.get("citations"):
                        with st.expander("📚 Citations"):
                            unique_papers = {}
                            for citation in result["citations"]:
                                pid = citation["paper_id"]
                                if pid not in unique_papers:
                                    unique_papers[pid] = {"chunks": [], "scores": []}
                                unique_papers[pid]["chunks"].append(citation.get("chunk_index"))
                                unique_papers[pid]["scores"].append(citation.get("score", 0))
                            
                            for pid, info in unique_papers.items():
                                avg_score = sum(info["scores"]) / len(info["scores"]) if info["scores"] else 0
                                st.write(f"**Paper {pid}**: {len(info['chunks'])} chunks, Relevance: {avg_score:.2%}")
                    
                    st.session_state.chat_history.append({
                        "query": query,
                        "answer": result["answer"],
                        "citations": result.get("citations", []),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Tab 4: Compare Papers
with tab4:
    st.markdown("### 📊 Paper Comparison Mode")
    
    if st.session_state.comparison_papers:
        st.info(f"Comparing {len(st.session_state.comparison_papers)} papers")
        
        # Display comparison
        for i, paper in enumerate(st.session_state.comparison_papers, 1):
            st.markdown(f"### Paper {i}: {paper.get('title', 'Unknown')[:60]}...")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Authors:** {paper.get('authors_string', 'Unknown')}")
            with col2:
                st.write(f"**Year:** {paper.get('year', 'N/A')}")
            with col3:
                st.write(f"**Citations:** {paper.get('citation_count', 0)}")
        
        if st.button("🤖 Generate Comparison", type="primary"):
            with st.spinner("Generating comparison..."):
                try:
                    # Use RAG to compare papers
                    comparison_query = f"Compare these papers: {', '.join([p.get('title', '') for p in st.session_state.comparison_papers])}"
                    
                    paper_ids = [p.get("paper_id") for p in st.session_state.comparison_papers if p.get("paper_id")]
                    if paper_ids:
                        result = api.rag_compare(comparison_query)
                        st.write(result["answer"])
                    else:
                        st.warning("Papers need to be processed first. Processing...")
                        for paper in st.session_state.comparison_papers:
                            if paper.get("pdf_url"):
                                process_paper_for_rag(paper)
                except Exception as e:
                    st.error(f"Comparison error: {e}")
    else:
        st.info("Select papers from search results to compare them")

# Tab 5: Upload
with tab5:
    st.markdown("### 📤 Upload & Process Papers")
    
    upload_method = st.radio("Upload method:", ["PDF File", "PDF URL", "ArXiv ID", "DOI"], horizontal=True)
    
    if upload_method == "PDF URL":
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
    
    elif upload_method == "DOI":
        doi = st.text_input("Enter DOI:", placeholder="e.g., 10.1128/mbio.01735-25")
        if st.button("📥 Fetch & Process", type="primary"):
            if doi:
                with st.spinner(f"Fetching metadata for {doi}..."):
                    try:
                        # Try Crossref first
                        paper = get_crossref_by_doi(doi)
                        if paper and paper.get("pdf_url"):
                            result_id = ingest_pdf_from_url(
                                paper["pdf_url"],
                                paper_id=doi.replace("/", "_")
                            )
                            st.success(f"✅ Paper processed! ID: {result_id}")
                        else:
                            st.warning("No PDF found for this DOI")
                    except Exception as e:
                        st.error(f"Processing failed: {e}")

# Tab 6: Advanced Search
with tab6:
    st.markdown("### 🔎 Advanced Search Features")
    
    search_mode = st.radio("Search Mode:", ["Semantic Search", "Snippet Search", "Batch Lookup"], horizontal=True)
    
    if search_mode == "Semantic Search":
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

# Tab 7: Analytics
with tab7:
    st.markdown("### 📈 Analytics & Insights")
    
    analysis_type = st.selectbox("Choose analysis:", [
        "Paper Summaries",
        "Citation Rankings",
        "Author Statistics",
        "Topic Clustering"
    ])
    
    if analysis_type == "Paper Summaries":
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
                        if summary.get("bullets"):
                            st.write("**Key Points:**")
                            for bullet in summary["bullets"]:
                                st.write(f"• {bullet}")
                    except Exception as e:
                        st.error(f"Error: {e}")

# Footer
st.divider()
st.markdown("---")
st.markdown("**ScholarX** - Better Than Google Scholar | Built with Streamlit")

