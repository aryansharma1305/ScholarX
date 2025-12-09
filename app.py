"""Streamlit app for ScholarX - Interactive RAG Pipeline."""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from main import query_rag, search_papers
from api.main_api import api
from manage_papers import get_statistics
from config.settings import settings

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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        margin-top: 1rem;
    }
    .paper-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .citation {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-left: 3px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stat-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def display_statistics():
    """Display collection statistics."""
    with st.spinner("Loading statistics..."):
        try:
            stats_output = []
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                get_statistics()
            stats_output = f.getvalue()
            
            # Parse and display stats
            st.markdown("### 📊 Collection Statistics")
            st.code(stats_output, language=None)
        except Exception as e:
            st.error(f"Error loading statistics: {e}")


def main():
    """Main Streamlit app."""
    st.markdown('<div class="main-header">📚 ScholarX - Research Paper RAG</div>', unsafe_allow_html=True)
    st.markdown("### Ask questions about research papers - papers are fetched automatically from APIs!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        max_papers = st.slider(
            "Papers to fetch per query",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of papers to fetch from APIs for each query"
        )
        settings.max_papers_per_query = max_papers
        
        top_k = st.slider(
            "Top K chunks",
            min_value=3,
            max_value=15,
            value=5,
            help="Number of context chunks to use for answer"
        )
        
        use_enhanced = st.checkbox(
            "Use enhanced features",
            value=True,
            help="Hybrid search, re-ranking, query expansion"
        )
        
        st.divider()
        st.markdown("### 📊 Collection Info")
        if st.button("Refresh Statistics"):
            display_statistics()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Ask Question", "📚 Search Papers", "📊 Collection", "ℹ️ About"])
    
    # Tab 1: Ask Question
    with tab1:
        st.markdown("### Ask a Question")
        st.markdown("Enter any research question. The system will automatically fetch relevant papers from APIs and answer your question.")
        
        query = st.text_input(
            "Your question:",
            placeholder="e.g., What is transformer architecture?",
            key="query_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
        
        if ask_button and query:
            with st.spinner("Fetching papers from APIs and generating answer..."):
                try:
                    # Query the RAG pipeline
                    result = query_rag(
                        query=query,
                        top_k=top_k,
                        fetch_papers=True,
                        use_enhanced=use_enhanced
                    )
                    
                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.markdown(result["answer"])
                    
                    # Display citations
                    st.markdown("### 📚 Citations")
                    unique_papers = {}
                    for citation in result["citations"]:
                        paper_id = citation["paper_id"]
                        if paper_id not in unique_papers:
                            unique_papers[paper_id] = {
                                "paper_id": paper_id,
                                "chunks": [],
                                "scores": []
                            }
                        unique_papers[paper_id]["chunks"].append(citation["chunk_index"])
                        unique_papers[paper_id]["scores"].append(citation["score"])
                    
                    for paper_id, info in unique_papers.items():
                        avg_score = sum(info["scores"]) / len(info["scores"])
                        st.markdown(f"""
                        <div class="citation">
                            <strong>Paper ID:</strong> {paper_id}<br>
                            <strong>Chunks cited:</strong> {len(info['chunks'])}<br>
                            <strong>Relevance:</strong> {avg_score:.2%}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display context chunks
                    with st.expander("📄 View Context Chunks"):
                        for i, chunk in enumerate(result["context_chunks"], 1):
                            st.markdown(f"**Chunk {i}** (Paper: {chunk['paper_id']}, Score: {chunk['score']:.2%})")
                            st.text(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
                            st.divider()
                    
                    # Success message
                    st.success(f"✅ Used {len(unique_papers)} papers to answer your question!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
    
    # Tab 2: Search Papers
    with tab2:
        st.markdown("### Search Papers")
        st.markdown("Search for papers by topic, author, or year. Papers are fetched from APIs.")
        
        search_type = st.radio(
            "Search by:",
            ["Topic", "Author", "Year"],
            horizontal=True
        )
        
        if search_type == "Topic":
            topic = st.text_input(
                "Enter topic:",
                placeholder="e.g., machine learning, transformers, attention",
                key="topic_search"
            )
            if st.button("🔍 Search", type="primary"):
                if topic:
                    with st.spinner(f"Searching for papers on '{topic}'..."):
                        try:
                            papers = search_papers(topic, max_papers=max_papers)
                            if papers:
                                st.success(f"Found {len(papers)} papers!")
                                for i, paper in enumerate(papers, 1):
                                    with st.container():
                                        st.markdown(f"""
                                        <div class="paper-card">
                                            <h4>{i}. {paper.get('title', 'Unknown')}</h4>
                                            <p><strong>Authors:</strong> {paper.get('authors_string', 'Unknown')}</p>
                                            <p><strong>Year:</strong> {paper.get('year', 'N/A')}</p>
                                            <p><strong>Source:</strong> {paper.get('source', 'unknown')}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.warning("No papers found. Try a different topic.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        elif search_type == "Author":
            author = st.text_input(
                "Enter author name:",
                placeholder="e.g., Geoffrey Hinton",
                key="author_search"
            )
            if st.button("🔍 Search", type="primary"):
                if author:
                    with st.spinner(f"Searching for papers by '{author}'..."):
                        try:
                            results = api.search(author=author, limit=10)
                            papers = results.get("papers", [])
                            if papers:
                                st.success(f"Found {len(papers)} papers!")
                                for i, paper in enumerate(papers, 1):
                                    st.markdown(f"""
                                    <div class="paper-card">
                                        <h4>{i}. {paper.get('title', 'Unknown')}</h4>
                                        <p><strong>Authors:</strong> {paper.get('authors', 'Unknown')}</p>
                                        <p><strong>Year:</strong> {paper.get('year', 'N/A')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning("No papers found by this author.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        elif search_type == "Year":
            year = st.number_input(
                "Enter year:",
                min_value=1900,
                max_value=2025,
                value=2024,
                key="year_search"
            )
            if st.button("🔍 Search", type="primary"):
                with st.spinner(f"Searching for papers from {year}..."):
                    try:
                        results = api.search(year=year, limit=10)
                        papers = results.get("papers", [])
                        if papers:
                            st.success(f"Found {len(papers)} papers!")
                            for i, paper in enumerate(papers, 1):
                                st.markdown(f"""
                                <div class="paper-card">
                                    <h4>{i}. {paper.get('title', 'Unknown')}</h4>
                                    <p><strong>Authors:</strong> {paper.get('authors', 'Unknown')}</p>
                                    <p><strong>Year:</strong> {paper.get('year', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning(f"No papers found from {year}.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Tab 3: Collection
    with tab3:
        st.markdown("### Collection Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Statistics")
            if st.button("🔄 Refresh Stats"):
                display_statistics()
            else:
                display_statistics()
        
        with col2:
            st.markdown("#### 🛠️ Actions")
            if st.button("📋 List All Papers"):
                try:
                    collection = api.get_citation_rankings()
                    papers = collection.get("ranked_papers", [])
                    if papers:
                        st.success(f"Found {len(papers)} papers in collection")
                        for paper in papers[:10]:  # Show top 10
                            st.text(f"• {paper.get('title', 'Unknown')[:60]}...")
                    else:
                        st.info("No papers in collection yet.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Tab 4: About
    with tab4:
        st.markdown("### About ScholarX")
        st.markdown("""
        **ScholarX** is a research paper RAG (Retrieval-Augmented Generation) system that:
        
        - 🔄 **Fetches papers on-demand** from APIs (ArXiv, Semantic Scholar)
        - 🔍 **Searches semantically** using embeddings
        - 💡 **Generates answers** with citations
        - 📚 **Stores papers** locally in ChromaDB
        
        ### Features:
        - ✅ On-demand paper fetching
        - ✅ Semantic search
        - ✅ Hybrid search (semantic + keyword)
        - ✅ Answer generation with citations
        - ✅ Author/year/topic search
        - ✅ Free APIs (no keys needed)
        
        ### APIs Used:
        - **ArXiv API**: Free, unlimited
        - **Semantic Scholar API**: Free, 100 req/5min
        
        ### How It Works:
        1. You ask a question
        2. System searches APIs for relevant papers
        3. Fetches and ingests papers automatically
        4. Generates answer using retrieved context
        5. Returns answer with citations
        """)


if __name__ == "__main__":
    main()

