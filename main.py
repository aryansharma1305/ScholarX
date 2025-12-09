"""Main entry point for the RAG pipeline."""
import json
import uuid
from typing import Optional
from config.settings import settings
from ingestion.pdf_loader import load_pdf_from_url, extract_pdf_metadata
from ingestion.semantic_scholar import fetch_paper_metadata
from ingestion.paper_fetcher import fetch_papers_by_topic
from ingestion.ingest_pipeline import ingest_pdf_from_url
from rag.pipeline import run_rag_pipeline
from utils.logger import get_logger
from utils.timers import timer

logger = get_logger(__name__)


# ingest_pdf_from_url is now imported from ingestion.ingest_pipeline


def ingest_from_semantic_scholar(paper_id: str) -> str:
    """
    Ingest a paper from Semantic Scholar.
    
    Args:
        paper_id: Semantic Scholar paper ID
        
    Returns:
        Internal paper ID
    """
    logger.info(f"Ingesting paper from Semantic Scholar: {paper_id}")
    
    with timer("Semantic Scholar Ingestion"):
        # Fetch metadata
        metadata = fetch_paper_metadata(paper_id)
        
        if not metadata.get("pdf_url"):
            raise ValueError("No PDF URL available for this paper")
        
        # Use PDF URL ingestion
        internal_paper_id = ingest_pdf_from_url(
            pdf_url=metadata["pdf_url"],
            paper_id=paper_id,
            metadata={
                "title": metadata.get("title", ""),
                "authors": metadata.get("authors_string", ""),
                "abstract": metadata.get("abstract", ""),
                "year": metadata.get("year"),
                "source": "semantic_scholar",
            }
        )
    
    logger.info(f"Successfully ingested paper from Semantic Scholar")
    return internal_paper_id


def query_rag(
    query: str,
    top_k: int = 5,
    fetch_papers: bool = True,
    use_enhanced: bool = True
) -> dict:
    """
    Query the RAG pipeline with enhanced features.
    
    Args:
        query: User query
        top_k: Number of context chunks to retrieve
        fetch_papers: Whether to fetch papers on-demand if needed
        use_enhanced: Whether to use enhanced pipeline (hybrid search, re-ranking)
        
    Returns:
        RAG response as dictionary
    """
    logger.info(f"Processing RAG query: {query}")
    
    with timer("RAG Query"):
        if use_enhanced:
            response = run_rag_pipeline(
                query=query,
                top_k=top_k,
                fetch_papers=fetch_papers,
                use_hybrid_search=True,
                use_reranking=True
            )
        else:
            from rag.pipeline import run_simple_rag_pipeline
            response = run_simple_rag_pipeline(query, top_k=top_k)
        
        return {
            "query": response.query,
            "answer": response.answer,
            "citations": response.citations,
            "context_chunks": response.context_chunks,
        }


def search_papers(topic: str, max_papers: int = 5) -> list:
    """
    Search and fetch papers by topic (without ingesting).
    
    Args:
        topic: Search topic/query
        max_papers: Maximum number of papers to fetch
        
    Returns:
        List of paper dictionaries
    """
    logger.info(f"Searching papers for topic: {topic}")
    papers = fetch_papers_by_topic(topic, max_papers=max_papers)
    return papers


def main():
    """Example usage of the RAG pipeline."""
    # Validate settings
    try:
        settings.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return
    
    print("=" * 60)
    print("ScholarX Enhanced RAG Pipeline")
    print("=" * 60)
    print()
    print("Features:")
    print("  ✅ ChromaDB vector storage")
    print("  ✅ Multi-source paper fetching (Semantic Scholar, ArXiv)")
    print("  ✅ Query expansion")
    print("  ✅ Hybrid search (semantic + keyword)")
    print("  ✅ Quality scoring & re-ranking")
    print("  ✅ On-demand paper ingestion")
    print()
    print("Example usage:")
    print("  from main import query_rag, search_papers")
    print("  result = query_rag('What is transformer architecture?')")
    print("  papers = search_papers('machine learning', max_papers=5)")


if __name__ == "__main__":
    main()
