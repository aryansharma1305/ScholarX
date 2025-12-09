"""Complete RAG pipeline orchestration with enhanced features."""
from typing import Optional, Dict, List
from rag.retriever import retrieve_context
from rag.generator import generate_answer, RAGResponse
from rag.query_expander import expand_query_with_llm, normalize_query
from rag.hybrid_search import hybrid_search
from rag.reranker import rerank_results, ensure_diversity
from rag.quality_scorer import enhance_paper_metadata
from ingestion.paper_fetcher import fetch_papers_by_topic
from ingestion.ingest_pipeline import ingest_pdf_from_url
from config.settings import settings
from utils.logger import get_logger
from utils.timers import timer

logger = get_logger(__name__)


def run_rag_pipeline(
    query: str,
    top_k: int = 5,
    system_prompt: Optional[str] = None,
    fetch_papers: bool = True,
    use_hybrid_search: bool = True,
    use_reranking: bool = True
) -> RAGResponse:
    """
    Run the complete enhanced RAG pipeline.
    
    Args:
        query: User query
        top_k: Number of context chunks to retrieve
        system_prompt: Optional custom system prompt
        fetch_papers: Whether to fetch papers on-demand if needed
        use_hybrid_search: Whether to use hybrid search
        use_reranking: Whether to re-rank results
        
    Returns:
        RAGResponse with answer, citations, and context
    """
    logger.info(f"Running enhanced RAG pipeline for query: {query[:50]}...")
    
    # Step 1: Normalize and expand query
    normalized_query = normalize_query(query)
    expanded_queries = expand_query_with_llm(normalized_query)
    logger.info(f"Query expanded to {len(expanded_queries)} variations")
    
    # Step 2: Always fetch papers from APIs when fetch_papers is enabled
    if fetch_papers:
        logger.info("Fetching papers from APIs based on query...")
        
        with timer("On-demand Paper Fetching"):
            # Fetch papers related to the query from APIs
            papers = fetch_papers_by_topic(normalized_query, max_papers=settings.max_papers_per_query)
            
            if papers:
                logger.info(f"Found {len(papers)} papers from APIs, ingesting...")
                # Ingest fetched papers
                paper_metadata_map = {}
                ingested_count = 0
                for paper in papers:
                    try:
                        paper_id = ingest_pdf_from_url(
                            pdf_url=paper["pdf_url"],
                            paper_id=paper["paper_id"],
                            metadata={
                                "title": paper.get("title", ""),
                                "authors": paper.get("authors_string", ""),
                                "abstract": paper.get("abstract", ""),
                                "year": paper.get("year"),
                                "source": paper.get("source", "api"),
                            }
                        )
                        paper_metadata_map[paper_id] = paper
                        ingested_count += 1
                        logger.info(f"Ingested paper: {paper.get('title', 'Unknown')[:50]}")
                    except Exception as e:
                        logger.warning(f"Failed to ingest paper {paper.get('paper_id')}: {e}")
                
                logger.info(f"Successfully ingested {ingested_count}/{len(papers)} papers")
            else:
                logger.warning("No papers found from APIs for this query")
    
    # Step 3: Retrieve context from ingested papers
    if use_hybrid_search:
        context_chunks = hybrid_search(normalized_query, top_k=top_k * 2)
    else:
        from rag.retriever import retrieve_context
        context_chunks = retrieve_context(normalized_query, top_k=top_k * 2)
    
    if not context_chunks:
        raise ValueError("No relevant context found for the query")
    
    # Step 4: Re-rank results if enabled
    if use_reranking:
        # Build paper metadata map for re-ranking
        paper_ids = list(set(chunk.paper_id for chunk in context_chunks))
        paper_metadata_map = {pid: {} for pid in paper_ids}  # Would need to fetch from storage
        
        context_chunks = rerank_results(context_chunks, paper_metadata_map)
        
        # Ensure diversity (max 2 chunks per paper)
        context_chunks = ensure_diversity(context_chunks, max_per_paper=2)
    
    # Step 5: Take top_k after re-ranking
    context_chunks = context_chunks[:top_k]
    
    # Step 6: Generate answer using retrieved context
    response = generate_answer(
        query=query,
        context_chunks=context_chunks,
        system_prompt=system_prompt
    )
    
    logger.info("Enhanced RAG pipeline completed successfully")
    return response


def run_simple_rag_pipeline(
    query: str,
    top_k: int = 5,
    system_prompt: Optional[str] = None
) -> RAGResponse:
    """
    Run simple RAG pipeline (original version, no enhancements).
    
    For backward compatibility.
    """
    logger.info(f"Running simple RAG pipeline for query: {query[:50]}...")
    
    # Retrieve relevant context
    context_chunks = retrieve_context(query, top_k=top_k)
    
    if not context_chunks:
        raise ValueError("No relevant context found for the query")
    
    # Generate answer using retrieved context
    response = generate_answer(
        query=query,
        context_chunks=context_chunks,
        system_prompt=system_prompt
    )
    
    logger.info("Simple RAG pipeline completed successfully")
    return response
