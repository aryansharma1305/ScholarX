"""Complete RAG pipeline orchestration with enhanced features."""
from typing import Callable, Optional, Dict, List
from rag.retriever import retrieve_context
from rag.generator import generate_answer, RAGResponse
from rag.query_expander import expand_query_with_llm, normalize_query
from rag.hybrid_search import hybrid_search
from rag.reranker import rerank_results, ensure_diversity
from rag.quality_scorer import enhance_paper_metadata
from rag.citation_graph_retriever import apply_citation_boost
from ingestion.paper_fetcher import fetch_papers_by_topic
from ingestion.ingest_pipeline import ingest_pdf_from_url
from config.settings import settings
from utils.logger import get_logger
from utils.timers import timer

logger = get_logger(__name__)


def _log_stage_diagnostics(chunks: List, stage: str) -> None:
    """
    Emit per-stage score distribution logs for ablation study analysis.
    Produces concrete numbers suitable for inclusion in a paper:
      - count of citation-boosted vs unboosted chunks
      - average overlap fraction among boosted chunks
      - score delta: boosted avg vs unboosted avg
    """
    boosted = [c for c in chunks if c.metadata.get("citation_boost", 0) > 0]
    unboosted = [c for c in chunks if c.metadata.get("citation_boost", 0) == 0]
    avg = lambda lst: sum(c.score for c in lst) / len(lst) if lst else 0.0

    logger.info(
        "[Stage: %s] %d/%d chunks citation-boosted",
        stage, len(boosted), len(chunks),
    )
    if boosted:
        avg_overlap = sum(
            c.metadata.get("citation_overlap_fraction", 0) for c in boosted
        ) / len(boosted)
        hop_dist = {}
        for c in boosted:
            hop = c.metadata.get("citation_hop", "?") 
            hop_dist[hop] = hop_dist.get(hop, 0) + 1
        logger.info(
            "[Stage: %s] avg overlap fraction: %.3f | hop distribution: %s",
            stage, avg_overlap, hop_dist,
        )
    logger.info(
        "[Stage: %s] score delta (boosted=%.4f vs unboosted=%.4f, Δ=%.4f)",
        stage, avg(boosted), avg(unboosted), avg(boosted) - avg(unboosted),
    )


def run_rag_pipeline(
    query: str,
    top_k: int = 5,
    system_prompt: Optional[str] = None,
    fetch_papers: bool = True,
    use_hybrid_search: bool = True,
    use_reranking: bool = True,
    use_citation_boost: bool = True,
    debug_mode: bool = False,
    selected_paper_ids: Optional[List[str]] = None,
    thread_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    use_langgraph: Optional[bool] = None,
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
        use_citation_boost: Whether to apply citation-graph-aware score boosting
        debug_mode: If True, emit per-stage score distribution logs for ablation analysis
        selected_paper_ids: Restrict retrieval to specific papers
        thread_id: Persistent LangGraph conversation identifier
        progress_callback: Receives graph stage name and user-facing status
        use_langgraph: Override the USE_LANGGRAPH setting for this call
        
    Returns:
        RAGResponse with answer, citations, and context
    """
    effective_langgraph = (
        settings.use_langgraph if use_langgraph is None else use_langgraph
    )
    # Ablation runs require the original explicit stage implementation.
    if effective_langgraph and not debug_mode:
        from rag.langgraph_pipeline import run_research_graph

        result = run_research_graph(
            query=query,
            top_k=top_k,
            fetch_papers=fetch_papers,
            selected_paper_ids=selected_paper_ids,
            system_prompt=system_prompt,
            use_hybrid_search=use_hybrid_search,
            use_reranking=use_reranking,
            use_citation_boost=use_citation_boost,
            thread_id=thread_id,
            progress_callback=progress_callback,
        )
        return RAGResponse(
            query=result["query"],
            answer=result["answer"],
            citations=result["citations"],
            context_chunks=result["context_chunks"],
            metadata=result.get("workflow", {}),
        )

    logger.info(f"Running legacy enhanced RAG pipeline for query: {query[:50]}...")
    
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
                        pdf_url = paper.get("pdf_url")
                        if not pdf_url:
                            logger.warning(
                                "Skipping paper %s: no PDF URL available from source %s",
                                paper.get("paper_id"),
                                paper.get("source", "unknown")
                            )
                            continue

                        paper_id = ingest_pdf_from_url(
                            pdf_url=pdf_url,
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
    
    # Step 4b: Citation-graph-aware score boosting
    if use_citation_boost and settings.citation_boost_weight > 0:
        with timer("Citation Graph Boost"):
            context_chunks = apply_citation_boost(context_chunks)

    # Step 4c: Stage-level diagnostic logging (ablation study support)
    if debug_mode:
        _log_stage_diagnostics(context_chunks, stage="post-citation-boost")

    # Step 5: Take top_k after re-ranking + citation boost
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
