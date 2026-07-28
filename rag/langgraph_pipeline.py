"""Stateful LangGraph orchestration for ScholarX RAG."""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from api.literature_review_search import add_quality_labels, deduplicate_papers
from config.chroma_client import get_collection
from config.settings import settings
from ingestion.ingest_pipeline import ingest_pdf_from_url
from ingestion.paper_fetcher import fetch_papers_by_topic
from processing.embeddings import generate_embedding
from rag.citation_graph_retriever import apply_citation_boost
from rag.generator import generate_answer
from rag.hybrid_search import hybrid_search
from rag.langchain_models import (
    generate_grounded_answer,
    grade_evidence,
    model_calls_disabled,
    rewrite_query_for_retry,
    validate_answer,
)
from rag.query_expander import expand_query_with_llm, normalize_query
from rag.reranker import ensure_diversity, rerank_results
from utils.logger import get_logger
from vectorstore.query import QueryResult, query_vectors

logger = get_logger(__name__)

ProgressCallback = Callable[[str, str], None]


class ResearchState(TypedDict, total=False):
    """Serializable state persisted for each research conversation."""

    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    active_query: str
    normalized_query: str
    expanded_queries: List[str]
    selected_paper_ids: List[str]
    discovered_papers: List[Dict[str, Any]]
    ingested_paper_ids: List[str]
    context_chunks: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    answer: str
    system_prompt: Optional[str]
    top_k: int
    fetch_papers: bool
    use_hybrid_search: bool
    use_reranking: bool
    use_citation_boost: bool
    retrieval_attempts: int
    external_searches: int
    answer_retries: int
    evidence_sufficient: bool
    evidence_grade: Dict[str, Any]
    answer_supported: bool
    answer_grade: Dict[str, Any]
    warnings: List[str]


def _result_to_dict(result: QueryResult) -> Dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "paper_id": result.paper_id,
        "chunk_index": result.chunk_index,
        "text": result.text,
        "score": float(result.score),
        "metadata": result.metadata or {},
    }


def _dict_to_result(chunk: Dict[str, Any]) -> QueryResult:
    return QueryResult(
        chunk_id=str(chunk.get("chunk_id") or ""),
        paper_id=str(chunk.get("paper_id") or ""),
        chunk_index=int(chunk.get("chunk_index", -1)),
        text=str(chunk.get("text") or ""),
        score=float(chunk.get("score", 0.0)),
        metadata=dict(chunk.get("metadata") or {}),
    )


def normalize_node(state: ResearchState) -> Dict[str, Any]:
    normalized = normalize_query(state["query"])
    expanded = expand_query_with_llm(normalized)
    unique_expansions = list(dict.fromkeys([normalized, *expanded]))
    return {
        "active_query": normalized,
        "normalized_query": normalized,
        "expanded_queries": unique_expansions[:5],
    }


def _retrieve_selected_papers(
    query: str,
    paper_ids: List[str],
    top_k: int,
) -> List[QueryResult]:
    embedding = generate_embedding(query)
    results = []
    per_paper = max(2, top_k)
    for paper_id in paper_ids:
        results.extend(
            query_vectors(
                query_embedding=embedding,
                top_k=per_paper,
                filter_metadata={"paper_id": paper_id},
            )
        )
    return results


def retrieve_node(state: ResearchState) -> Dict[str, Any]:
    attempt = state.get("retrieval_attempts", 0) + 1
    top_k = max(int(state.get("top_k", settings.default_top_k)), 1)
    active_query = state.get("active_query") or state["normalized_query"]
    queries = [active_query]
    if attempt > 1:
        queries.extend(state.get("expanded_queries", [])[:3])
    queries = list(dict.fromkeys(query for query in queries if query))

    combined: Dict[str, QueryResult] = {}
    warnings = list(state.get("warnings", []))
    for query in queries:
        try:
            if state.get("selected_paper_ids"):
                results = _retrieve_selected_papers(
                    query,
                    state["selected_paper_ids"],
                    top_k * 2,
                )
            elif state.get("use_hybrid_search", True):
                results = hybrid_search(query, top_k=top_k * 2)
            else:
                embedding = generate_embedding(query)
                results = query_vectors(embedding, top_k=top_k * 2)

            for result in results:
                key = result.chunk_id or f"{result.paper_id}:{result.chunk_index}"
                previous = combined.get(key)
                if previous is None or result.score > previous.score:
                    combined[key] = result
        except Exception as exc:
            message = f"Retrieval failed for '{query}': {exc}"
            logger.warning(message)
            warnings.append(message)

    ranked = sorted(combined.values(), key=lambda result: result.score, reverse=True)
    return {
        "context_chunks": [_result_to_dict(result) for result in ranked[: top_k * 3]],
        "retrieval_attempts": attempt,
        "warnings": warnings,
    }


def grade_evidence_node(state: ResearchState) -> Dict[str, Any]:
    grade = grade_evidence(
        query=state["query"],
        context_chunks=state.get("context_chunks", []),
    )
    return {
        "evidence_sufficient": bool(grade.get("sufficient")),
        "evidence_grade": grade,
    }


def route_after_evidence(state: ResearchState) -> str:
    if state.get("evidence_sufficient"):
        return "rerank"

    has_context = bool(state.get("context_chunks"))
    external_search_allowed = (
        state.get("fetch_papers", True)
        and not state.get("selected_paper_ids")
        and state.get("external_searches", 0)
        < settings.langgraph_max_external_searches
    )
    if external_search_allowed:
        return "external_search"
    return "rerank" if has_context else "generate"


def external_search_node(state: ResearchState) -> Dict[str, Any]:
    search_count = state.get("external_searches", 0) + 1
    warnings = list(state.get("warnings", []))
    try:
        papers = fetch_papers_by_topic(
            topic=state.get("active_query") or state["normalized_query"],
            max_papers=max(settings.max_papers_per_query, settings.langgraph_ingest_limit),
        )
        papers = deduplicate_papers(papers)
        papers = add_quality_labels(state["query"], papers)
        papers.sort(
            key=lambda paper: (
                bool(paper.get("pdf_url")),
                float(paper.get("review_score", 0.0)),
                int(paper.get("citation_count") or 0),
            ),
            reverse=True,
        )
    except Exception as exc:
        message = f"External paper search failed: {exc}"
        logger.warning(message)
        warnings.append(message)
        papers = []

    return {
        "discovered_papers": papers,
        "external_searches": search_count,
        "warnings": warnings,
    }


def _paper_is_ingested(paper_id: str) -> bool:
    if not paper_id:
        return False
    try:
        existing = get_collection().get(
            where={"paper_id": str(paper_id)},
            limit=1,
        )
        return bool(existing.get("ids"))
    except Exception:
        return False


def ingest_papers_node(state: ResearchState) -> Dict[str, Any]:
    warnings = list(state.get("warnings", []))
    ingested_ids = []
    candidates = [
        paper
        for paper in state.get("discovered_papers", [])
        if paper.get("paper_id") and paper.get("pdf_url")
    ][: settings.langgraph_ingest_limit]

    for paper in candidates:
        paper_id = str(paper["paper_id"])
        if _paper_is_ingested(paper_id):
            ingested_ids.append(paper_id)
            continue

        try:
            saved_id = ingest_pdf_from_url(
                pdf_url=paper["pdf_url"],
                paper_id=paper_id,
                metadata={
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors_string", ""),
                    "abstract": paper.get("abstract", ""),
                    "year": paper.get("year"),
                    "source": paper.get("source", "api"),
                    "doi": paper.get("doi", ""),
                    "citation_count": int(paper.get("citation_count") or 0),
                },
            )
            ingested_ids.append(saved_id)
        except Exception as exc:
            message = f"Could not ingest {paper.get('title', paper_id)}: {exc}"
            logger.warning(message)
            warnings.append(message)

    if not candidates:
        warnings.append("External search returned no ingestible PDF papers.")

    return {
        "ingested_paper_ids": ingested_ids,
        "warnings": warnings,
    }


def rerank_node(state: ResearchState) -> Dict[str, Any]:
    results = [_dict_to_result(chunk) for chunk in state.get("context_chunks", [])]
    if not results:
        return {"context_chunks": []}

    if state.get("use_reranking", True):
        metadata_map = {
            result.paper_id: dict(result.metadata or {})
            for result in results
        }
        results = rerank_results(results, metadata_map)
        results = ensure_diversity(results, max_per_paper=2)

    warnings = list(state.get("warnings", []))
    if state.get("use_citation_boost", True) and settings.citation_boost_weight > 0:
        try:
            results = apply_citation_boost(results)
        except Exception as exc:
            message = f"Citation boost skipped: {exc}"
            logger.warning(message)
            warnings.append(message)

    results.sort(key=lambda result: result.score, reverse=True)
    top_k = max(int(state.get("top_k", settings.default_top_k)), 1)
    return {
        "context_chunks": [_result_to_dict(result) for result in results[:top_k]],
        "warnings": warnings,
    }


def generate_node(state: ResearchState) -> Dict[str, Any]:
    context_chunks = state.get("context_chunks", [])
    if not context_chunks:
        answer = (
            "I could not find enough relevant evidence in the selected library or "
            "available paper sources to answer this question reliably."
        )
        return {
            "answer": answer,
            "citations": [],
            "messages": [AIMessage(content=answer)],
        }

    answer = generate_grounded_answer(
        query=state["query"],
        context_chunks=context_chunks,
        system_prompt=state.get("system_prompt"),
    )
    if answer is None:
        response = generate_answer(
            query=state["query"],
            context_chunks=[_dict_to_result(chunk) for chunk in context_chunks],
            system_prompt=state.get("system_prompt"),
            force_simple=model_calls_disabled(),
        )
        answer = response.answer
        citations = response.citations
    else:
        citations = [
            {
                "paper_id": chunk.get("paper_id"),
                "chunk_index": chunk.get("chunk_index"),
                "score": chunk.get("score", 0.0),
            }
            for chunk in context_chunks
        ]

    return {
        "answer": answer,
        "citations": citations,
        "messages": [AIMessage(content=answer)],
    }


def validate_answer_node(state: ResearchState) -> Dict[str, Any]:
    grade = validate_answer(
        query=state["query"],
        answer=state.get("answer", ""),
        citations=state.get("citations", []),
        context_chunks=state.get("context_chunks", []),
    )
    return {
        "answer_supported": bool(grade.get("supported")),
        "answer_grade": grade,
    }


def route_after_validation(state: ResearchState) -> str:
    if state.get("answer_supported"):
        return "finalize"
    if state.get("answer_retries", 0) >= settings.langgraph_max_answer_retries:
        return "finalize"
    return "prepare_retry"


def prepare_retry_node(state: ResearchState) -> Dict[str, Any]:
    reason = str(state.get("answer_grade", {}).get("reason") or "weak evidence")
    rewritten = rewrite_query_for_retry(state["query"], reason)
    expansions = list(dict.fromkeys([rewritten, *state.get("expanded_queries", [])]))
    return {
        "active_query": rewritten,
        "expanded_queries": expansions[:5],
        "answer_retries": state.get("answer_retries", 0) + 1,
        "context_chunks": [],
        "citations": [],
        "answer": "",
        "evidence_sufficient": False,
    }


def finalize_node(state: ResearchState) -> Dict[str, Any]:
    if state.get("answer_supported"):
        return {}

    answer = state.get("answer", "")
    warning = (
        "\n\nEvidence note: the automated validation could not fully confirm "
        "support for every part of this answer. Review the cited paper chunks "
        "before using it in a literature review."
    )
    if warning.strip() not in answer:
        answer = f"{answer.rstrip()}{warning}"
    return {"answer": answer}


def build_research_graph(checkpointer=None):
    """Build the graph independently so nodes and routing remain testable."""
    builder = StateGraph(ResearchState)
    builder.add_node("normalize", normalize_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_evidence", grade_evidence_node)
    builder.add_node("external_search", external_search_node)
    builder.add_node("ingest_papers", ingest_papers_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("generate", generate_node)
    builder.add_node("validate_answer", validate_answer_node)
    builder.add_node("prepare_retry", prepare_retry_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "retrieve")
    builder.add_edge("retrieve", "grade_evidence")
    builder.add_conditional_edges(
        "grade_evidence",
        route_after_evidence,
        {
            "rerank": "rerank",
            "external_search": "external_search",
            "generate": "generate",
        },
    )
    builder.add_edge("external_search", "ingest_papers")
    builder.add_edge("ingest_papers", "retrieve")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", "validate_answer")
    builder.add_conditional_edges(
        "validate_answer",
        route_after_validation,
        {
            "finalize": "finalize",
            "prepare_retry": "prepare_retry",
        },
    )
    builder.add_edge("prepare_retry", "retrieve")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


STAGE_MESSAGES = {
    "normalize": "Normalized and expanded the research question",
    "retrieve": "Searched the local paper library",
    "grade_evidence": "Checked whether the retrieved evidence is sufficient",
    "external_search": "Searched external scholarly sources",
    "ingest_papers": "Added the strongest available PDF papers to the library",
    "rerank": "Reranked evidence and applied citation signals",
    "generate": "Generated a grounded answer",
    "validate_answer": "Validated the answer against its evidence",
    "prepare_retry": "Prepared one controlled retrieval retry",
    "finalize": "Finalized the answer and citations",
}


def _initial_state(
    query: str,
    *,
    top_k: int,
    fetch_papers: bool,
    selected_paper_ids: Optional[List[str]],
    system_prompt: Optional[str],
    use_hybrid_search: bool,
    use_reranking: bool,
    use_citation_boost: bool,
) -> ResearchState:
    return {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "active_query": query,
        "normalized_query": "",
        "expanded_queries": [],
        "selected_paper_ids": selected_paper_ids or [],
        "discovered_papers": [],
        "ingested_paper_ids": [],
        "context_chunks": [],
        "citations": [],
        "answer": "",
        "system_prompt": system_prompt,
        "top_k": top_k,
        "fetch_papers": fetch_papers,
        "use_hybrid_search": use_hybrid_search,
        "use_reranking": use_reranking,
        "use_citation_boost": use_citation_boost,
        "retrieval_attempts": 0,
        "external_searches": 0,
        "answer_retries": 0,
        "evidence_sufficient": False,
        "evidence_grade": {},
        "answer_supported": False,
        "answer_grade": {},
        "warnings": [],
    }


def run_research_graph(
    query: str,
    *,
    top_k: int = 5,
    fetch_papers: bool = True,
    selected_paper_ids: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    use_hybrid_search: bool = True,
    use_reranking: bool = True,
    use_citation_boost: bool = True,
    thread_id: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Run the persistent graph and return the existing ScholarX response shape."""
    checkpoint_path = Path(settings.langgraph_checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    graph = build_research_graph(checkpointer=checkpointer)

    graph_thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": graph_thread_id}}
    initial_state = _initial_state(
        query,
        top_k=top_k,
        fetch_papers=fetch_papers,
        selected_paper_ids=selected_paper_ids,
        system_prompt=system_prompt,
        use_hybrid_search=use_hybrid_search,
        use_reranking=use_reranking,
        use_citation_boost=use_citation_boost,
    )
    stages = []

    try:
        if progress_callback:
            for update in graph.stream(initial_state, config=config, stream_mode="updates"):
                for stage in update:
                    if stage not in STAGE_MESSAGES:
                        continue
                    stages.append(stage)
                    progress_callback(stage, STAGE_MESSAGES[stage])
            final_state = dict(graph.get_state(config).values)
        else:
            final_state = dict(graph.invoke(initial_state, config=config))
            stages = ["completed"]
    finally:
        connection.close()

    return {
        "query": query,
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []),
        "context_chunks": final_state.get("context_chunks", []),
        "workflow": {
            "engine": "langgraph",
            "thread_id": graph_thread_id,
            "stages": stages,
            "retrieval_attempts": final_state.get("retrieval_attempts", 0),
            "external_searches": final_state.get("external_searches", 0),
            "answer_retries": final_state.get("answer_retries", 0),
            "evidence_grade": final_state.get("evidence_grade", {}),
            "answer_grade": final_state.get("answer_grade", {}),
            "ingested_paper_ids": final_state.get("ingested_paper_ids", []),
            "warnings": final_state.get("warnings", []),
        },
    }
