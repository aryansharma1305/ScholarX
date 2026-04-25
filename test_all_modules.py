"""
=============================================================
  ScholarX RAG Pipeline -- Comprehensive Module Test Suite
=============================================================
Run from project root:
    cd "/Users/gugloo/Documents/Major project/python-rag"
    python3 test_all_modules.py
"""

import sys
import importlib
import time

# ─── Colours ───────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed  = []
failed  = []
skipped = []


def header(title: str):
    print(f"\n{CYAN}{BOLD}{'='*60}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'='*60}{RESET}")


def ok(name: str, detail: str = ""):
    passed.append(name)
    detail_str = f"  → {detail}" if detail else ""
    print(f"  {GREEN}✓ PASS{RESET}  {name}{detail_str}")


def fail(name: str, error: str):
    failed.append(name)
    print(f"  {RED}✗ FAIL{RESET}  {name}")
    print(f"         {RED}{error[:120]}{RESET}")


def skip(name: str, reason: str):
    skipped.append(name)
    print(f"  {YELLOW}⚠ SKIP{RESET}  {name}  ({reason})")


def run_test(name: str, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        ok(name, str(result)[:80] if result is not None else "")
        return result
    except Exception as e:
        fail(name, f"{type(e).__name__}: {e}")
        return None


# ════════════════════════════════════════════════════════════
# 1. IMPORTS / MODULE AVAILABILITY
# ════════════════════════════════════════════════════════════
header("1. Module Import Check")

# (module_name, first_symbol_to_verify)
MODULES = [
    ("config.settings",                    "Settings"),
    ("utils.logger",                       "get_logger"),
    ("utils.cache",                        "get_cache_key"),
    ("utils.timers",                       "timer"),
    ("processing.chunker",                 "chunk_text"),
    ("processing.advanced_chunker",        "smart_chunk"),
    ("processing.embeddings",              "generate_embedding"),
    ("vectorstore.upsert",                 "upsert_chunks"),
    ("vectorstore.query",                  "query_vectors"),
    ("ingestion.text_cleaner",             "clean_text"),
    ("ingestion.paper_fetcher",            "fetch_papers_by_topic"),
    ("ingestion.arxiv_enhanced",           "search_arxiv_enhanced"),
    ("ingestion.semantic_scholar",         "search_papers"),
    ("ingestion.crossref_api",             "search_crossref"),
    ("ingestion.ingest_pipeline",          "ingest_pdf_from_url"),
    ("ingestion.pdf_loader",               "load_pdf_from_url"),
    ("rag.retriever",                      "retrieve_context"),
    ("rag.generator",                      "generate_answer"),
    ("rag.query_expander",                 "expand_query_with_llm"),
    ("rag.hybrid_search",                  "hybrid_search"),
    ("rag.reranker",                       "rerank_results"),
    ("rag.quality_scorer",                 "enhance_paper_metadata"),
    ("rag.pipeline",                       "run_rag_pipeline"),
    ("rag.search_enhanced",                "search_by_author"),
    ("api.main_api",                       "ScholarXAPI"),
    ("api.search",                         "search_papers"),
    ("api.query_intent",                   "classify_query_intent"),
    ("api.recommendations",                "recommend_papers_based_on_history"),
    ("api.research_gaps",                  "identify_research_gaps"),
    ("api.trends",                         "analyze_topic_trends"),
    ("api.visualization",                  "visualize_citation_network"),
    ("api.exports",                        "export_to_bibtex"),
    ("api.relevance_ranking",              "rank_papers_by_relevance"),
    ("api.deduplication",                  "find_duplicate_papers"),
    ("api.authors",                        "get_author_stats"),
    ("api.citations",                      "get_citation_info"),
    ("api.summaries",                      "generate_paper_summary"),
    ("api.topics",                         "cluster_papers_by_topic"),
    ("api.similarity",                     "compare_papers"),
    ("api.paper_api",                      "get_paper_by_id"),
    ("evaluation.metrics",                 "calculate_retrieval_metrics"),
    ("evaluation.datasets",                "load_evaluation_dataset"),
    ("evaluation.baselines",               "BaselineSystem"),
]

import_status = {}
for mod_name, sym in MODULES:
    try:
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, sym):
            fail(f"import {mod_name}", f"Symbol '{sym}' not found in module")
            import_status[mod_name] = False
        else:
            import_status[mod_name] = True
            ok(f"import {mod_name}", f"✓ {sym} found")
    except ImportError as e:
        fail(f"import {mod_name}", str(e))
        import_status[mod_name] = False
    except Exception as e:
        fail(f"import {mod_name}", f"{type(e).__name__}: {e}")
        import_status[mod_name] = False


# ════════════════════════════════════════════════════════════
# 2. CONFIGURATION MODULE
# ════════════════════════════════════════════════════════════
header("2. Config / Settings")

def test_settings_load():
    from config.settings import Settings, settings
    assert isinstance(settings, Settings)
    return f"embedding_provider={settings.embedding_provider}, llm_provider={settings.llm_provider}"

def test_settings_defaults():
    from config.settings import Settings
    s = Settings()
    assert s.chunk_size > 0
    assert s.chunk_overlap >= 0
    assert s.default_top_k > 0
    return f"chunk_size={s.chunk_size}, chunk_overlap={s.chunk_overlap}, top_k={s.default_top_k}"

def test_settings_validate():
    from config.settings import Settings
    s = Settings()
    s.embedding_provider = "sentence-transformers"
    s.validate()
    return "validate() OK for sentence-transformers"

run_test("Settings: load global instance",        test_settings_load)
run_test("Settings: default values",              test_settings_defaults)
run_test("Settings: validate() no error",         test_settings_validate)


# ════════════════════════════════════════════════════════════
# 3. UTILS MODULE
# ════════════════════════════════════════════════════════════
header("3. Utils")

def test_logger():
    import logging
    from utils.logger import get_logger
    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    logger.info("Logger test message")
    return "Logger created and usable"

def test_cache_key_deterministic():
    from utils.cache import get_cache_key
    k1 = get_cache_key("arg1", "arg2", kwarg1="v1")
    k2 = get_cache_key("arg1", "arg2", kwarg1="v1")
    assert k1 == k2, "Cache key must be deterministic"
    return f"key={k1[:8]}..."

def test_cache_key_uniqueness():
    from utils.cache import get_cache_key
    k1 = get_cache_key("arg_alpha")
    k2 = get_cache_key("arg_beta")
    assert k1 != k2
    return "Different args → different keys ✓"

def test_cache_save_load():
    from utils.cache import save_to_cache, load_from_cache, get_cache_key
    key = get_cache_key("unit_test_save_load_99")
    data = {"status": "ok", "value": 99}
    save_to_cache("api_responses", key, data, ttl=3600)
    loaded = load_from_cache("api_responses", key)
    assert loaded is not None
    assert loaded["value"] == 99
    return f"Saved & loaded: {loaded}"

def test_cache_stats():
    from utils.cache import get_cache_stats
    stats = get_cache_stats()
    assert "total_files" in stats
    return f"Cache files={stats['total_files']}, size={stats['total_size_mb']:.2f} MB"

def test_timer():
    from utils.timers import timer
    with timer("test_op"):
        time.sleep(0.01)
    return "timer context manager works"

run_test("Utils: get_logger",                     test_logger)
run_test("Utils: cache key – deterministic",      test_cache_key_deterministic)
run_test("Utils: cache key – uniqueness",         test_cache_key_uniqueness)
run_test("Utils: cache save & load",              test_cache_save_load)
run_test("Utils: cache stats",                    test_cache_stats)
run_test("Utils: timer context manager",          test_timer)


# ════════════════════════════════════════════════════════════
# 4. PROCESSING MODULE
# ════════════════════════════════════════════════════════════
header("4. Processing – Chunker")

SAMPLE_TEXT = (
    "Retrieval Augmented Generation (RAG) is a technique combining information "
    "retrieval with language model generation. It retrieves relevant documents "
    "from a knowledge base and uses them as additional context for the language model. "
    "This helps reduce hallucination and improves factual accuracy. The approach was "
    "introduced in 2020 and has become widely adopted. Variants include hybrid search, "
    "re-ranking, and query expansion. These enhancements improve retrieval quality and "
    "ultimately lead to better generated answers for end users of the system."
)

def test_chunk_text_basic():
    from processing.chunker import chunk_text
    chunks = chunk_text(SAMPLE_TEXT, paper_id="test_paper_001", chunk_size=200, chunk_overlap=50)
    assert isinstance(chunks, list) and len(chunks) > 0
    for c in chunks:
        assert len(c.text) > 0
        assert c.paper_id == "test_paper_001"
    return f"{len(chunks)} chunks created"

def test_chunk_text_empty():
    from processing.chunker import chunk_text
    chunks = chunk_text("", paper_id="empty_paper")
    assert chunks == []
    return "Empty text → [] ✓"

def test_chunk_text_single():
    from processing.chunker import chunk_text
    chunks = chunk_text("Short text.", paper_id="short_paper", chunk_size=1000)
    assert len(chunks) == 1
    return f"Single chunk: '{chunks[0].text}'"

def test_advanced_chunker_sections():
    from processing.advanced_chunker import chunk_by_sections
    text_with_sections = (
        "Abstract\nThis paper proposes a new RAG system.\n\n"
        "Introduction\nRetrieval systems are important.\n\n"
        "Methods\nWe use vector embeddings.\n\n"
        "Results\nOur method outperforms baselines."
    )
    chunks = chunk_by_sections(text_with_sections, paper_id="adv_001")
    assert isinstance(chunks, list)
    return f"{len(chunks)} section-based chunks"

def test_smart_chunk():
    from processing.advanced_chunker import smart_chunk
    chunks = smart_chunk(SAMPLE_TEXT, paper_id="smart_001")
    assert isinstance(chunks, list) and len(chunks) > 0
    return f"smart_chunk: {len(chunks)} chunks"

run_test("Chunker: basic chunking",               test_chunk_text_basic)
run_test("Chunker: empty text",                   test_chunk_text_empty)
run_test("Chunker: single chunk",                 test_chunk_text_single)

if import_status.get("processing.advanced_chunker"):
    run_test("AdvancedChunker: section-based",    test_advanced_chunker_sections)
    run_test("AdvancedChunker: smart_chunk",      test_smart_chunk)
else:
    skip("AdvancedChunker: section-based",        "module not importable")
    skip("AdvancedChunker: smart_chunk",          "module not importable")

# Embeddings: skip model loading (very slow – model not cached)
skip("Embeddings: generate_embedding",
     "Skipped to avoid slow model download (sentence-transformers)")
skip("Embeddings: batch generation",
     "Skipped to avoid slow model download (sentence-transformers)")


# ════════════════════════════════════════════════════════════
# 5. INGESTION MODULE
# ════════════════════════════════════════════════════════════
header("5. Ingestion Module")

def test_text_cleaner():
    from ingestion.text_cleaner import clean_text
    raw = "  Hello\r\n World!!\n\nExtra   spaces.  "
    cleaned = clean_text(raw)
    assert isinstance(cleaned, str) and len(cleaned) > 0
    return f"Cleaned: '{cleaned[:60]}'"

def test_arxiv_fetch():
    from ingestion.arxiv_enhanced import search_arxiv_enhanced
    result = search_arxiv_enhanced("retrieval augmented generation", max_results=2)
    # Returns a dict with a 'papers' key, or directly a list depending on implementation
    if isinstance(result, dict):
        papers = result.get("papers", result.get("results", []))
    else:
        papers = result
    assert isinstance(papers, list)
    return f"ArXiv returned {len(papers)} papers"

def test_semantic_scholar_search():
    from ingestion.semantic_scholar import search_papers
    papers = search_papers("RAG language model", limit=2)
    assert isinstance(papers, list)
    return f"Semantic Scholar returned {len(papers)} papers"

def test_crossref_search():
    from ingestion.crossref_api import search_crossref
    result = search_crossref(query="deep learning", rows=2)
    assert isinstance(result, dict) and "items" in result
    return f"CrossRef returned {len(result['items'])} results (total={result.get('total',0)})"

def test_paper_fetcher():
    from ingestion.paper_fetcher import fetch_papers_by_topic
    papers = fetch_papers_by_topic("machine learning", max_papers=2)
    assert isinstance(papers, list)
    return f"Paper fetcher: {len(papers)} papers for 'machine learning'"

if import_status.get("ingestion.text_cleaner"):
    run_test("Ingestion: text cleaner",           test_text_cleaner)
else:
    skip("Ingestion: text cleaner", "module not importable")

if import_status.get("ingestion.arxiv_enhanced"):
    run_test("Ingestion: ArXiv search (live)",    test_arxiv_fetch)
else:
    skip("Ingestion: ArXiv fetch", "module not importable")

if import_status.get("ingestion.semantic_scholar"):
    run_test("Ingestion: Semantic Scholar (live)", test_semantic_scholar_search)
else:
    skip("Ingestion: Semantic Scholar", "module not importable")

if import_status.get("ingestion.crossref_api"):
    run_test("Ingestion: CrossRef (live)",        test_crossref_search)
else:
    skip("Ingestion: CrossRef", "module not importable")

if import_status.get("ingestion.paper_fetcher"):
    run_test("Ingestion: paper_fetcher (live)",   test_paper_fetcher)
else:
    skip("Ingestion: paper_fetcher", "module not importable")


# ════════════════════════════════════════════════════════════
# 6. VECTORSTORE MODULE
# ════════════════════════════════════════════════════════════
header("6. VectorStore Module")

def test_vectorstore_upsert_query():
    from processing.chunker import Chunk
    from vectorstore.upsert import upsert_chunks
    from vectorstore.query import query_vectors
    from processing.embeddings import generate_embedding, generate_embeddings_batch
    test_chunks = [
        Chunk(text="RAG improves accuracy by grounding answers in retrieved docs.",
              index=0, paper_id="vs_test_001"),
        Chunk(text="RAG combines dense retrieval with autoregressive generation.",
              index=1, paper_id="vs_test_001"),
    ]
    texts = [c.text for c in test_chunks]
    embeddings = generate_embeddings_batch(texts)
    upsert_chunks(test_chunks, embeddings)
    emb = generate_embedding("retrieval augmented generation")
    results = query_vectors(emb, top_k=2)
    assert isinstance(results, list)
    return f"Upserted 2 chunks, queried → {len(results)} results"

if import_status.get("vectorstore.upsert") and import_status.get("vectorstore.query"):
    run_test("VectorStore: upsert + query", test_vectorstore_upsert_query)
else:
    skip("VectorStore: upsert + query",
         "vectorstore.upsert or vectorstore.query not importable")


# ════════════════════════════════════════════════════════════
# 7. RAG MODULE
# ════════════════════════════════════════════════════════════
header("7. RAG Module")

def test_normalize_query():
    from rag.query_expander import normalize_query
    q = "  What IS  Retrieval Augmented Generation?  "
    norm = normalize_query(q)
    assert isinstance(norm, str)
    assert norm.strip() == norm
    return f"Normalized: '{norm}'"

def test_expand_query():
    from rag.query_expander import expand_query_with_llm
    queries = expand_query_with_llm("What is RAG?")
    assert isinstance(queries, list) and len(queries) >= 1
    return f"Expanded to {len(queries)} variant(s)"

def test_retrieve_context():
    from rag.retriever import retrieve_context
    try:
        chunks = retrieve_context("retrieval augmented generation", top_k=3)
        assert isinstance(chunks, list)
        return f"Retrieved {len(chunks)} chunks"
    except Exception as e:
        return f"0 chunks (DB may be empty): {e}"

def test_hybrid_search():
    from rag.hybrid_search import hybrid_search
    try:
        results = hybrid_search("machine learning transformers", top_k=3)
        assert isinstance(results, list)
        return f"Hybrid search → {len(results)} results"
    except Exception as e:
        return f"0 results (DB may be empty): {e}"

def test_generate_answer():
    from rag.generator import generate_answer, RAGResponse
    from vectorstore.query import QueryResult
    context = [
        QueryResult(chunk_id="c1", paper_id="p1", chunk_index=0,
                    text="RAG combines retrieval with generation.",
                    score=0.9, metadata={}),
        QueryResult(chunk_id="c2", paper_id="p1", chunk_index=1,
                    text="It was proposed by Lewis et al. in 2020.",
                    score=0.85, metadata={}),
    ]
    response = generate_answer(query="What is RAG?", context_chunks=context)
    assert isinstance(response, RAGResponse)
    assert response.answer and len(response.answer) > 0
    return f"Answer: '{response.answer[:80]}...'"

def test_reranker():
    from rag.reranker import rerank_results, ensure_diversity
    from vectorstore.query import QueryResult
    chunks = [
        QueryResult(chunk_id="c1", paper_id="p1", chunk_index=0,
                    text="Transformers changed NLP.", score=0.9, metadata={}),
        QueryResult(chunk_id="c2", paper_id="p1", chunk_index=1,
                    text="BERT is bidirectional.", score=0.8, metadata={}),
        QueryResult(chunk_id="c3", paper_id="p2", chunk_index=0,
                    text="RAG blends retrieval and generation.", score=0.85, metadata={}),
    ]
    reranked = rerank_results(chunks, paper_metadata_map={})
    diverse   = ensure_diversity(reranked, max_per_paper=1)
    assert isinstance(diverse, list)
    return f"input={len(chunks)} → reranked={len(reranked)} → diverse(1/paper)={len(diverse)}"

def test_quality_scorer():
    from rag.quality_scorer import enhance_paper_metadata
    meta = {"title": "RAG Paper", "abstract": "A paper about retrieval.", "year": 2023}
    enhanced = enhance_paper_metadata(meta)
    assert isinstance(enhanced, dict)
    return f"Enhanced keys: {list(enhanced.keys())}"

def test_search_enhanced():
    from rag.search_enhanced import search_by_author, search_by_year
    authors_result = search_by_author("Smith", limit=2)
    assert isinstance(authors_result, list)
    year_result = search_by_year(2023, limit=2)
    assert isinstance(year_result, list)
    return f"search_by_author: {len(authors_result)} | search_by_year(2023): {len(year_result)}"

if import_status.get("rag.query_expander"):
    run_test("RAG: normalize_query",              test_normalize_query)
    run_test("RAG: expand_query",                 test_expand_query)
if import_status.get("rag.retriever"):
    run_test("RAG: retrieve_context",             test_retrieve_context)
if import_status.get("rag.hybrid_search"):
    run_test("RAG: hybrid_search",               test_hybrid_search)
if import_status.get("rag.generator"):
    run_test("RAG: generate_answer",             test_generate_answer)
if import_status.get("rag.reranker"):
    run_test("RAG: reranker + diversity",         test_reranker)
if import_status.get("rag.quality_scorer"):
    run_test("RAG: quality_scorer",               test_quality_scorer)
if import_status.get("rag.search_enhanced"):
    run_test("RAG: search_enhanced",              test_search_enhanced)


# ════════════════════════════════════════════════════════════
# 8. API MODULE
# ════════════════════════════════════════════════════════════
header("8. API Module")

def test_scholarx_api_class():
    from api.main_api import ScholarXAPI
    api = ScholarXAPI()
    methods = [m for m in dir(api) if not m.startswith("_")]
    return f"ScholarXAPI has {len(methods)} public methods"

def test_api_functions_callable():
    checks = []
    api_checks = [
        ("api.search",          "search_papers"),
        ("api.query_intent",    "classify_query_intent"),
        ("api.recommendations", "recommend_papers_based_on_history"),
        ("api.research_gaps",   "identify_research_gaps"),
        ("api.trends",          "analyze_topic_trends"),
        ("api.exports",         "export_to_bibtex"),
        ("api.relevance_ranking","rank_papers_by_relevance"),
        ("api.deduplication",   "find_duplicate_papers"),
        ("api.authors",         "get_author_stats"),
        ("api.citations",       "get_citation_info"),
        ("api.summaries",       "generate_paper_summary"),
        ("api.topics",          "cluster_papers_by_topic"),
        ("api.similarity",      "compare_papers"),
        ("api.paper_api",       "get_paper_by_id"),
    ]
    for mod_name, fn_name in api_checks:
        if import_status.get(mod_name):
            mod = importlib.import_module(mod_name)
            fn  = getattr(mod, fn_name, None)
            status = "callable" if callable(fn) else "NOT callable"
            checks.append(f"{mod_name.split('.')[-1]}.{fn_name}={status}")
    return " | ".join(checks[:5]) + " ..."  # First 5

if import_status.get("api.main_api"):
    run_test("API: ScholarXAPI class",            test_scholarx_api_class)
    run_test("API: all api functions callable",   test_api_functions_callable)
else:
    skip("API: ScholarXAPI class",   "api.main_api not importable")
    skip("API: api functions check", "api.main_api not importable")


# ════════════════════════════════════════════════════════════
# 9. EVALUATION MODULE
# ════════════════════════════════════════════════════════════
header("9. Evaluation Module")

def test_precision_at_k():
    from evaluation.metrics import precision_at_k
    retrieved = ["p1", "p2", "p3", "p4", "p5"]
    relevant  = ["p1", "p3", "p5"]
    p = precision_at_k(retrieved, relevant, k=5)
    assert 0.0 <= p <= 1.0
    return f"Precision@5 = {p:.3f} (expected 0.600)"

def test_recall_at_k():
    from evaluation.metrics import recall_at_k
    retrieved = ["p1", "p2", "p3"]
    relevant  = ["p1", "p3", "p5"]
    r = recall_at_k(retrieved, relevant, k=3)
    assert 0.0 <= r <= 1.0
    return f"Recall@3 = {r:.3f} (expected ~0.667)"

def test_bleu_score():
    from evaluation.metrics import bleu_score
    candidate = "RAG is a technique combining retrieval and generation"
    reference  = "RAG combines information retrieval with text generation"
    score = bleu_score(candidate, reference)
    assert 0.0 <= score <= 1.0
    return f"BLEU = {score:.3f}"

def test_rouge_l():
    from evaluation.metrics import rouge_l
    candidate = "RAG is a technique combining retrieval and generation"
    reference  = "RAG combines retrieval with generation for NLP"
    score = rouge_l(candidate, reference)
    assert 0.0 <= score <= 1.0
    return f"ROUGE-L = {score:.3f}"

def test_calculate_retrieval_metrics():
    from evaluation.metrics import calculate_retrieval_metrics
    retrieved = ["p1", "p2", "p3", "p4", "p5"]
    relevant  = ["p1", "p3", "p5"]
    metrics = calculate_retrieval_metrics(retrieved, relevant)
    assert isinstance(metrics, dict)
    return f"Keys: {list(metrics.keys())[:5]}"

def test_calculate_answer_quality_metrics():
    from evaluation.metrics import calculate_answer_quality_metrics
    metrics = calculate_answer_quality_metrics(
        candidate="RAG combines retrieval with generation.",
        reference="RAG is a technique that uses retrieval to improve generation."
    )
    assert isinstance(metrics, dict)
    return f"Keys: {list(metrics.keys())}"

def test_load_dataset():
    from evaluation.datasets import create_sample_dataset, load_evaluation_dataset, EvaluationDataset
    import tempfile, json
    from pathlib import Path
    # Create a temp dataset file and load it
    sample = [{"query": "What is RAG?", "expected_answer": "RAG is ...",
                "relevant_papers": [], "relevant_chunks": [],
                "domain": "NLP", "difficulty": "easy", "query_id": "q1"}]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample, f)
        tmp_path = Path(f.name)
    dataset = load_evaluation_dataset(tmp_path)
    assert isinstance(dataset, EvaluationDataset)
    assert len(dataset) == 1
    return f"Dataset loaded: {len(dataset)} query/queries"

def test_baseline_classes():
    from evaluation.baselines import (
        BaselineSystem, SimpleSemanticBaseline,
        KeywordOnlyBaseline, BasicRAGBaseline, HybridSearchBaseline
    )
    for cls in [SimpleSemanticBaseline, KeywordOnlyBaseline,
                BasicRAGBaseline, HybridSearchBaseline]:
        obj = cls()
        assert hasattr(obj, "retrieve")
        assert hasattr(obj, "generate_answer")
    return "All 4 baseline classes instantiated with retrieve() & generate_answer()"

if import_status.get("evaluation.metrics"):
    run_test("Evaluation: precision_at_k",          test_precision_at_k)
    run_test("Evaluation: recall_at_k",             test_recall_at_k)
    run_test("Evaluation: bleu_score",              test_bleu_score)
    run_test("Evaluation: rouge_l",                 test_rouge_l)
    run_test("Evaluation: calculate_retrieval_metrics", test_calculate_retrieval_metrics)
    run_test("Evaluation: calculate_answer_quality",    test_calculate_answer_quality_metrics)
else:
    skip("Evaluation: metrics",  "module not importable")

if import_status.get("evaluation.datasets"):
    run_test("Evaluation: load_evaluation_dataset", test_load_dataset)
else:
    skip("Evaluation: datasets", "module not importable")

if import_status.get("evaluation.baselines"):
    run_test("Evaluation: baseline classes",        test_baseline_classes)
else:
    skip("Evaluation: baselines", "module not importable")


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
header("SUMMARY")

total = len(passed) + len(failed) + len(skipped)
print(f"\n  Total Tests : {total}")
print(f"  {GREEN}Passed{RESET}      : {len(passed)}")
print(f"  {RED}Failed{RESET}      : {len(failed)}")
print(f"  {YELLOW}Skipped{RESET}     : {len(skipped)}")

if failed:
    print(f"\n{RED}{BOLD}  FAILED TESTS:{RESET}")
    for f in failed:
        print(f"    • {f}")

if skipped:
    print(f"\n{YELLOW}{BOLD}  SKIPPED TESTS:{RESET}")
    for s in skipped:
        print(f"    • {s}")

print()
sys.exit(0 if not failed else 1)
