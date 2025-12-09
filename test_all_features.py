"""Comprehensive test suite for all ScholarX features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from api.main_api import api
from config.settings import settings
from config.chroma_client import get_collection

# Set to free mode
settings.embedding_provider = "sentence-transformers"
settings.llm_provider = "simple"

# Test results
results = {"passed": [], "failed": [], "warnings": []}


def test(name: str, func, *args, **kwargs):
    """Run a test and record results."""
    try:
        result = func(*args, **kwargs)
        results["passed"].append(name)
        return result
    except Exception as e:
        results["failed"].append((name, str(e)))
        print(f"❌ {name}: {e}")
        return None


def test_paper_api():
    """Test Paper API features."""
    print("\n" + "=" * 70)
    print("📚 Testing Paper API")
    print("=" * 70)
    
    collection = get_collection()
    sample = collection.get(limit=1)
    if not sample.get("ids"):
        print("⚠️  No papers in collection - skipping paper API tests")
        return
    
    paper_id = sample["metadatas"][0].get("paper_id")
    
    # Test get_paper
    paper = test("get_paper", api.get_paper, paper_id)
    if paper:
        assert "title" in paper, "Missing title"
        assert "authors" in paper, "Missing authors"
        print(f"✅ get_paper: {paper['title'][:50]}...")
    
    # Test get_paper_summary
    summary = test("get_paper_summary", api.get_paper_summary_api, paper_id)
    if summary:
        assert "abstract" in summary, "Missing abstract"
        print(f"✅ get_paper_summary: {len(summary.get('key_insights', []))} insights")
    
    # Test generate_summary
    summary_data = test("generate_summary", api.generate_summary, paper_id, False)
    if summary_data:
        assert "short" in summary_data, "Missing short summary"
        print(f"✅ generate_summary: {len(summary_data.get('bullets', []))} bullets")


def test_citations():
    """Test Citation features."""
    print("\n" + "=" * 70)
    print("🔗 Testing Citations")
    print("=" * 70)
    
    collection = get_collection()
    sample = collection.get(limit=1)
    if not sample.get("ids"):
        print("⚠️  No papers - skipping citation tests")
        return
    
    paper_id = sample["metadatas"][0].get("paper_id")
    
    # Test get_citations
    citations = test("get_citations", api.get_citations, paper_id)
    if citations:
        assert "related_papers" in citations, "Missing related_papers"
        print(f"✅ get_citations: {len(citations.get('related_papers', []))} related")
    
    # Test get_related_papers
    related = test("get_related_papers", api.get_related_papers, paper_id, 5)
    if related:
        print(f"✅ get_related_papers: {len(related)} papers")


def test_authors():
    """Test Author features."""
    print("\n" + "=" * 70)
    print("👥 Testing Authors")
    print("=" * 70)
    
    # Test get_author_statistics
    stats = test("get_author_statistics", api.get_author_statistics)
    if stats:
        assert "total_authors" in stats, "Missing total_authors"
        print(f"✅ get_author_statistics: {stats.get('total_authors', 0)} authors")
        
        if stats.get('top_authors'):
            top_author = stats['top_authors'][0]['name']
            # Test get_author
            profile = test("get_author", api.get_author, top_author)
            if profile:
                assert "paper_count" in profile, "Missing paper_count"
                print(f"✅ get_author: {profile.get('paper_count', 0)} papers")


def test_topics():
    """Test Topic clustering."""
    print("\n" + "=" * 70)
    print("🎯 Testing Topics")
    print("=" * 70)
    
    collection = get_collection()
    count = collection.count()
    
    if count < 3:
        print("⚠️  Not enough papers for clustering - skipping")
        return
    
    # Test cluster_topics
    clusters = test("cluster_topics", api.cluster_topics, 3)
    if clusters:
        assert len(clusters) > 0, "No clusters created"
        print(f"✅ cluster_topics: {len(clusters)} clusters")
        
        # Test get_paper_topics
        sample = collection.get(limit=1)
        if sample.get("ids"):
            paper_id = sample["metadatas"][0].get("paper_id")
            topics = test("get_paper_topics", api.get_paper_topics_api, paper_id)
            if topics is not None:
                print(f"✅ get_paper_topics: {len(topics)} topics")


def test_rag_modes():
    """Test Advanced RAG modes."""
    print("\n" + "=" * 70)
    print("💬 Testing RAG Modes")
    print("=" * 70)
    
    test_query = "What is transformer architecture?"
    
    # Test concise mode
    result = test("rag_concise", api.rag_concise, test_query)
    if result:
        assert "answer" in result, "Missing answer"
        print(f"✅ rag_concise: {len(result['answer'])} chars")
    
    # Test detailed mode
    result = test("rag_detailed", api.rag_detailed, test_query)
    if result:
        assert "answer" in result, "Missing answer"
        print(f"✅ rag_detailed: {len(result['answer'])} chars")
    
    # Test explain mode
    result = test("rag_explain", api.rag_explain, test_query)
    if result:
        assert "answer" in result, "Missing answer"
        print(f"✅ rag_explain: {len(result['answer'])} chars")
    
    # Test compare mode
    result = test("rag_compare", api.rag_compare, test_query)
    if result:
        assert "answer" in result, "Missing answer"
        print(f"✅ rag_compare: {len(result.get('citations_by_paper', {}))} papers")
    
    # Test survey mode
    result = test("rag_survey", api.rag_survey, "neural networks")
    if result:
        assert "survey" in result, "Missing survey"
        print(f"✅ rag_survey: {result.get('total_papers_cited', 0)} papers cited")


def test_deduplication():
    """Test Deduplication features."""
    print("\n" + "=" * 70)
    print("🔍 Testing Deduplication")
    print("=" * 70)
    
    # Test find_duplicates
    duplicates = test("find_duplicates", api.find_duplicates, 0.85)
    if duplicates is not None:
        print(f"✅ find_duplicates: {len(duplicates)} groups")
    
    # Test normalize_versions
    versions = test("normalize_versions", api.normalize_versions)
    if versions is not None:
        print(f"✅ normalize_versions: {len(versions)} papers with versions")


def test_similarity():
    """Test Similarity features."""
    print("\n" + "=" * 70)
    print("🔎 Testing Similarity")
    print("=" * 70)
    
    collection = get_collection()
    sample = collection.get(limit=2)
    
    if len(sample.get("ids", [])) >= 2:
        paper_id1 = sample["metadatas"][0].get("paper_id")
        paper_id2 = sample["metadatas"][1].get("paper_id")
        
        # Test compare_papers
        comparison = test("compare_papers", api.compare_two_papers, paper_id1, paper_id2)
        if comparison:
            assert "similarity" in comparison, "Missing similarity"
            print(f"✅ compare_papers: {comparison.get('similarity_percent', 'N/A')}")
    
    # Test check_similarity
    similar = test("check_similarity", api.check_similarity, "transformer architecture neural networks", 0.5)
    if similar is not None:
        print(f"✅ check_similarity: {len(similar)} similar papers")


def test_ranking():
    """Test Ranking features."""
    print("\n" + "=" * 70)
    print("📊 Testing Rankings")
    print("=" * 70)
    
    # Test get_citation_rankings
    rankings = test("get_citation_rankings", api.get_citation_rankings)
    if rankings:
        assert "ranked_papers" in rankings, "Missing ranked_papers"
        print(f"✅ get_citation_rankings: {len(rankings.get('ranked_papers', []))} papers")
        
        if rankings.get('ranked_papers'):
            paper_id = rankings['ranked_papers'][0]['paper_id']
            # Test get_paper_ranking
            rank = test("get_paper_ranking", api.get_paper_ranking, paper_id)
            if rank:
                assert "rank" in rank, "Missing rank"
                print(f"✅ get_paper_ranking: Rank {rank.get('rank')}")


def test_analytics():
    """Test Analytics features."""
    print("\n" + "=" * 70)
    print("📈 Testing Analytics")
    print("=" * 70)
    
    # Test get_query_statistics
    stats = test("get_query_statistics", api.get_query_statistics)
    if stats:
        assert "total_queries" in stats, "Missing total_queries"
        print(f"✅ get_query_statistics: {stats.get('total_queries', 0)} queries")


def test_multi_document():
    """Test Multi-document RAG."""
    print("\n" + "=" * 70)
    print("📚 Testing Multi-Document RAG")
    print("=" * 70)
    
    collection = get_collection()
    all_data = collection.get(limit=3)
    
    if len(all_data.get("ids", [])) >= 2:
        paper_ids = []
        seen = set()
        for metadata in all_data.get("metadatas", []):
            pid = metadata.get("paper_id")
            if pid and pid not in seen:
                paper_ids.append(pid)
                seen.add(pid)
                if len(paper_ids) >= 2:
                    break
        
        if len(paper_ids) >= 2:
            result = test("rag_multi_document", api.rag_multi_document, paper_ids, "What is the main contribution?")
            if result:
                assert "answer" in result, "Missing answer"
                print(f"✅ rag_multi_document: {len(result.get('papers_used', []))} papers used")


def main():
    """Run all tests."""
    print("=" * 70)
    print("ScholarX - Comprehensive Feature Test Suite")
    print("=" * 70)
    
    # Check collection
    collection = get_collection()
    count = collection.count()
    print(f"\n📊 Collection: {count} chunks")
    
    if count == 0:
        print("\n⚠️  No papers in collection. Some tests will be skipped.")
        print("   Run: python3 add_papers.py")
    
    # Run all test suites
    test_paper_api()
    test_citations()
    test_authors()
    test_topics()
    test_rag_modes()
    test_deduplication()
    test_similarity()
    test_ranking()
    test_analytics()
    test_multi_document()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    print(f"✅ Passed: {len(results['passed'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    
    if results['failed']:
        print("\n❌ Failed Tests:")
        for name, error in results['failed']:
            print(f"   - {name}: {error}")
    
    if len(results['passed']) > 0:
        print("\n✅ All tests completed!")
    else:
        print("\n⚠️  No tests passed")


if __name__ == "__main__":
    main()

