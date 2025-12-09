"""Advanced features demonstration and usage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from api.main_api import api
from config.settings import settings

settings.embedding_provider = "sentence-transformers"
settings.llm_provider = "simple"


def demo_all_features():
    """Demonstrate all advanced features."""
    print("=" * 70)
    print("ScholarX - Advanced Features Demo")
    print("=" * 70)
    
    # Get a paper ID for demos
    from config.chroma_client import get_collection
    collection = get_collection()
    sample = collection.get(limit=1)
    
    if not sample.get("ids"):
        print("\n⚠️  No papers in collection. Add some papers first!")
        print("   Run: python3 add_papers.py")
        return
    
    paper_id = sample["metadatas"][0].get("paper_id")
    
    print(f"\n📄 Using paper: {paper_id}")
    print("=" * 70)
    
    # 1. Paper API
    print("\n1️⃣  Paper Details API")
    paper = api.get_paper(paper_id)
    if paper:
        print(f"   ✅ Title: {paper['title'][:50]}...")
        print(f"   ✅ Authors: {paper['authors'][:40]}...")
        print(f"   ✅ Chunks: {paper['chunk_count']}")
    
    # 2. Summaries
    print("\n2️⃣  Automatic Summaries")
    summary = api.generate_summary(paper_id)
    if summary:
        print(f"   ✅ Short: {summary['short'][:80]}...")
        print(f"   ✅ Bullets: {len(summary.get('bullets', []))} insights")
    
    # 3. Citations
    print("\n3️⃣  Citation Graph")
    citations = api.get_citations(paper_id)
    print(f"   ✅ Related Papers: {len(citations.get('related_papers', []))}")
    
    # 4. Authors
    print("\n4️⃣  Author Graph")
    author_stats = api.get_author_statistics()
    print(f"   ✅ Total Authors: {author_stats.get('total_authors', 0)}")
    if author_stats.get('top_authors'):
        print(f"   ✅ Top Author: {author_stats['top_authors'][0]['name']}")
    
    # 5. Topics
    print("\n5️⃣  Topic Clustering")
    clusters = api.cluster_topics(num_clusters=3)
    print(f"   ✅ Clusters: {len(clusters)}")
    
    # 6. RAG Modes
    print("\n6️⃣  Advanced RAG Modes")
    test_query = "What is the main contribution?"
    
    print("   - Concise Mode:")
    result = api.rag_concise(test_query)
    print(f"     Answer length: {len(result['answer'])} chars")
    
    print("   - Compare Mode:")
    result = api.rag_compare(test_query)
    print(f"     Papers compared: {len(result.get('citations_by_paper', {}))}")
    
    # 7. Deduplication
    print("\n7️⃣  Deduplication")
    duplicates = api.find_duplicates()
    print(f"   ✅ Duplicate groups: {len(duplicates)}")
    
    # 8. Similarity
    print("\n8️⃣  Similarity Checking")
    if len(collection.get(limit=2).get("ids", [])) >= 2:
        paper_id2 = collection.get(limit=2)["metadatas"][1].get("paper_id")
        comparison = api.compare_two_papers(paper_id, paper_id2)
        print(f"   ✅ Similarity: {comparison.get('similarity_percent', 'N/A')}")
    
    # 9. Rankings
    print("\n9️⃣  Citation Rankings")
    rankings = api.get_citation_rankings()
    print(f"   ✅ Ranked Papers: {len(rankings.get('ranked_papers', []))}")
    
    # 10. Analytics
    print("\n🔟 Query Analytics")
    stats = api.get_query_statistics()
    print(f"   ✅ Total Queries: {stats.get('total_queries', 0)}")
    
    print("\n" + "=" * 70)
    print("✅ All Features Demonstrated!")
    print("=" * 70)
    print("\n💡 Use api.main_api.api to access all features programmatically")


if __name__ == "__main__":
    demo_all_features()

