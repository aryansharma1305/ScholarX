"""Showcase all ScholarX features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from api.main_api import api
from config.settings import settings

# Set to free mode
settings.embedding_provider = "sentence-transformers"
settings.llm_provider = "simple"


def showcase_features():
    """Demonstrate all available features."""
    print("=" * 70)
    print("ScholarX - Feature Showcase")
    print("=" * 70)
    
    # Get collection stats first
    from manage_papers import get_statistics
    get_statistics()
    
    print("\n" + "=" * 70)
    print("📚 Paper API Features")
    print("=" * 70)
    
    # Get a sample paper ID
    from config.chroma_client import get_collection
    collection = get_collection()
    sample = collection.get(limit=1)
    if sample.get("ids"):
        paper_id = sample["metadatas"][0].get("paper_id")
        
        print(f"\n1. Get Paper Details (ID: {paper_id})")
        paper = api.get_paper(paper_id)
        if paper:
            print(f"   Title: {paper['title'][:60]}...")
            print(f"   Authors: {paper['authors'][:50]}...")
            print(f"   Chunks: {paper['chunk_count']}")
        
        print(f"\n2. Get Paper Summary")
        summary = api.get_paper_summary_api(paper_id)
        if summary:
            print(f"   Abstract: {summary['abstract'][:100]}...")
            print(f"   Key Insights: {len(summary.get('key_insights', []))} points")
        
        print(f"\n3. Get Citations")
        citations = api.get_citations(paper_id)
        print(f"   Related Papers: {len(citations.get('related_papers', []))}")
        
        print(f"\n4. Generate Summary")
        summary_data = api.generate_summary(paper_id, use_llm=False)
        if summary_data:
            print(f"   Short: {summary_data.get('short', '')[:80]}...")
            print(f"   Bullets: {len(summary_data.get('bullets', []))}")
    
    print("\n" + "=" * 70)
    print("👥 Author Features")
    print("=" * 70)
    
    author_stats = api.get_author_statistics()
    print(f"Total Authors: {author_stats.get('total_authors', 0)}")
    print(f"Top Authors: {len(author_stats.get('top_authors', []))}")
    
    if author_stats.get('top_authors'):
        top = author_stats['top_authors'][0]
        print(f"\nTop Author: {top['name']}")
        print(f"   Papers: {top['paper_count']}")
        
        print(f"\n5. Get Author Profile")
        profile = api.get_author(top['name'])
        print(f"   Papers: {profile.get('paper_count', 0)}")
        print(f"   Co-authors: {len(profile.get('co_authors', []))}")
    
    print("\n" + "=" * 70)
    print("🎯 Topic Clustering")
    print("=" * 70)
    
    print("\n6. Cluster Papers by Topic")
    clusters = api.cluster_topics(num_clusters=3)
    print(f"   Created {len(clusters)} clusters")
    for cluster_id, cluster_info in list(clusters.items())[:3]:
        print(f"   {cluster_id}: {cluster_info['topic']} ({cluster_info['paper_count']} papers)")
    
    print("\n" + "=" * 70)
    print("💬 Advanced RAG Modes")
    print("=" * 70)
    
    test_query = "What is transformer architecture?"
    
    print(f"\n7. Concise Mode")
    result = api.rag_concise(test_query)
    print(f"   Answer: {result['answer'][:100]}...")
    
    print(f"\n8. Compare Mode")
    result = api.rag_compare(test_query)
    print(f"   Papers compared: {len(result.get('citations_by_paper', {}))}")
    
    print("\n" + "=" * 70)
    print("🔍 Deduplication")
    print("=" * 70)
    
    print("\n9. Find Duplicates")
    duplicates = api.find_duplicates(threshold=0.85)
    print(f"   Found {len(duplicates)} duplicate groups")
    
    print("\n10. Normalize ArXiv Versions")
    versions = api.normalize_versions()
    print(f"   Papers with multiple versions: {len(versions)}")
    
    print("\n" + "=" * 70)
    print("📊 Analytics")
    print("=" * 70)
    
    print("\n11. Query Statistics")
    stats = api.get_query_statistics()
    print(f"   Total Queries: {stats.get('total_queries', 0)}")
    print(f"   Avg Time: {stats.get('avg_time_seconds', 0):.2f}s")
    
    print("\n12. Citation Rankings")
    rankings = api.get_citation_rankings()
    print(f"   Ranked Papers: {len(rankings.get('ranked_papers', []))}")
    
    print("\n" + "=" * 70)
    print("✅ Feature Showcase Complete!")
    print("=" * 70)


if __name__ == "__main__":
    showcase_features()

