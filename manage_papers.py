"""Paper management and batch processing tool."""
import sys
import json
from pathlib import Path
from typing import List, Dict
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.paper_fetcher import fetch_papers_by_topic, search_semantic_scholar, search_arxiv
from ingestion.ingest_pipeline import ingest_pdf_from_url
from config.settings import settings
from config.chroma_client import get_collection
from utils.logger import get_logger
from utils.timers import timer

logger = get_logger(__name__)

# Set to free mode
settings.embedding_provider = "sentence-transformers"
settings.llm_provider = "simple"


def batch_add_papers(topics: List[str], papers_per_topic: int = 10):
    """Batch add papers from multiple topics."""
    print(f"\n📚 Batch Adding Papers")
    print("=" * 70)
    print(f"Topics: {', '.join(topics)}")
    print(f"Papers per topic: {papers_per_topic}")
    print("=" * 70)
    
    all_papers = []
    seen_titles = set()
    
    for topic in topics:
        print(f"\n🔍 Fetching papers for: '{topic}'")
        papers = fetch_papers_by_topic(topic, max_papers=papers_per_topic)
        
        for paper in papers:
            title_lower = paper['title'].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                all_papers.append(paper)
    
    print(f"\n✅ Found {len(all_papers)} unique papers total")
    print(f"\n🚀 Starting batch ingestion...\n")
    
    successful = 0
    failed = 0
    failed_papers = []
    
    with timer("Batch Ingestion"):
        for i, paper in enumerate(all_papers, 1):
            print(f"[{i}/{len(all_papers)}] {paper['title'][:60]}...")
            
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
                print(f"   ✅ Ingested (ID: {paper_id[:20]}...)")
                successful += 1
            except Exception as e:
                print(f"   ❌ Failed: {str(e)[:50]}")
                failed += 1
                failed_papers.append({
                    "title": paper.get("title"),
                    "url": paper.get("pdf_url"),
                    "error": str(e)
                })
    
    print("\n" + "=" * 70)
    print(f"📊 Batch Ingestion Complete!")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print("=" * 70)
    
    if failed_papers:
        with open("failed_papers.json", "w") as f:
            json.dump(failed_papers, f, indent=2)
        print(f"\n💾 Failed papers saved to: failed_papers.json")
    
    return successful, failed


def add_papers_from_file(filepath: str):
    """Add papers from a JSON file with paper URLs/metadata."""
    print(f"\n📄 Adding papers from file: {filepath}")
    print("=" * 70)
    
    try:
        with open(filepath, 'r') as f:
            papers = json.load(f)
        
        if not isinstance(papers, list):
            print("❌ File must contain a JSON array of papers")
            return
        
        print(f"Found {len(papers)} papers in file\n")
        
        successful = 0
        for i, paper in enumerate(papers, 1):
            pdf_url = paper.get("pdf_url") or paper.get("url")
            if not pdf_url:
                print(f"[{i}] ❌ No PDF URL found")
                continue
            
            print(f"[{i}/{len(papers)}] {paper.get('title', 'Unknown')[:60]}...")
            try:
                paper_id = ingest_pdf_from_url(
                    pdf_url=pdf_url,
                    paper_id=paper.get("paper_id"),
                    metadata=paper.get("metadata", {})
                )
                print(f"   ✅ Ingested")
                successful += 1
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        print(f"\n✅ Successfully added {successful}/{len(papers)} papers")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def list_papers():
    """List all papers in the collection."""
    collection = get_collection()
    count = collection.count()
    
    print(f"\n📚 Papers in Collection: {count} chunks")
    print("=" * 70)
    
    if count == 0:
        print("No papers found.")
        return
    
    # Get all papers
    all_data = collection.get(limit=count)
    
    # Group by paper_id
    papers = {}
    for i, paper_id in enumerate(all_data.get("ids", [])):
        metadata = all_data.get("metadatas", [{}])[i] if all_data.get("metadatas") else {}
        pid = metadata.get("paper_id", "unknown")
        
        if pid not in papers:
            papers[pid] = {
                "title": metadata.get("title", "Unknown"),
                "authors": metadata.get("authors", "Unknown"),
                "year": metadata.get("year"),
                "source": metadata.get("source", "unknown"),
                "chunks": 0
            }
        papers[pid]["chunks"] += 1
    
    print(f"\nFound {len(papers)} unique papers:\n")
    for i, (paper_id, info) in enumerate(sorted(papers.items()), 1):
        print(f"{i}. {info['title'][:60]}...")
        print(f"   ID: {paper_id}")
        print(f"   Authors: {info['authors'][:50]}...")
        print(f"   Year: {info.get('year', 'Unknown')}")
        print(f"   Source: {info['source']}")
        print(f"   Chunks: {info['chunks']}")
        print()


def delete_paper(paper_id: str):
    """Delete all chunks for a specific paper."""
    collection = get_collection()
    
    # Get all chunks for this paper
    try:
        chunks = collection.get(
            where={"paper_id": paper_id}
        )
        
        if not chunks.get("ids"):
            print(f"❌ No paper found with ID: {paper_id}")
            return
        
        chunk_ids = chunks["ids"]
        print(f"Found {len(chunk_ids)} chunks for paper {paper_id}")
        
        confirm = input("Delete all chunks? (yes/no): ").strip().lower()
        if confirm == "yes":
            collection.delete(ids=chunk_ids)
            print(f"✅ Deleted {len(chunk_ids)} chunks")
        else:
            print("Cancelled")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def export_papers_list(filename: str = "papers_export.json"):
    """Export list of all papers to JSON."""
    collection = get_collection()
    count = collection.count()
    
    if count == 0:
        print("No papers to export")
        return
    
    all_data = collection.get(limit=count)
    
    # Group by paper_id
    papers = {}
    for i, paper_id in enumerate(all_data.get("ids", [])):
        metadata = all_data.get("metadatas", [{}])[i] if all_data.get("metadatas") else {}
        pid = metadata.get("paper_id", "unknown")
        
        if pid not in papers:
            papers[pid] = {
                "paper_id": pid,
                "title": metadata.get("title", "Unknown"),
                "authors": metadata.get("authors", "Unknown"),
                "year": metadata.get("year"),
                "source": metadata.get("source", "unknown"),
                "pdf_url": metadata.get("pdf_url"),
                "abstract": metadata.get("abstract"),
                "chunk_count": 0
            }
        papers[pid]["chunk_count"] += 1
    
    papers_list = list(papers.values())
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(papers_list, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(papers_list)} papers to {filename}")


def get_statistics():
    """Get detailed statistics about the collection."""
    collection = get_collection()
    count = collection.count()
    
    print("\n" + "=" * 70)
    print("📊 Collection Statistics")
    print("=" * 70)
    
    if count == 0:
        print("Collection is empty")
        return
    
    all_data = collection.get(limit=count)
    
    # Analyze papers
    papers = {}
    sources = {}
    years = {}
    total_chars = 0
    
    for i, paper_id in enumerate(all_data.get("ids", [])):
        metadata = all_data.get("metadatas", [{}])[i] if all_data.get("metadatas") else {}
        document = all_data.get("documents", [""])[i] if all_data.get("documents") else ""
        
        pid = metadata.get("paper_id", "unknown")
        source = metadata.get("source", "unknown")
        year = metadata.get("year")
        
        if pid not in papers:
            papers[pid] = {
                "title": metadata.get("title", "Unknown"),
                "chunks": 0,
                "chars": 0
            }
        
        papers[pid]["chunks"] += 1
        papers[pid]["chars"] += len(document)
        total_chars += len(document)
        
        sources[source] = sources.get(source, 0) + 1
        if year:
            years[year] = years.get(year, 0) + 1
    
    print(f"\n📚 Papers: {len(papers)} unique papers")
    print(f"📄 Chunks: {count} total chunks")
    print(f"📝 Characters: {total_chars:,} total")
    print(f"📊 Avg chunks per paper: {count / len(papers):.1f}")
    print(f"📊 Avg chars per chunk: {total_chars / count:.0f}")
    
    print(f"\n📦 Sources:")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"   {source}: {count} chunks")
    
    if years:
        print(f"\n📅 Years:")
        for year in sorted(years.keys(), reverse=True)[:10]:
            print(f"   {year}: {years[year]} chunks")
    
    print("\n" + "=" * 70)


def main():
    """Main management interface."""
    print("=" * 70)
    print("ScholarX - Paper Management System")
    print("=" * 70)
    print("\nOptions:")
    print("1. Batch add papers by topics")
    print("2. Add papers from JSON file")
    print("3. List all papers")
    print("4. Delete a paper")
    print("5. Export papers list")
    print("6. View statistics")
    print("7. Exit")
    
    choice = input("\nEnter choice (1-7): ").strip()
    
    if choice == "1":
        topics_input = input("Enter topics (comma-separated): ").strip()
        topics = [t.strip() for t in topics_input.split(",") if t.strip()]
        
        if not topics:
            print("❌ No topics provided")
            return
        
        num = input("Papers per topic (default 10): ").strip()
        papers_per_topic = int(num) if num.isdigit() else 10
        
        batch_add_papers(topics, papers_per_topic)
    
    elif choice == "2":
        filepath = input("Enter JSON file path: ").strip()
        if filepath:
            add_papers_from_file(filepath)
        else:
            print("❌ No file path provided")
    
    elif choice == "3":
        list_papers()
    
    elif choice == "4":
        paper_id = input("Enter paper ID to delete: ").strip()
        if paper_id:
            delete_paper(paper_id)
        else:
            print("❌ No paper ID provided")
    
    elif choice == "5":
        filename = input("Export filename (default: papers_export.json): ").strip()
        filename = filename or "papers_export.json"
        export_papers_list(filename)
    
    elif choice == "6":
        get_statistics()
    
    elif choice == "7":
        print("Goodbye!")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()

