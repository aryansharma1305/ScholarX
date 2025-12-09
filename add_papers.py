"""Script to add more papers to the RAG pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.paper_fetcher import fetch_papers_by_topic
from ingestion.ingest_pipeline import ingest_pdf_from_url
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Set to free mode
settings.embedding_provider = "sentence-transformers"
settings.llm_provider = "simple"


def add_papers_by_topic(topic: str, num_papers: int = 5):
    """Add papers on a specific topic."""
    print(f"\n📚 Fetching {num_papers} papers on: '{topic}'")
    print("=" * 70)
    
    papers = fetch_papers_by_topic(topic, max_papers=num_papers)
    
    if not papers:
        print("❌ No papers found with PDFs")
        return
    
    print(f"\n✅ Found {len(papers)} papers. Starting ingestion...\n")
    
    successful = 0
    failed = 0
    
    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}/{len(papers)}] Processing: {paper['title'][:60]}...")
        print(f"   Source: {paper.get('source', 'unknown')}")
        print(f"   Authors: {paper.get('authors_string', 'Unknown')[:50]}...")
        
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
            print(f"   ✅ Successfully ingested! (ID: {paper_id[:20]}...)")
            successful += 1
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Summary: {successful} successful, {failed} failed")
    print("=" * 70)


def add_paper_from_url(pdf_url: str, paper_id: str = None):
    """Add a single paper from PDF URL."""
    print(f"\n📄 Adding paper from URL: {pdf_url}")
    print("=" * 70)
    
    try:
        paper_id = ingest_pdf_from_url(pdf_url, paper_id=paper_id)
        print(f"✅ Successfully ingested paper! (ID: {paper_id})")
        return paper_id
    except Exception as e:
        print(f"❌ Failed to ingest: {e}")
        return None


def main():
    """Interactive paper addition."""
    print("=" * 70)
    print("ScholarX - Add Papers to RAG Pipeline")
    print("=" * 70)
    print("\nOptions:")
    print("1. Add papers by topic (fetches from ArXiv/Semantic Scholar)")
    print("2. Add paper from PDF URL")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        topic = input("\nEnter topic/keywords: ").strip()
        if not topic:
            print("❌ Topic cannot be empty")
            return
        
        num = input("Number of papers to fetch (default 5): ").strip()
        num_papers = int(num) if num.isdigit() else 5
        
        add_papers_by_topic(topic, num_papers)
    
    elif choice == "2":
        url = input("\nEnter PDF URL: ").strip()
        if not url:
            print("❌ URL cannot be empty")
            return
        
        paper_id = input("Paper ID (optional, press Enter to auto-generate): ").strip()
        paper_id = paper_id if paper_id else None
        
        add_paper_from_url(url, paper_id)
    
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()

