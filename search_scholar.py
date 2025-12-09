"""Google Scholar-like search interface."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from api.search import search_papers, compare_papers_side_by_side, get_paper_details_scholar_style
from api.main_api import api
from config.settings import settings
from manage_papers import get_statistics

settings.embedding_provider = "sentence-transformers"
settings.llm_provider = "simple"


def display_search_results(results: dict):
    """Display search results in Google Scholar style."""
    papers = results.get("papers", [])
    total = results.get("total", 0)
    query = results.get("query", "")
    filters = results.get("filters", {})
    
    print("\n" + "=" * 80)
    print(f"📚 Search Results ({total} papers found)")
    print("=" * 80)
    
    if query:
        print(f"Query: '{query}'")
    if filters.get("author"):
        print(f"Author: '{filters['author']}'")
    if filters.get("year"):
        print(f"Year: {filters['year']}")
    
    print("-" * 80)
    
    if not papers:
        print("No papers found. Try a different search.")
        return
    
    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] {paper.get('title', 'Unknown')}")
        print(f"    Authors: {paper.get('authors', 'Unknown')}")
        if paper.get('year'):
            print(f"    Year: {paper.get('year')}")
        if paper.get('abstract'):
            print(f"    Abstract: {paper.get('abstract')[:150]}...")
        if paper.get('score'):
            print(f"    Relevance: {paper.get('score'):.2%}")
        print(f"    Paper ID: {paper.get('paper_id')}")
        print("-" * 80)


def interactive_search():
    """Interactive Google Scholar-like search."""
    print("=" * 80)
    print("🔍 ScholarX - Google Scholar-like Search")
    print("=" * 80)
    
    get_statistics()
    
    print("\n💡 Search Options:")
    print("   1. Search by topic/keyword")
    print("   2. Search by author")
    print("   3. Search by year")
    print("   4. Combined search (topic + author + year)")
    print("   5. Compare papers")
    print("   6. View paper details")
    print("   7. Exit")
    
    while True:
        print("\n" + "-" * 80)
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == "1":
            query = input("Enter search query: ").strip()
            if query:
                results = search_papers(query=query, limit=10)
                display_search_results(results)
        
        elif choice == "2":
            author = input("Enter author name: ").strip()
            if author:
                results = search_papers(author=author, limit=10)
                display_search_results(results)
        
        elif choice == "3":
            try:
                year = int(input("Enter year: ").strip())
                results = search_papers(year=year, limit=10)
                display_search_results(results)
            except ValueError:
                print("Invalid year. Please enter a number.")
        
        elif choice == "4":
            query = input("Enter search query (optional): ").strip() or None
            author = input("Enter author name (optional): ").strip() or None
            year_str = input("Enter year (optional): ").strip()
            year = int(year_str) if year_str else None
            
            results = search_papers(query=query, author=author, year=year, limit=10)
            display_search_results(results)
        
        elif choice == "5":
            print("\nEnter paper IDs to compare (comma-separated):")
            paper_ids_input = input("Paper IDs: ").strip()
            paper_ids = [pid.strip() for pid in paper_ids_input.split(",") if pid.strip()]
            
            if len(paper_ids) >= 2:
                comparison = compare_papers_side_by_side(paper_ids)
                
                print("\n" + "=" * 80)
                print("📊 Paper Comparison")
                print("=" * 80)
                
                for paper in comparison.get("papers", []):
                    print(f"\n📄 {paper['title']}")
                    print(f"   Authors: {paper['authors']}")
                    print(f"   Year: {paper.get('year', 'N/A')}")
                    print(f"   Chunks: {paper['chunk_count']}")
                    print(f"   Related Papers: {paper['related_papers_count']}")
                    print(f"   Citation Score: {paper['citation_score']}")
                    print(f"   Rank: {paper['rank']}")
                
                if comparison.get("comparison_metrics", {}).get("similarities"):
                    print("\n🔗 Similarities:")
                    for sim in comparison["comparison_metrics"]["similarities"]:
                        print(f"   {sim['paper1']} ↔ {sim['paper2']}: {sim['similarity_percent']}")
            else:
                print("Please provide at least 2 paper IDs.")
        
        elif choice == "6":
            paper_id = input("Enter paper ID: ").strip()
            if paper_id:
                details = get_paper_details_scholar_style(paper_id)
                if details:
                    print("\n" + "=" * 80)
                    print(f"📄 {details['title']}")
                    print("=" * 80)
                    print(f"Authors: {details['authors']}")
                    print(f"Year: {details.get('year', 'N/A')}")
                    print(f"\nAbstract:\n{details.get('abstract', 'N/A')}")
                    print(f"\nMetrics:")
                    for key, value in details.get('metrics', {}).items():
                        print(f"  {key}: {value}")
                    if details.get('summary', {}).get('bullets'):
                        print(f"\nKey Points:")
                        for bullet in details['summary']['bullets'][:5]:
                            print(f"  • {bullet}")
                else:
                    print("Paper not found.")
        
        elif choice == "7":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("Invalid option. Please select 1-7.")


if __name__ == "__main__":
    interactive_search()

