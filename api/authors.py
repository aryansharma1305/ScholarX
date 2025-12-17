"""Author graph and author-related features."""
from typing import List, Dict
from collections import defaultdict
from config.chroma_client import get_collection
from utils.logger import get_logger

logger = get_logger(__name__)


def get_author_stats() -> Dict:
    """
    Get statistics about authors in the collection.
    
    Returns:
        Dictionary with top authors, co-author networks, etc.
    """
    collection = get_collection()
    count = collection.count()
    
    if count == 0:
        return {
            "total_authors": 0,
            "top_authors": [],
            "co_author_network": {}
        }
    
    all_data = collection.get(limit=count)
    
    # Collect author information
    author_papers = defaultdict(set)  # author -> set of paper_ids
    author_chunks = defaultdict(int)  # author -> chunk count
    co_authors = defaultdict(set)  # author -> set of co-authors
    
    for metadata in all_data.get("metadatas", []):
        authors_str = metadata.get("authors", "")
        paper_id = metadata.get("paper_id", "")
        
        if authors_str:
            # Parse authors (comma-separated)
            authors = [a.strip() for a in authors_str.split(",")]
            
            for author in authors:
                if author and len(author) > 2:
                    author_papers[author].add(paper_id)
                    author_chunks[author] += 1
                    
                    # Build co-author network
                    for co_author in authors:
                        if co_author != author:
                            co_authors[author].add(co_author)
    
    # Get top authors by paper count
    top_authors = sorted(
        author_papers.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:20]
    
    top_authors_list = [
        {
            "name": author,
            "paper_count": len(papers),
            "chunk_count": author_chunks[author]
        }
        for author, papers in top_authors
    ]
    
    return {
        "total_authors": len(author_papers),
        "top_authors": top_authors_list,
        "co_author_network": {
            author: list(co_authors[author])[:10]  # Limit co-authors
            for author in list(author_papers.keys())[:50]  # Limit to top 50
        }
    }


def get_author_profile(author_name: str) -> Dict:
    """
    Get profile for a specific author.
    
    Returns:
        Author profile with papers, co-authors, etc.
    """
    collection = get_collection()
    count = collection.count()
    
    if count == 0:
        return {}
    
    all_data = collection.get(limit=count)
    
    papers = {}
    co_authors = set()
    
    for metadata in all_data.get("metadatas", []):
        authors_str = metadata.get("authors", "")
        paper_id = metadata.get("paper_id", "")
        
        if author_name.lower() in authors_str.lower():
            # This author is in this paper
            papers[paper_id] = {
                "paper_id": paper_id,
                "title": metadata.get("title", "Unknown"),
                "year": metadata.get("year"),
                "authors": authors_str
            }
            
            # Collect co-authors
            authors = [a.strip() for a in authors_str.split(",")]
            for co_author in authors:
                if co_author.lower() != author_name.lower():
                    co_authors.add(co_author)
    
    return {
        "name": author_name,
        "paper_count": len(papers),
        "papers": list(papers.values()),
        "co_authors": list(co_authors)[:20]
    }



