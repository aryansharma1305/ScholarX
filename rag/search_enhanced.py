"""Enhanced search capabilities."""
from typing import List, Dict, Optional
from vectorstore.query import QueryResult, query_vectors
from config.chroma_client import get_collection
from processing.embeddings import generate_embedding
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def search_by_author(author_name: str, limit: int = 10) -> List[Dict]:
    """Search for papers by author name."""
    collection = get_collection()
    
    try:
        # ChromaDB doesn't support full-text search in metadata well,
        # so we'll get all and filter
        all_data = collection.get(limit=10000)  # Get up to 10k
        
        papers = {}
        for i, metadata in enumerate(all_data.get("metadatas", [])):
            authors = metadata.get("authors", "").lower()
            if author_name.lower() in authors:
                paper_id = metadata.get("paper_id", "unknown")
                if paper_id not in papers:
                    papers[paper_id] = {
                        "paper_id": paper_id,
                        "title": metadata.get("title", "Unknown"),
                        "authors": metadata.get("authors", "Unknown"),
                        "year": metadata.get("year"),
                        "source": metadata.get("source"),
                    }
        
        results = list(papers.values())[:limit]
        logger.info(f"Found {len(results)} papers by author '{author_name}'")
        return results
        
    except Exception as e:
        logger.error(f"Error searching by author: {e}")
        return []


def search_by_year(year: int, limit: int = 10) -> List[Dict]:
    """Search for papers by publication year."""
    collection = get_collection()
    
    try:
        all_data = collection.get(
            where={"year": year},
            limit=limit * 10  # Get more to find unique papers
        )
        
        papers = {}
        for i, metadata in enumerate(all_data.get("metadatas", [])):
            paper_id = metadata.get("paper_id", "unknown")
            if paper_id not in papers:
                papers[paper_id] = {
                    "paper_id": paper_id,
                    "title": metadata.get("title", "Unknown"),
                    "authors": metadata.get("authors", "Unknown"),
                    "year": metadata.get("year"),
                }
        
        results = list(papers.values())[:limit]
        logger.info(f"Found {len(results)} papers from year {year}")
        return results
        
    except Exception as e:
        logger.error(f"Error searching by year: {e}")
        return []


def search_by_keyword(keyword: str, limit: int = 10) -> List[QueryResult]:
    """Search for chunks containing a specific keyword."""
    # Use semantic search for keyword
    query_embedding = generate_embedding(keyword)
    results = query_vectors(query_embedding, top_k=limit)
    
    # Filter to ensure keyword appears in text
    filtered = [r for r in results if keyword.lower() in r.text.lower()]
    
    logger.info(f"Found {len(filtered)} chunks containing '{keyword}'")
    return filtered


def get_related_papers(paper_id: str, limit: int = 5) -> List[Dict]:
    """Get papers related to a given paper (by topic similarity)."""
    collection = get_collection()
    
    try:
        # Get a sample chunk from the paper
        paper_chunks = collection.get(
            where={"paper_id": paper_id},
            limit=1
        )
        
        if not paper_chunks.get("ids"):
            logger.warning(f"No chunks found for paper {paper_id}")
            return []
        
        # Get embedding of first chunk
        # Note: ChromaDB doesn't return embeddings in get(), so we'll use the text
        chunk_text = paper_chunks.get("documents", [""])[0] if paper_chunks.get("documents") else ""
        if not chunk_text:
            return []
        
        # Generate embedding and search
        query_embedding = generate_embedding(chunk_text[:500])  # Use first 500 chars
        
        # Search for similar chunks
        similar = query_vectors(query_embedding, top_k=limit * 3)
        
        # Get unique papers (excluding the original)
        related_papers = {}
        for result in similar:
            pid = result.paper_id
            if pid != paper_id and pid not in related_papers:
                related_papers[pid] = {
                    "paper_id": pid,
                    "score": result.score
                }
        
        results = list(related_papers.values())[:limit]
        results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Found {len(results)} related papers")
        return results
        
    except Exception as e:
        logger.error(f"Error finding related papers: {e}")
        return []

