"""Paper API - Get paper details, metadata, chunks."""
from typing import Dict, List, Optional
from config.chroma_client import get_collection
from vectorstore.query import query_by_paper_id
from utils.logger import get_logger

logger = get_logger(__name__)


def get_paper_by_id(paper_id: str) -> Optional[Dict]:
    """
    Get paper details by ID.
    
    Returns:
        Paper metadata with abstract, authors, top chunks, etc.
    """
    collection = get_collection()
    
    try:
        # Get all chunks for this paper
        chunks = collection.get(
            where={"paper_id": paper_id},
            limit=1000
        )
        
        if not chunks.get("ids"):
            return None
        
        # Get metadata from first chunk
        metadata = chunks["metadatas"][0] if chunks.get("metadatas") else {}
        
        # Get top chunks (by index, representing different sections)
        chunk_data = []
        seen_indices = set()
        for i, chunk_id in enumerate(chunks["ids"]):
            chunk_meta = chunks["metadatas"][i] if chunks.get("metadatas") else {}
            chunk_text = chunks["documents"][i] if chunks.get("documents") else ""
            chunk_idx = chunk_meta.get("chunk_index", i)
            
            if chunk_idx not in seen_indices:
                chunk_data.append({
                    "chunk_index": chunk_idx,
                    "text": chunk_text[:500],  # Preview
                    "full_text": chunk_text
                })
                seen_indices.add(chunk_idx)
                if len(chunk_data) >= 10:  # Top 10 chunks
                    break
        
        return {
            "paper_id": paper_id,
            "title": metadata.get("title", "Unknown"),
            "authors": metadata.get("authors", "Unknown"),
            "abstract": metadata.get("abstract", ""),
            "year": metadata.get("year"),
            "pdf_url": metadata.get("pdf_url", ""),
            "source": metadata.get("source", "unknown"),
            "doi": metadata.get("doi", ""),
            "arxiv_id": metadata.get("arxiv_id", ""),
            "keywords": metadata.get("keywords", ""),
            "word_count": metadata.get("word_count", 0),
            "chunk_count": len(chunks["ids"]),
            "top_chunks": chunk_data,
            "ingestion_date": metadata.get("ingestion_date", "")
        }
        
    except Exception as e:
        logger.error(f"Error getting paper: {e}")
        return None


def get_paper_summary(paper_id: str) -> Optional[Dict]:
    """
    Get paper summary (abstract + key chunks).
    
    Returns:
        Summary with abstract and key insights
    """
    paper = get_paper_by_id(paper_id)
    if not paper:
        return None
    
    # Extract key sentences from top chunks
    key_insights = []
    for chunk in paper.get("top_chunks", [])[:5]:
        text = chunk.get("full_text", chunk.get("text", ""))
        # Get first sentence
        first_sentence = text.split('.')[0] if '.' in text else text[:200]
        if first_sentence.strip():
            key_insights.append(first_sentence.strip() + ".")
    
    return {
        "paper_id": paper_id,
        "title": paper["title"],
        "abstract": paper["abstract"],
        "key_insights": key_insights,
        "authors": paper["authors"],
        "year": paper["year"]
    }

