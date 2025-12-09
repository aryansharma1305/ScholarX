"""Vector query operations for ChromaDB."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from config.chroma_client import get_collection
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Represents a query result from ChromaDB."""
    chunk_id: str
    paper_id: str
    chunk_index: int
    text: str
    score: float
    metadata: Dict[str, Any]


def query_vectors(
    query_embedding: List[float],
    top_k: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None,
    include_metadata: bool = True
) -> List[QueryResult]:
    """
    Query ChromaDB for similar vectors.
    
    Args:
        query_embedding: Query embedding vector
        top_k: Number of results to return
        filter_metadata: Optional metadata filter (e.g., {"paper_id": "123"})
        include_metadata: Whether to include metadata in results
        
    Returns:
        List of QueryResult objects sorted by score (descending)
    """
    collection = get_collection()
    
    try:
        # Build where clause for filtering
        where = None
        if filter_metadata:
            where = filter_metadata
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        query_results = []
        
        # ChromaDB returns results in a nested structure
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][i]
                text = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                
                # Convert distance to similarity score (ChromaDB uses distance, lower is better)
                # Cosine distance ranges from 0 to 2, so similarity = 1 - (distance / 2)
                score = 1.0 - (distance / 2.0) if distance <= 2.0 else 0.0
                
                query_results.append(QueryResult(
                    chunk_id=chunk_id,
                    paper_id=metadata.get("paper_id", ""),
                    chunk_index=metadata.get("chunk_index", -1),
                    text=text,
                    score=score,
                    metadata=metadata
                ))
        
        # Sort by score (descending) - ChromaDB should already do this, but ensure it
        query_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Retrieved {len(query_results)} results from ChromaDB")
        return query_results
        
    except Exception as e:
        logger.error(f"Failed to query ChromaDB: {e}")
        raise ValueError(f"Failed to query ChromaDB: {e}")


def query_by_paper_id(
    paper_id: str,
    top_k: int = 10
) -> List[QueryResult]:
    """
    Query all chunks for a specific paper.
    
    Args:
        paper_id: Paper ID to filter by
        top_k: Maximum number of results
        
    Returns:
        List of QueryResult objects for the specified paper
    """
    collection = get_collection()
    
    try:
        # Query with metadata filter
        results = collection.get(
            where={"paper_id": str(paper_id)},
            limit=top_k
        )
        
        query_results = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                chunk_id = results["ids"][i]
                text = results["documents"][i] if results["documents"] else ""
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                
                query_results.append(QueryResult(
                    chunk_id=chunk_id,
                    paper_id=metadata.get("paper_id", ""),
                    chunk_index=metadata.get("chunk_index", -1),
                    text=text,
                    score=1.0,  # No distance score for get() queries
                    metadata=metadata
                ))
        
        logger.info(f"Retrieved {len(query_results)} chunks for paper {paper_id}")
        return query_results
        
    except Exception as e:
        logger.error(f"Failed to query by paper_id: {e}")
        raise ValueError(f"Failed to query by paper_id: {e}")
