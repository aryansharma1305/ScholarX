"""Vector storage operations for ChromaDB."""
from typing import List, Dict, Any
from config.chroma_client import get_collection
from processing.chunker import Chunk
from utils.logger import get_logger

logger = get_logger(__name__)


def upsert_chunks(
    chunks: List[Chunk],
    embeddings: List[List[float]],
    metadata: Dict[str, Any] = None
) -> None:
    """
    Upsert chunks with their embeddings into ChromaDB.
    
    Args:
        chunks: List of Chunk objects
        embeddings: List of embedding vectors (one per chunk)
        metadata: Additional metadata to include with each chunk
    """
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks must match number of embeddings")
    
    collection = get_collection()
    
    ids = []
    documents = []
    metadatas = []
    
    for chunk, embedding in zip(chunks, embeddings):
        # Build chunk ID
        chunk_id = f"{chunk.paper_id}_chunk_{chunk.index}"
        
        # Build metadata
        chunk_metadata = {
            "paper_id": str(chunk.paper_id),
            "chunk_index": chunk.index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }
        
        # Add any additional metadata
        if metadata:
            for key, value in metadata.items():
                # ChromaDB metadata must be strings, numbers, or bools
                if isinstance(value, (str, int, float, bool)):
                    chunk_metadata[key] = value
                else:
                    chunk_metadata[key] = str(value)
        
        ids.append(chunk_id)
        documents.append(chunk.text)  # ChromaDB stores text separately
        metadatas.append(chunk_metadata)
    
    try:
        # ChromaDB can handle embeddings directly or compute them
        # We'll add embeddings as embeddings parameter
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Successfully upserted {len(ids)} chunks to ChromaDB")
    except Exception as e:
        logger.error(f"Failed to upsert chunks to ChromaDB: {e}")
        raise ValueError(f"Failed to upsert vectors to ChromaDB: {e}")


def upsert_single_chunk(
    chunk: Chunk,
    embedding: List[float],
    metadata: Dict[str, Any] = None
) -> None:
    """
    Upsert a single chunk with its embedding.
    
    Args:
        chunk: Chunk object
        embedding: Embedding vector
        metadata: Additional metadata
    """
    upsert_chunks([chunk], [embedding], metadata)
