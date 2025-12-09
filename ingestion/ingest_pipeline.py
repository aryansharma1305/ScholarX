"""Paper ingestion pipeline functions with enhanced features."""
import uuid
from typing import Optional
from ingestion.pdf_loader import load_pdf_from_url, extract_pdf_metadata
from ingestion.enhanced_metadata import extract_enhanced_metadata, extract_authors_enhanced
from processing.chunker import chunk_text
from processing.advanced_chunker import smart_chunk
from processing.embeddings import generate_embeddings_batch
from vectorstore.upsert import upsert_chunks
from config.settings import settings
from utils.logger import get_logger
from utils.timers import timer

logger = get_logger(__name__)


def ingest_pdf_from_url(
    pdf_url: str,
    paper_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    use_smart_chunking: bool = True
) -> str:
    """
    Ingest a PDF from URL into the RAG pipeline with enhanced features.
    
    Args:
        pdf_url: URL to the PDF file
        paper_id: Optional paper ID (generated if not provided)
        metadata: Optional additional metadata
        use_smart_chunking: Use smart chunking (section/paragraph-based) instead of fixed-size
        
    Returns:
        Generated paper ID
    """
    if paper_id is None:
        paper_id = str(uuid.uuid4())
    
    logger.info(f"Ingesting PDF from URL: {pdf_url}")
    
    with timer("PDF Ingestion"):
        # Load and extract text
        text = load_pdf_from_url(pdf_url)
        
        # Extract enhanced metadata
        enhanced_meta = extract_enhanced_metadata(text)
        
        # Extract authors if not in metadata
        if not metadata or not metadata.get("authors"):
            authors = extract_authors_enhanced(text)
            if authors:
                enhanced_meta["authors"] = ", ".join(authors)
        
        # Merge with provided metadata
        if metadata:
            enhanced_meta.update(metadata)
        
        # Smart chunking or fixed-size
        if use_smart_chunking:
            try:
                chunks = smart_chunk(text, paper_id=paper_id)
            except Exception as e:
                logger.warning(f"Smart chunking failed, using fixed-size: {e}")
                chunks = chunk_text(text, paper_id=paper_id)
        else:
            chunks = chunk_text(text, paper_id=paper_id)
        
        if not chunks:
            raise ValueError("No chunks generated from PDF")
        
        # Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = generate_embeddings_batch(chunk_texts)
        
        # Prepare metadata for storage
        from datetime import datetime
        upsert_metadata = {
            "title": enhanced_meta.get("title", "Untitled"),
            "abstract": enhanced_meta.get("abstract", ""),
            "authors": enhanced_meta.get("authors", enhanced_meta.get("authors_string", "Unknown")),
            "year": enhanced_meta.get("year"),
            "keywords": ", ".join(enhanced_meta.get("keywords", [])),
            "doi": enhanced_meta.get("doi", ""),
            "arxiv_id": enhanced_meta.get("arxiv_id", ""),
            "word_count": enhanced_meta.get("word_count", 0),
            "source": enhanced_meta.get("source", "pdf_url"),
            "pdf_url": pdf_url,
            "ingestion_date": datetime.now().isoformat(),
        }
        
        # Add any additional metadata
        if metadata:
            for key, value in metadata.items():
                if key not in upsert_metadata and isinstance(value, (str, int, float, bool)):
                    upsert_metadata[key] = value
        
        # Upsert to ChromaDB
        upsert_chunks(chunks, embeddings, metadata=upsert_metadata)
    
    logger.info(f"Successfully ingested PDF with {len(chunks)} chunks")
    return paper_id
