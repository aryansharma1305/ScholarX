"""Advanced chunking strategies for better semantic units."""
from typing import List
from processing.chunker import Chunk, chunk_text
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def chunk_by_sections(text: str, paper_id: str) -> List[Chunk]:
    """
    Chunk text by sections (Introduction, Methods, Results, etc.).
    More semantically meaningful than fixed-size chunks.
    """
    import re
    
    # Common section headers
    section_pattern = r'\n\s*(?:Abstract|Introduction|Background|Related Work|Methods?|Methodology|Results?|Discussion|Conclusion|References?)\s*\n'
    
    sections = re.split(section_pattern, text, flags=re.IGNORECASE)
    
    chunks = []
    chunk_index = 0
    
    for section_text in sections:
        if not section_text.strip():
            continue
        
        # Further chunk large sections
        section_chunks = chunk_text(section_text, paper_id, 
                                   chunk_size=settings.chunk_size,
                                   chunk_overlap=settings.chunk_overlap)
        
        for chunk in section_chunks:
            chunk.index = chunk_index
            chunk_index += 1
            chunks.append(chunk)
    
    logger.info(f"Created {len(chunks)} section-based chunks")
    return chunks


def chunk_by_paragraphs(text: str, paper_id: str, max_chunk_size: int = None) -> List[Chunk]:
    """
    Chunk by paragraphs, combining small paragraphs and splitting large ones.
    """
    max_chunk_size = max_chunk_size or settings.chunk_size
    
    # Split by paragraphs (double newlines)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    chunk_index = 0
    
    for para in paragraphs:
        # If adding this paragraph would exceed size, save current chunk
        if current_chunk and len(current_chunk) + len(para) > max_chunk_size:
            if current_chunk:
                chunks.append(Chunk(
                    text=current_chunk.strip(),
                    index=chunk_index,
                    paper_id=paper_id
                ))
                chunk_index += 1
            current_chunk = para
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        
        # If single paragraph is too large, split it
        if len(current_chunk) > max_chunk_size * 1.5:
            # Split large paragraph
            para_chunks = chunk_text(current_chunk, paper_id, 
                                    chunk_size=max_chunk_size,
                                    chunk_overlap=settings.chunk_overlap)
            chunks.extend(para_chunks)
            chunk_index += len(para_chunks)
            current_chunk = ""
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(Chunk(
            text=current_chunk.strip(),
            index=chunk_index,
            paper_id=paper_id
        ))
    
    logger.info(f"Created {len(chunks)} paragraph-based chunks")
    return chunks


def smart_chunk(text: str, paper_id: str) -> List[Chunk]:
    """
    Smart chunking: tries section-based first, falls back to paragraph-based,
    then to fixed-size if needed.
    """
    # Try section-based first
    section_chunks = chunk_by_sections(text, paper_id)
    
    if len(section_chunks) >= 3:  # If we got good section-based chunks
        return section_chunks
    
    # Fall back to paragraph-based
    para_chunks = chunk_by_paragraphs(text, paper_id)
    
    if len(para_chunks) >= 3:
        return para_chunks
    
    # Final fallback: fixed-size chunking
    return chunk_text(text, paper_id)

