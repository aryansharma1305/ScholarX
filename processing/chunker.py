"""Text chunking utilities."""
from dataclasses import dataclass
from typing import List
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    index: int
    paper_id: str
    start_char: int = 0
    end_char: int = 0


def chunk_text(
    text: str,
    paper_id: str,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Chunk]:
    """
    Chunk text into semantically meaningful segments.
    
    Uses character-based chunking with overlap to preserve context.
    Attempts to break at sentence boundaries when possible.
    
    Args:
        text: Input text to chunk
        paper_id: ID of the paper/document
        chunk_size: Maximum characters per chunk (defaults to settings)
        chunk_overlap: Overlap between chunks in characters (defaults to settings)
        
    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []
    
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    chunks: List[Chunk] = []
    start_index = 0
    chunk_index = 0
    
    # Clean text: normalize whitespace
    cleaned_text = text.replace('\r\n', '\n').replace('\r', '\n')
    cleaned_text = ' '.join(cleaned_text.split())
    
    while start_index < len(cleaned_text):
        end_index = min(start_index + chunk_size, len(cleaned_text))
        chunk_text = cleaned_text[start_index:end_index]
        
        # Try to break at sentence boundaries if not at end
        if end_index < len(cleaned_text):
            # Look for sentence endings
            last_period = chunk_text.rfind('. ')
            last_newline = chunk_text.rfind('\n')
            last_exclamation = chunk_text.rfind('! ')
            last_question = chunk_text.rfind('? ')
            
            break_point = max(last_period, last_newline, last_exclamation, last_question)
            
            # Only break if we're past halfway point
            if break_point > chunk_size * 0.5:
                chunk_text = chunk_text[:break_point + 1]
                start_index = start_index + break_point + 1
            else:
                start_index = end_index
        else:
            start_index = end_index
        
        if chunk_text.strip():
            chunks.append(Chunk(
                text=chunk_text.strip(),
                index=chunk_index,
                paper_id=paper_id,
                start_char=start_index - len(chunk_text),
                end_char=start_index
            ))
            chunk_index += 1
        
        # Apply overlap for next chunk
        if start_index < len(cleaned_text):
            start_index = max(0, start_index - chunk_overlap)
    
    logger.info(f"Created {len(chunks)} chunks for paper {paper_id}")
    return chunks


def chunk_by_tokens(
    text: str,
    paper_id: str,
    max_tokens: int = 256,
    overlap_tokens: int = 50
) -> List[Chunk]:
    """
    Chunk text by token count (more accurate for LLM context windows).
    
    Requires tiktoken for token counting.
    
    Args:
        text: Input text to chunk
        paper_id: ID of the paper/document
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks in tokens
        
    Returns:
        List of Chunk objects
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")  # Used by GPT-4
    except ImportError:
        logger.warning("tiktoken not available, falling back to character-based chunking")
        return chunk_text(text, paper_id)
    
    if not text or not text.strip():
        return []
    
    chunks: List[Chunk] = []
    tokens = encoding.encode(text)
    chunk_index = 0
    start_token = 0
    
    while start_token < len(tokens):
        end_token = min(start_token + max_tokens, len(tokens))
        chunk_tokens = tokens[start_token:end_token]
        
        # Decode tokens back to text
        chunk_text = encoding.decode(chunk_tokens)
        
        if chunk_text.strip():
            chunks.append(Chunk(
                text=chunk_text.strip(),
                index=chunk_index,
                paper_id=paper_id,
                start_char=0,  # Token-based doesn't track char positions
                end_char=0
            ))
            chunk_index += 1
        
        # Apply overlap
        if end_token < len(tokens):
            start_token = max(0, end_token - overlap_tokens)
        else:
            start_token = end_token
    
    logger.info(f"Created {len(chunks)} token-based chunks for paper {paper_id}")
    return chunks



