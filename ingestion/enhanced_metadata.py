"""Enhanced metadata extraction from papers."""
import re
from typing import Dict, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


def extract_enhanced_metadata(text: str) -> Dict:
    """
    Extract enhanced metadata from paper text.
    
    Returns:
        Dictionary with title, authors, abstract, keywords, references, etc.
    """
    metadata = {}
    
    # Extract title (usually first few lines, before abstract)
    title_match = re.search(r'^(.{10,200}?)(?:\n\n|Abstract|INTRODUCTION)', text, re.MULTILINE | re.IGNORECASE)
    if title_match:
        metadata['title'] = title_match.group(1).strip()
    else:
        # Fallback: first line
        first_line = text.split('\n')[0].strip()
        metadata['title'] = first_line[:200] if len(first_line) > 10 else "Untitled Paper"
    
    # Extract abstract
    abstract_patterns = [
        r'Abstract[:\s]*\n(.*?)(?:\n\n|Introduction|1\.|Keywords)',
        r'ABSTRACT[:\s]*\n(.*?)(?:\n\n|INTRODUCTION|1\.|Keywords)',
    ]
    
    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1).strip()
            # Clean up abstract
            abstract = re.sub(r'\s+', ' ', abstract)
            metadata['abstract'] = abstract[:1000]  # Limit length
            break
    
    # Extract keywords
    keywords_patterns = [
        r'Keywords?[:\s]*\n(.*?)(?:\n\n|1\.|Introduction)',
        r'KEYWORDS?[:\s]*\n(.*?)(?:\n\n|1\.|INTRODUCTION)',
    ]
    
    for pattern in keywords_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            keywords_text = match.group(1).strip()
            # Split by comma, semicolon, or newline
            keywords = [k.strip() for k in re.split(r'[,;\n]', keywords_text) if k.strip()]
            metadata['keywords'] = keywords[:10]  # Limit to 10
            break
    
    # Extract year (look for 4-digit years in reasonable range)
    year_match = re.search(r'\b(19|20)\d{2}\b', text[:2000])
    if year_match:
        year = int(year_match.group(0))
        if 1990 <= year <= 2025:
            metadata['year'] = year
    
    # Extract DOI
    doi_match = re.search(r'DOI[:\s]*([0-9.]+/[^\s]+)', text, re.IGNORECASE)
    if doi_match:
        metadata['doi'] = doi_match.group(1)
    
    # Extract arXiv ID
    arxiv_match = re.search(r'arXiv[:\s]*([0-9]+\.[0-9]+v?[0-9]*)', text, re.IGNORECASE)
    if arxiv_match:
        metadata['arxiv_id'] = arxiv_match.group(1)
    
    # Count references (look for reference section)
    ref_pattern = r'(?:References?|Bibliography|REFERENCES?|BIBLIOGRAPHY)'
    ref_match = re.search(ref_pattern, text, re.IGNORECASE)
    if ref_match:
        ref_section = text[ref_match.end():]
        # Count numbered references
        ref_count = len(re.findall(r'^\[?\d+\]', ref_section[:5000], re.MULTILINE))
        metadata['reference_count'] = ref_count
    
    # Estimate paper length
    word_count = len(text.split())
    metadata['word_count'] = word_count
    metadata['estimated_pages'] = word_count // 250  # Rough estimate
    
    return metadata


def extract_authors_enhanced(text: str) -> List[str]:
    """Extract author names with better pattern matching."""
    authors = []
    
    # Look for author patterns after title
    title_end = text.find('\n\n', 0, 500)
    if title_end == -1:
        title_end = 200
    
    author_section = text[title_end:title_end + 500]
    
    # Pattern: Name, Name, and Name
    author_patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:\s*,\s*|and\s+)(?=[A-Z])',
        r'([A-Z]\.[\s-]?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    ]
    
    for pattern in author_patterns:
        matches = re.findall(pattern, author_section)
        if matches:
            authors.extend(matches)
            break
    
    # Clean and deduplicate
    authors = [a.strip() for a in authors if len(a.strip()) > 3]
    authors = list(dict.fromkeys(authors))  # Remove duplicates
    
    return authors[:10]  # Limit to 10 authors



