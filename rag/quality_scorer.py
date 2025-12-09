"""Paper quality scoring for better ranking."""
from typing import Dict, Optional
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)


def calculate_quality_score(paper_metadata: Dict) -> float:
    """
    Calculate a quality score for a paper.
    
    Args:
        paper_metadata: Paper metadata dictionary
        
    Returns:
        Quality score between 0 and 1
    """
    score = 0.0
    
    # Citation count (if available) - normalized
    citation_count = paper_metadata.get("citation_count", 0)
    if citation_count > 0:
        # Log scale: log(1 + citations) / log(1000)
        import math
        citation_score = min(1.0, math.log(1 + citation_count) / math.log(1000))
        score += citation_score * 0.3  # 30% weight
    
    # Recency (newer papers get higher score)
    year = paper_metadata.get("year")
    if year:
        current_year = datetime.utcnow().year
        age = current_year - year
        if age <= 1:
            recency_score = 1.0
        elif age <= 5:
            recency_score = 0.8
        elif age <= 10:
            recency_score = 0.6
        else:
            recency_score = max(0.3, 1.0 - (age / 50))
        score += recency_score * 0.2  # 20% weight
    
    # Open access / PDF available
    if paper_metadata.get("pdf_url"):
        score += 0.15  # 15% weight
    
    # Source quality
    source = paper_metadata.get("source", "")
    if source == "arxiv":
        score += 0.1  # ArXiv is reputable
    elif source == "semantic_scholar":
        score += 0.1
    
    # Abstract quality (length as proxy)
    abstract = paper_metadata.get("abstract", "")
    if abstract and len(abstract) > 200:
        score += 0.1  # 10% weight
    
    # Author count (more authors might indicate collaboration)
    authors = paper_metadata.get("authors", [])
    if len(authors) >= 2:
        score += 0.05  # 5% weight
    
    # Title quality (not too short, not too long)
    title = paper_metadata.get("title", "")
    title_len = len(title.split())
    if 5 <= title_len <= 20:
        score += 0.05  # 5% weight
    
    # Normalize to 0-1
    return min(1.0, score)


def enhance_paper_metadata(paper: Dict) -> Dict:
    """
    Enhance paper metadata with quality scores and additional info.
    
    Args:
        paper: Paper dictionary
        
    Returns:
        Enhanced paper dictionary
    """
    enhanced = paper.copy()
    
    # Calculate quality score
    enhanced["quality_score"] = calculate_quality_score(paper)
    
    # Add citation count if not present (would need API call)
    # For now, we'll set it to 0 if not available
    if "citation_count" not in enhanced:
        enhanced["citation_count"] = 0
    
    return enhanced

