"""Query expansion and enhancement for better retrieval."""
from typing import List, Set
import re
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def expand_query(query: str) -> List[str]:
    """
    Expand a query with synonyms and related terms.
    
    Args:
        query: Original query
        
    Returns:
        List of expanded query variations
    """
    expanded = [query]  # Always include original
    
    # Extract key terms
    key_terms = extract_key_terms(query)
    
    # Generate variations
    variations = []
    
    # Add plural/singular forms
    for term in key_terms:
        if term.endswith('s'):
            variations.append(term[:-1])  # Remove 's'
        else:
            variations.append(term + 's')
    
    # Add common synonyms (simple approach - can be enhanced with WordNet or LLM)
    synonym_map = {
        "neural network": ["deep learning", "artificial neural network", "ANN"],
        "transformer": ["attention mechanism", "self-attention"],
        "machine learning": ["ML", "statistical learning"],
        "natural language processing": ["NLP", "computational linguistics"],
    }
    
    for term in key_terms:
        term_lower = term.lower()
        if term_lower in synonym_map:
            variations.extend(synonym_map[term_lower])
    
    # Combine variations
    if variations:
        expanded.append(" ".join(variations[:3]))  # Limit to avoid too long queries
    
    logger.info(f"Expanded query '{query}' to {len(expanded)} variations")
    return expanded


def extract_key_terms(query: str) -> List[str]:
    """Extract key terms from a query."""
    # Remove stop words (simple list)
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "what", "how", "why", "when", "where"}
    
    # Split and filter
    words = re.findall(r'\b\w+\b', query.lower())
    key_terms = [w for w in words if w not in stop_words and len(w) > 2]
    
    return key_terms


def expand_query_with_llm(query: str) -> List[str]:
    """
    Use LLM to expand query with related terms and synonyms.
    Falls back to simple expansion if LLM not available.
    
    Args:
        query: Original query
        
    Returns:
        List of expanded queries
    """
    # If using OpenAI and it's available, use it
    if settings.llm_provider == "openai" and settings.openai_api_key:
        try:
            from config.openai_client import client
            prompt = f"""Given this research query: "{query}"

Generate 3-5 related search queries that would help find relevant research papers. 
Include:
- Synonyms
- Related concepts
- Alternative phrasings
- Broader and narrower terms

Return only the queries, one per line, without numbering."""

            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "You are a research assistant that helps expand search queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            expanded_text = response.choices[0].message.content or ""
            expanded_queries = [q.strip() for q in expanded_text.split('\n') if q.strip()]
            
            # Combine with original
            all_queries = [query] + expanded_queries[:4]  # Limit to 4 expansions
            
            logger.info(f"LLM expanded query to {len(all_queries)} variations")
            return all_queries
            
        except Exception as e:
            logger.warning(f"Failed to expand query with LLM: {e}, using simple expansion")
            return expand_query(query)
    else:
        # Use simple expansion (no LLM needed)
        logger.info("Using simple query expansion (no LLM)")
        return expand_query(query)


def normalize_query(query: str) -> str:
    """
    Normalize query for better matching.
    
    - Remove extra whitespace
    - Fix common typos
    - Expand acronyms (optional)
    """
    # Remove extra whitespace
    query = " ".join(query.split())
    
    # Common acronym expansions (can be enhanced)
    acronym_map = {
        "nlp": "natural language processing",
        "ml": "machine learning",
        "dl": "deep learning",
        "ai": "artificial intelligence",
        "cv": "computer vision",
    }
    
    words = query.lower().split()
    expanded = []
    for word in words:
        if word in acronym_map:
            expanded.append(acronym_map[word])
        else:
            expanded.append(word)
    
    return " ".join(expanded)

