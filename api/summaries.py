"""Automatic paper summarization."""
from typing import Dict, List
from config.openai_client import client
from config.settings import settings
from api.paper_api import get_paper_by_id
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_paper_summary(paper_id: str, use_llm: bool = False) -> Dict:
    """
    Generate automatic summaries for a paper.
    
    Args:
        paper_id: Paper ID
        use_llm: Whether to use LLM (requires OpenAI) or extractive method
        
    Returns:
        Dictionary with short, medium, and bullet-point summaries
    """
    paper = get_paper_by_id(paper_id)
    if not paper:
        return {}
    
    abstract = paper.get("abstract", "")
    title = paper.get("title", "")
    top_chunks = paper.get("top_chunks", [])
    
    # Extract key text
    key_text = " ".join([chunk.get("full_text", chunk.get("text", ""))[:500] 
                        for chunk in top_chunks[:5]])
    
    if use_llm and settings.llm_provider == "openai" and settings.openai_api_key:
        # Use LLM for better summaries
        try:
            prompt = f"""Paper Title: {title}

Abstract: {abstract}

Key Content: {key_text[:2000]}

Generate three summaries:
1. Short summary (1-2 sentences)
2. Medium summary (2-3 paragraphs)
3. Bullet-point insights (5-7 key points)

Format as JSON with keys: short, medium, bullets (array)."""

            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "You are a research paper summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            import json
            summary_text = response.choices[0].message.content
            # Try to parse JSON, fallback to extractive if fails
            try:
                summaries = json.loads(summary_text)
                return {
                    "paper_id": paper_id,
                    "short": summaries.get("short", abstract[:200]),
                    "medium": summaries.get("medium", abstract),
                    "bullets": summaries.get("bullets", [])
                }
            except:
                pass
        except Exception as e:
            logger.warning(f"LLM summary failed: {e}, using extractive")
    
    # Extractive method (no LLM needed)
    short = abstract[:200] if abstract else title
    medium = abstract if abstract else key_text[:500]
    
    # Extract bullet points from key chunks
    bullets = []
    for chunk in top_chunks[:5]:
        text = chunk.get("full_text", chunk.get("text", ""))
        first_sentence = text.split('.')[0] if '.' in text else text[:150]
        if first_sentence.strip():
            bullets.append(first_sentence.strip() + ".")
    
    return {
        "paper_id": paper_id,
        "short": short,
        "medium": medium,
        "bullets": bullets[:7]
    }



