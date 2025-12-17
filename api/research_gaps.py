"""Research gap identification - find underexplored areas."""
from typing import List, Dict, Optional
from config.chroma_client import get_collection
from vectorstore.query import query_vectors
from processing.embeddings import generate_embedding
from rag.hybrid_search import hybrid_search
from utils.logger import get_logger
from collections import defaultdict

logger = get_logger(__name__)


def identify_research_gaps(topic: str, min_papers: int = 5) -> Dict:
    """
    Identify research gaps in a given topic area.
    
    A research gap is identified when:
    1. A subtopic has very few papers compared to the main topic
    2. Recent papers are declining in a subtopic
    3. A combination of concepts has not been explored
    
    Args:
        topic: Main topic to analyze
        min_papers: Minimum papers needed to consider a subtopic explored
        
    Returns:
        Dictionary with identified gaps
    """
    collection = get_collection()
    
    try:
        # Search for papers on the topic
        query_embedding = generate_embedding(topic)
        results = query_vectors(query_embedding, top_k=100)
        
        if len(results) < min_papers:
            return {
                "topic": topic,
                "status": "understudied",
                "gap_type": "insufficient_research",
                "paper_count": len(results),
                "recommendation": f"Only {len(results)} papers found. This is an underexplored area."
            }
        
        # Analyze subtopics
        subtopics = defaultdict(int)
        subtopic_years = defaultdict(list)
        
        for result in results:
            meta = result.metadata or {}
            title = meta.get("title", "").lower()
            abstract = meta.get("abstract", "").lower()
            year = meta.get("year")
            
            # Common subtopics/keywords
            keywords = [
                "application", "method", "algorithm", "framework", "model", "system",
                "evaluation", "benchmark", "dataset", "comparison", "survey", "review",
                "optimization", "efficiency", "scalability", "robustness", "accuracy"
            ]
            
            for keyword in keywords:
                if keyword in title or keyword in abstract:
                    subtopics[keyword] += 1
                    if year:
                        subtopic_years[keyword].append(year)
        
        # Find gaps (subtopics with few papers)
        gaps = []
        for subtopic, count in subtopics.items():
            if count < min_papers:
                gaps.append({
                    "subtopic": subtopic,
                    "paper_count": count,
                    "gap_type": "underexplored_subtopic",
                    "recommendation": f"'{subtopic}' aspect of '{topic}' needs more research"
                })
        
        # Find declining subtopics
        declining = []
        for subtopic, years in subtopic_years.items():
            if len(years) >= 3:
                recent_years = sorted(years, reverse=True)[:3]
                older_years = sorted(years, reverse=True)[3:6] if len(years) >= 6 else []
                
                if older_years:
                    recent_avg = sum(recent_years) / len(recent_years)
                    older_avg = sum(older_years) / len(older_years)
                    
                    if recent_avg < older_avg - 2:  # Declining by more than 2 years
                        declining.append({
                            "subtopic": subtopic,
                            "trend": "declining",
                            "recent_avg_year": recent_avg,
                            "older_avg_year": older_avg,
                            "gap_type": "declining_research",
                            "recommendation": f"Research on '{subtopic}' in '{topic}' is declining - potential gap"
                        })
        
        # Find unexplored combinations
        # (This is simplified - in practice, would use more sophisticated NLP)
        combination_gaps = []
        main_keywords = topic.lower().split()
        
        # Check if combinations exist
        for result in results[:20]:  # Sample
            meta = result.metadata or {}
            text = f"{meta.get('title', '')} {meta.get('abstract', '')}".lower()
            
            # If topic keywords are sparse in results, it's a gap
            keyword_density = sum(1 for kw in main_keywords if kw in text) / len(main_keywords) if main_keywords else 0
            if keyword_density < 0.3:
                combination_gaps.append({
                    "gap_type": "low_keyword_coverage",
                    "recommendation": f"Papers on '{topic}' may not fully cover all aspects"
                })
                break  # Only report once
        
        return {
            "topic": topic,
            "total_papers": len(results),
            "status": "explored" if len(results) >= min_papers else "understudied",
            "gaps": gaps,
            "declining_subtopics": declining,
            "combination_gaps": combination_gaps[:3],  # Limit to 3
            "recommendations": [
                gap["recommendation"] for gap in gaps[:5]
            ] + [d["recommendation"] for d in declining[:3]]
        }
        
    except Exception as e:
        logger.error(f"Error identifying research gaps: {e}")
        return {"error": str(e)}


def find_underexplored_combinations(concept1: str, concept2: str) -> Dict:
    """
    Find if a combination of two concepts is underexplored.
    
    Args:
        concept1: First concept
        concept2: Second concept
        
    Returns:
        Dictionary with analysis
    """
    # Search for each concept individually
    embedding1 = generate_embedding(concept1)
    embedding2 = generate_embedding(concept2)
    embedding_combined = generate_embedding(f"{concept1} {concept2}")
    
    results1 = query_vectors(embedding1, top_k=50)
    results2 = query_vectors(embedding2, top_k=50)
    results_combined = query_vectors(embedding_combined, top_k=50)
    
    # Count papers mentioning both
    collection = get_collection()
    both_count = 0
    
    for result in results_combined:
        meta = result.metadata or {}
        text = f"{meta.get('title', '')} {meta.get('abstract', '')}".lower()
        if concept1.lower() in text and concept2.lower() in text:
            both_count += 1
    
    concept1_count = len(results1)
    concept2_count = len(results2)
    
    # Calculate overlap ratio
    if concept1_count > 0 and concept2_count > 0:
        expected_overlap = (concept1_count * concept2_count) / 100  # Rough estimate
        overlap_ratio = both_count / expected_overlap if expected_overlap > 0 else 0
    else:
        overlap_ratio = 0
    
    is_gap = overlap_ratio < 0.5  # Less than 50% of expected overlap
    
    return {
        "concept1": concept1,
        "concept2": concept2,
        "concept1_papers": concept1_count,
        "concept2_papers": concept2_count,
        "combined_papers": both_count,
        "overlap_ratio": overlap_ratio,
        "is_research_gap": is_gap,
        "recommendation": f"Combination of '{concept1}' and '{concept2}' is {'underexplored' if is_gap else 'well-explored'}"
    }


def suggest_research_directions(topic: str) -> List[str]:
    """
    Suggest specific research directions based on gaps.
    
    Args:
        topic: Research topic
        
    Returns:
        List of suggested research directions
    """
    gaps = identify_research_gaps(topic)
    
    suggestions = []
    
    if gaps.get("status") == "understudied":
        suggestions.append(f"Conduct foundational research on {topic}")
        suggestions.append(f"Develop initial frameworks/models for {topic}")
    
    for gap in gaps.get("gaps", [])[:3]:
        suggestions.append(gap["recommendation"])
    
    for declining in gaps.get("declining_subtopics", [])[:2]:
        suggestions.append(declining["recommendation"])
    
    if not suggestions:
        suggestions.append(f"Explore novel applications of {topic}")
        suggestions.append(f"Investigate scalability aspects of {topic}")
        suggestions.append(f"Compare different approaches to {topic}")
    
    return suggestions[:5]  # Top 5 suggestions

