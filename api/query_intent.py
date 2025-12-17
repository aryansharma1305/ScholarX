"""Query intent classification and smart routing."""
from typing import Dict, List, Optional
from utils.logger import get_logger
import re

logger = get_logger(__name__)


class QueryIntent:
    """Query intent types."""
    FACTUAL = "factual"  # "What is X?"
    COMPARISON = "comparison"  # "Compare X and Y"
    HOW_TO = "how_to"  # "How to do X?"
    EXPLANATION = "explanation"  # "Explain X"
    LITERATURE_SURVEY = "literature_survey"  # "Survey of X"
    RECOMMENDATION = "recommendation"  # "Recommend papers on X"
    TREND_ANALYSIS = "trend_analysis"  # "Trends in X"
    RESEARCH_GAP = "research_gap"  # "Gaps in X"
    UNKNOWN = "unknown"


def classify_query_intent(query: str) -> Dict:
    """
    Classify the intent of a user query.
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with intent classification and confidence
    """
    query_lower = query.lower()
    
    # Pattern matching for intent classification
    patterns = {
        QueryIntent.FACTUAL: [
            r"what is",
            r"what are",
            r"define",
            r"definition of",
            r"meaning of"
        ],
        QueryIntent.COMPARISON: [
            r"compare",
            r"difference between",
            r"vs\.",
            r"versus",
            r"contrast"
        ],
        QueryIntent.HOW_TO: [
            r"how to",
            r"how do",
            r"how can",
            r"steps to",
            r"method to"
        ],
        QueryIntent.EXPLANATION: [
            r"explain",
            r"why",
            r"how does",
            r"how works",
            r"describe"
        ],
        QueryIntent.LITERATURE_SURVEY: [
            r"survey",
            r"review",
            r"overview of",
            r"state of the art",
            r"literature on"
        ],
        QueryIntent.RECOMMENDATION: [
            r"recommend",
            r"suggest",
            r"find papers",
            r"papers about",
            r"related to"
        ],
        QueryIntent.TREND_ANALYSIS: [
            r"trend",
            r"popular",
            r"recent developments",
            r"evolution of",
            r"changes in"
        ],
        QueryIntent.RESEARCH_GAP: [
            r"gap",
            r"missing",
            r"underexplored",
            r"future research",
            r"open problem"
        ]
    }
    
    intent_scores = {}
    
    for intent, pattern_list in patterns.items():
        score = 0
        for pattern in pattern_list:
            matches = len(re.findall(pattern, query_lower))
            score += matches
        intent_scores[intent] = score
    
    # Determine primary intent
    max_score = max(intent_scores.values())
    primary_intent = QueryIntent.UNKNOWN
    
    if max_score > 0:
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        confidence = min(1.0, max_score / 3.0)  # Normalize confidence
    else:
        # Default to factual if no pattern matches
        primary_intent = QueryIntent.FACTUAL
        confidence = 0.3
    
    # Extract entities/topics
    entities = extract_entities(query)
    
    return {
        "intent": primary_intent,
        "confidence": confidence,
        "all_scores": intent_scores,
        "entities": entities,
        "query": query
    }


def extract_entities(query: str) -> List[str]:
    """
    Extract key entities/topics from query.
    
    Args:
        query: User query
        
    Returns:
        List of extracted entities
    """
    # Remove common stop words and question words
    stop_words = {
        "what", "is", "are", "the", "a", "an", "how", "why", "when", "where",
        "which", "who", "do", "does", "can", "could", "should", "would",
        "compare", "explain", "describe", "define", "about", "on", "in", "to"
    }
    
    words = query.lower().split()
    entities = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique_entities.append(e)
    
    return unique_entities[:5]  # Top 5 entities


def route_query_by_intent(query: str, intent_classification: Optional[Dict] = None) -> Dict:
    """
    Route query to appropriate handler based on intent.
    
    Args:
        query: User query
        intent_classification: Pre-computed intent (if available)
        
    Returns:
        Dictionary with routing information
    """
    if intent_classification is None:
        intent_classification = classify_query_intent(query)
    
    intent = intent_classification["intent"]
    
    routing = {
        "intent": intent,
        "recommended_mode": None,
        "recommended_api": None,
        "suggestions": []
    }
    
    # Map intent to recommended RAG mode or API
    if intent == QueryIntent.COMPARISON:
        routing["recommended_mode"] = "compare"
        routing["recommended_api"] = "rag_compare"
        routing["suggestions"].append("Use comparison mode for detailed side-by-side analysis")
    
    elif intent == QueryIntent.EXPLANATION:
        routing["recommended_mode"] = "explain"
        routing["recommended_api"] = "rag_explain"
        routing["suggestions"].append("Use explain mode for simplified explanations")
    
    elif intent == QueryIntent.LITERATURE_SURVEY:
        routing["recommended_mode"] = "survey"
        routing["recommended_api"] = "rag_survey"
        routing["suggestions"].append("Use survey mode for comprehensive literature reviews")
    
    elif intent == QueryIntent.RECOMMENDATION:
        routing["recommended_api"] = "recommend_papers"
        routing["suggestions"].append("Use recommendation API to find relevant papers")
    
    elif intent == QueryIntent.TREND_ANALYSIS:
        routing["recommended_api"] = "analyze_trends"
        routing["suggestions"].append("Use trend analysis to see how topics evolved over time")
    
    elif intent == QueryIntent.RESEARCH_GAP:
        routing["recommended_api"] = "identify_gaps"
        routing["suggestions"].append("Use research gap identification to find underexplored areas")
    
    elif intent == QueryIntent.FACTUAL or intent == QueryIntent.HOW_TO:
        routing["recommended_mode"] = "detailed"
        routing["recommended_api"] = "rag_detailed"
        routing["suggestions"].append("Use detailed mode for comprehensive answers")
    
    else:
        routing["recommended_mode"] = "concise"
        routing["recommended_api"] = "rag_concise"
        routing["suggestions"].append("Use concise mode for quick answers")
    
    return routing

