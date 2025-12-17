"""Research trend analysis - topic popularity over time."""
from typing import List, Dict, Optional
from config.chroma_client import get_collection
from datetime import datetime
from collections import defaultdict
from utils.logger import get_logger

logger = get_logger(__name__)


def analyze_topic_trends(years: Optional[List[int]] = None) -> Dict:
    """
    Analyze how topics/fields have changed in popularity over time.
    
    Args:
        years: List of years to analyze (if None, uses all available years)
        
    Returns:
        Dictionary with trend data
    """
    collection = get_collection()
    
    try:
        # Get all papers
        all_data = collection.get(limit=10000)
        metadatas = all_data.get("metadatas", [])
        
        # Group papers by year
        papers_by_year = defaultdict(list)
        for meta in metadatas:
            year = meta.get("year")
            if year:
                papers_by_year[year].append(meta)
        
        # Extract topics from titles/abstracts (simple keyword extraction)
        year_topics = {}
        for year, papers in papers_by_year.items():
            if years and year not in years:
                continue
            
            topics = defaultdict(int)
            for paper in papers:
                title = paper.get("title", "").lower()
                abstract = paper.get("abstract", "").lower()
                text = f"{title} {abstract}"
                
                # Common research keywords
                keywords = [
                    "transformer", "attention", "neural", "deep learning", "machine learning",
                    "reinforcement learning", "nlp", "computer vision", "cnn", "rnn", "lstm",
                    "gan", "bert", "gpt", "llm", "diffusion", "adversarial", "optimization",
                    "gradient", "backpropagation", "embedding", "vector", "semantic"
                ]
                
                for keyword in keywords:
                    if keyword in text:
                        topics[keyword] += 1
            
            year_topics[year] = dict(topics)
        
        # Calculate trends (growth/decline)
        trends = {}
        years_sorted = sorted(year_topics.keys())
        
        for i, year in enumerate(years_sorted):
            if i == 0:
                continue
            
            prev_year = years_sorted[i - 1]
            trends[year] = {}
            
            all_topics = set(list(year_topics[year].keys()) + list(year_topics[prev_year].keys()))
            
            for topic in all_topics:
                current_count = year_topics[year].get(topic, 0)
                prev_count = year_topics[prev_year].get(topic, 0)
                
                if prev_count > 0:
                    growth_rate = ((current_count - prev_count) / prev_count) * 100
                else:
                    growth_rate = 100 if current_count > 0 else 0
                
                trends[year][topic] = {
                    "count": current_count,
                    "growth_rate": growth_rate,
                    "trend": "rising" if growth_rate > 10 else "declining" if growth_rate < -10 else "stable"
                }
        
        # Find hottest topics (most growth in recent years)
        recent_years = sorted(years_sorted, reverse=True)[:3]
        topic_growth = defaultdict(float)
        
        for year in recent_years:
            if year in trends:
                for topic, data in trends[year].items():
                    topic_growth[topic] += data["growth_rate"]
        
        hottest_topics = sorted(topic_growth.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "year_topics": year_topics,
            "trends": trends,
            "hottest_topics": [{"topic": t, "total_growth": g} for t, g in hottest_topics],
            "years_analyzed": years_sorted,
            "total_papers": len(metadatas)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        return {"error": str(e)}


def get_field_popularity(field: str) -> Dict:
    """
    Get popularity trend for a specific field/topic.
    
    Args:
        field: Field name (e.g., "transformer", "nlp")
        
    Returns:
        Dictionary with popularity over time
    """
    collection = get_collection()
    
    try:
        all_data = collection.get(limit=10000)
        metadatas = all_data.get("metadatas", [])
        
        # Count papers by year for this field
        field_by_year = defaultdict(int)
        for meta in metadatas:
            year = meta.get("year")
            if not year:
                continue
            
            title = meta.get("title", "").lower()
            abstract = meta.get("abstract", "").lower()
            text = f"{title} {abstract}"
            
            if field.lower() in text:
                field_by_year[year] += 1
        
        years = sorted(field_by_year.keys())
        counts = [field_by_year[y] for y in years]
        
        # Calculate growth rates
        growth_rates = []
        for i in range(1, len(counts)):
            if counts[i-1] > 0:
                growth = ((counts[i] - counts[i-1]) / counts[i-1]) * 100
            else:
                growth = 100 if counts[i] > 0 else 0
            growth_rates.append(growth)
        
        return {
            "field": field,
            "years": years,
            "counts": counts,
            "growth_rates": growth_rates,
            "total_papers": sum(counts),
            "peak_year": years[counts.index(max(counts))] if counts else None,
            "recent_trend": "rising" if growth_rates and growth_rates[-1] > 0 else "declining" if growth_rates else "unknown"
        }
        
    except Exception as e:
        logger.error(f"Error getting field popularity: {e}")
        return {"error": str(e)}


def predict_future_trends(field: str, years_ahead: int = 3) -> Dict:
    """
    Predict future trends for a field using simple linear regression.
    
    Args:
        field: Field name
        years_ahead: Number of years to predict ahead
        
    Returns:
        Dictionary with predictions
    """
    popularity = get_field_popularity(field)
    
    if "error" in popularity or not popularity.get("years"):
        return {"error": "Insufficient data for prediction"}
    
    years = popularity["years"]
    counts = popularity["counts"]
    
    if len(years) < 3:
        return {"error": "Need at least 3 years of data"}
    
    # Simple linear regression
    n = len(years)
    sum_x = sum(years)
    sum_y = sum(counts)
    sum_xy = sum(years[i] * counts[i] for i in range(n))
    sum_x2 = sum(y * y for y in years)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    # Predict future
    current_year = max(years)
    predictions = []
    
    for i in range(1, years_ahead + 1):
        future_year = current_year + i
        predicted_count = slope * future_year + intercept
        predictions.append({
            "year": future_year,
            "predicted_count": max(0, int(predicted_count)),
            "confidence": "high" if len(years) >= 5 else "medium" if len(years) >= 3 else "low"
        })
    
    return {
        "field": field,
        "current_trend": popularity.get("recent_trend", "unknown"),
        "slope": slope,
        "predictions": predictions,
        "method": "linear_regression"
    }

