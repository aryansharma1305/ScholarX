"""Query logging for analytics."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from utils.logger import get_logger

logger = get_logger(__name__)

LOG_FILE = Path("query_logs.jsonl")


def log_query(
    query: str,
    answer: str,
    papers_used: List[str],
    chunks_retrieved: int,
    model_used: str = "simple",
    time_taken: float = 0.0,
    mode: str = "default"
) -> None:
    """
    Log a query for analytics.
    
    Args:
        query: User query
        answer: Generated answer
        papers_used: List of paper IDs used
        chunks_retrieved: Number of chunks retrieved
        model_used: Model/provider used
        time_taken: Time in seconds
        mode: RAG mode used
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer_length": len(answer),
        "papers_used": papers_used,
        "papers_count": len(papers_used),
        "chunks_retrieved": chunks_retrieved,
        "model_used": model_used,
        "mode": mode,
        "time_taken": time_taken
    }
    
    # Append to JSONL file
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    logger.debug(f"Logged query: {query[:50]}...")


def get_query_stats() -> Dict:
    """
    Get statistics from query logs.
    
    Returns:
        Dictionary with query statistics
    """
    if not LOG_FILE.exists():
        return {
            "total_queries": 0,
            "avg_time": 0,
            "most_common_queries": [],
            "papers_usage": {}
        }
    
    queries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    if not queries:
        return {"total_queries": 0}
    
    # Calculate stats
    total = len(queries)
    avg_time = sum(q.get("time_taken", 0) for q in queries) / total
    
    # Most common queries
    from collections import Counter
    query_texts = [q["query"] for q in queries]
    common_queries = Counter(query_texts).most_common(10)
    
    # Paper usage
    paper_usage = Counter()
    for q in queries:
        for pid in q.get("papers_used", []):
            paper_usage[pid] += 1
    
    return {
        "total_queries": total,
        "avg_time_seconds": avg_time,
        "most_common_queries": [{"query": q, "count": c} for q, c in common_queries],
        "top_papers_used": [
            {"paper_id": pid, "usage_count": count}
            for pid, count in paper_usage.most_common(10)
        ]
    }

