"""Topic clustering and paper organization."""
from typing import List, Dict
from collections import defaultdict
from config.chroma_client import get_collection
from utils.logger import get_logger

logger = get_logger(__name__)


def cluster_papers_by_topic(num_clusters: int = 5) -> Dict:
    """
    Cluster papers by topic using K-Means on embeddings.
    
    Args:
        num_clusters: Number of topic clusters to create
        
    Returns:
        Dictionary with clusters and their papers
    """
    try:
        from sklearn.cluster import KMeans
        import numpy as np
    except ImportError:
        logger.warning("scikit-learn not installed, cannot cluster")
        return {}
    
    collection = get_collection()
    count = collection.count()
    
    if count < num_clusters:
        logger.warning(f"Not enough papers ({count}) for {num_clusters} clusters")
        return {}
    
    # Get all paper embeddings
    all_data = collection.get(limit=count)
    
    # Group by paper_id and get representative embedding
    paper_embeddings = {}
    paper_metadata = {}
    
    for i, paper_id in enumerate(all_data.get("ids", [])):
        metadata = all_data.get("metadatas", [{}])[i] if all_data.get("metadatas") else {}
        pid = metadata.get("paper_id", "unknown")
        
        if pid not in paper_embeddings:
            # Get first chunk's embedding (or generate from title+abstract)
            title = metadata.get("title", "")
            abstract = metadata.get("abstract", "")
            text = f"{title} {abstract[:500]}"
            
            from processing.embeddings import generate_embedding
            embedding = generate_embedding(text)
            paper_embeddings[pid] = embedding
            paper_metadata[pid] = {
                "title": title,
                "authors": metadata.get("authors", ""),
                "year": metadata.get("year")
            }
    
    if len(paper_embeddings) < num_clusters:
        return {}
    
    # Perform K-Means clustering
    embeddings_list = list(paper_embeddings.values())
    paper_ids = list(paper_embeddings.keys())
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_list)
    
    # Organize results
    cluster_papers = defaultdict(list)
    for i, cluster_id in enumerate(clusters):
        cluster_papers[int(cluster_id)].append({
            "paper_id": paper_ids[i],
            **paper_metadata[paper_ids[i]]
        })
    
    # Generate topic labels (simplified - use most common words)
    result = {}
    for cluster_id, papers in cluster_papers.items():
        # Extract keywords from titles
        titles = [p["title"] for p in papers]
        words = []
        for title in titles:
            words.extend(title.lower().split()[:5])
        
        # Get most common words
        from collections import Counter
        common_words = [word for word, count in Counter(words).most_common(3)]
        topic_label = " / ".join(common_words)
        
        result[f"cluster_{cluster_id}"] = {
            "topic": topic_label,
            "paper_count": len(papers),
            "papers": papers
        }
    
    logger.info(f"Created {len(result)} topic clusters")
    return result


def get_paper_topics(paper_id: str, num_clusters: int = 5) -> List[str]:
    """
    Get topic clusters that a paper belongs to.
    
    Returns:
        List of topic labels
    """
    clusters = cluster_papers_by_topic(num_clusters)
    
    topics = []
    for cluster_info in clusters.values():
        paper_ids = [p["paper_id"] for p in cluster_info["papers"]]
        if paper_id in paper_ids:
            topics.append(cluster_info["topic"])
    
    return topics

