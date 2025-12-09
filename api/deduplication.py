"""Paper deduplication and DOI normalization."""
from typing import List, Dict
from difflib import SequenceMatcher
from config.chroma_client import get_collection
from utils.logger import get_logger

logger = get_logger(__name__)


def similarity_score(str1: str, str2: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def find_duplicate_papers(threshold: float = 0.85) -> List[Dict]:
    """
    Find duplicate papers in the collection.
    
    Args:
        threshold: Similarity threshold (0-1)
        
    Returns:
        List of duplicate groups
    """
    collection = get_collection()
    count = collection.count()
    
    if count == 0:
        return []
    
    all_data = collection.get(limit=count)
    
    # Group by paper_id and extract metadata
    papers = {}
    for i, paper_id in enumerate(all_data.get("ids", [])):
        metadata = all_data.get("metadatas", [{}])[i] if all_data.get("metadatas") else {}
        pid = metadata.get("paper_id", "unknown")
        
        if pid not in papers:
            papers[pid] = {
                "paper_id": pid,
                "title": metadata.get("title", ""),
                "authors": metadata.get("authors", ""),
                "doi": metadata.get("doi", ""),
                "arxiv_id": metadata.get("arxiv_id", ""),
                "year": metadata.get("year")
            }
    
    # Find duplicates
    duplicates = []
    checked = set()
    
    paper_list = list(papers.items())
    for i, (pid1, paper1) in enumerate(paper_list):
        if pid1 in checked:
            continue
        
        group = [paper1]
        
        for j, (pid2, paper2) in enumerate(paper_list[i+1:], i+1):
            if pid2 in checked:
                continue
            
            # Check DOI match
            if paper1["doi"] and paper2["doi"]:
                if paper1["doi"] == paper2["doi"]:
                    group.append(paper2)
                    checked.add(pid2)
                    continue
            
            # Check ArXiv ID match (handle versions)
            if paper1["arxiv_id"] and paper2["arxiv_id"]:
                base1 = paper1["arxiv_id"].split('v')[0]
                base2 = paper2["arxiv_id"].split('v')[0]
                if base1 == base2:
                    group.append(paper2)
                    checked.add(pid2)
                    continue
            
            # Check title similarity
            title_sim = similarity_score(paper1["title"], paper2["title"])
            if title_sim >= threshold:
                # Also check author similarity
                author_sim = similarity_score(paper1["authors"], paper2["authors"])
                if author_sim >= 0.7:  # Authors should also match
                    group.append(paper2)
                    checked.add(pid2)
        
        if len(group) > 1:
            duplicates.append({
                "group_id": f"group_{len(duplicates)}",
                "papers": group,
                "reason": "duplicate_detected"
            })
            checked.add(pid1)
    
    logger.info(f"Found {len(duplicates)} duplicate groups")
    return duplicates


def normalize_arxiv_versions() -> Dict:
    """
    Normalize ArXiv paper versions (v1, v2, etc.) to base ID.
    
    Returns:
        Dictionary mapping base IDs to versions
    """
    collection = get_collection()
    count = collection.count()
    
    if count == 0:
        return {}
    
    all_data = collection.get(limit=count)
    
    arxiv_groups = {}
    for metadata in all_data.get("metadatas", []):
        arxiv_id = metadata.get("arxiv_id", "")
        if arxiv_id:
            base_id = arxiv_id.split('v')[0]  # Remove version
            if base_id not in arxiv_groups:
                arxiv_groups[base_id] = []
            arxiv_groups[base_id].append({
                "paper_id": metadata.get("paper_id"),
                "version": arxiv_id,
                "title": metadata.get("title")
            })
    
    # Filter to only groups with multiple versions
    multiple_versions = {
        base: versions
        for base, versions in arxiv_groups.items()
        if len(versions) > 1
    }
    
    logger.info(f"Found {len(multiple_versions)} ArXiv papers with multiple versions")
    return multiple_versions


def merge_paper_metadata(paper_ids: List[str]) -> Dict:
    """
    Merge metadata from duplicate papers.
    
    Args:
        paper_ids: List of paper IDs to merge
        
    Returns:
        Merged metadata
    """
    collection = get_collection()
    
    all_metadata = {}
    for paper_id in paper_ids:
        chunks = collection.get(
            where={"paper_id": paper_id},
            limit=1
        )
        if chunks.get("metadatas"):
            all_metadata[paper_id] = chunks["metadatas"][0]
    
    if not all_metadata:
        return {}
    
    # Merge: prefer non-empty values, latest year, etc.
    merged = {}
    for key in ["title", "authors", "abstract", "doi", "arxiv_id", "year", "pdf_url"]:
        values = [meta.get(key) for meta in all_metadata.values() if meta.get(key)]
        if values:
            if key == "year":
                merged[key] = max(v for v in values if v)
            elif key == "arxiv_id":
                # Prefer latest version
                merged[key] = max(values, key=lambda x: int(x.split('v')[1]) if 'v' in x else 0)
            else:
                # Prefer longest/non-empty
                merged[key] = max(values, key=len) if values else None
    
    return merged

