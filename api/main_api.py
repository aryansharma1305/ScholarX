"""Main API interface - unified access to all features."""
from typing import Dict, List, Optional
from api.paper_api import get_paper_by_id, get_paper_summary
from api.citations import get_citation_info, find_citing_papers
from api.summaries import generate_paper_summary
from api.authors import get_author_stats, get_author_profile
from api.topics import cluster_papers_by_topic, get_paper_topics
from api.rag_modes import (
    rag_query_concise, rag_query_detailed, rag_query_explain_simple,
    rag_query_compare, rag_query_literature_survey, rag_query_multi_document
)
from api.deduplication import find_duplicate_papers, normalize_arxiv_versions, merge_paper_metadata
from api.similarity import compare_papers, check_text_similarity
from api.query_logger import get_query_stats
from api.ranking import calculate_citation_metrics, get_paper_rank
from api.search import search_papers, compare_papers_side_by_side, get_paper_details_scholar_style
from utils.logger import get_logger

logger = get_logger(__name__)


class ScholarXAPI:
    """Unified API for all ScholarX features."""
    
    # Paper Operations
    @staticmethod
    def get_paper(paper_id: str) -> Optional[Dict]:
        """Get paper details."""
        return get_paper_by_id(paper_id)
    
    @staticmethod
    def get_paper_summary_api(paper_id: str) -> Optional[Dict]:
        """Get paper summary."""
        return get_paper_summary(paper_id)
    
    # Citations
    @staticmethod
    def get_citations(paper_id: str) -> Dict:
        """Get citation information."""
        return get_citation_info(paper_id)
    
    @staticmethod
    def get_related_papers(paper_id: str, limit: int = 10) -> List[Dict]:
        """Get related papers."""
        return find_citing_papers(paper_id, limit)
    
    # Summaries
    @staticmethod
    def generate_summary(paper_id: str, use_llm: bool = False) -> Dict:
        """Generate paper summary."""
        return generate_paper_summary(paper_id, use_llm)
    
    # Authors
    @staticmethod
    def get_author_statistics() -> Dict:
        """Get author statistics."""
        return get_author_stats()
    
    @staticmethod
    def get_author(author_name: str) -> Dict:
        """Get author profile."""
        return get_author_profile(author_name)
    
    # Topics
    @staticmethod
    def cluster_topics(num_clusters: int = 5) -> Dict:
        """Cluster papers by topic."""
        return cluster_papers_by_topic(num_clusters)
    
    @staticmethod
    def get_paper_topics_api(paper_id: str) -> List[str]:
        """Get topics for a paper."""
        return get_paper_topics(paper_id)
    
    # RAG Modes
    @staticmethod
    def rag_concise(query: str) -> Dict:
        """RAG in concise mode."""
        return rag_query_concise(query)
    
    @staticmethod
    def rag_detailed(query: str) -> Dict:
        """RAG in detailed mode."""
        return rag_query_detailed(query)
    
    @staticmethod
    def rag_explain(query: str) -> Dict:
        """RAG in simple explanation mode."""
        return rag_query_explain_simple(query)
    
    @staticmethod
    def rag_compare(query: str) -> Dict:
        """RAG in compare mode."""
        return rag_query_compare(query)
    
    @staticmethod
    def rag_survey(topic: str) -> Dict:
        """Generate literature survey."""
        return rag_query_literature_survey(topic)
    
    @staticmethod
    def rag_multi_document(papers: List[str], query: str) -> Dict:
        """RAG across multiple papers."""
        return rag_query_multi_document(papers, query)
    
    # Deduplication
    @staticmethod
    def find_duplicates(threshold: float = 0.85) -> List[Dict]:
        """Find duplicate papers."""
        return find_duplicate_papers(threshold)
    
    @staticmethod
    def normalize_versions() -> Dict:
        """Normalize ArXiv versions."""
        return normalize_arxiv_versions()
    
    # Similarity
    @staticmethod
    def compare_two_papers(paper_id1: str, paper_id2: str) -> Dict:
        """Compare two papers."""
        return compare_papers(paper_id1, paper_id2)
    
    @staticmethod
    def check_similarity(text: str, threshold: float = 0.8) -> List[Dict]:
        """Check text similarity to papers."""
        return check_text_similarity(text, threshold)
    
    # Analytics
    @staticmethod
    def get_query_statistics() -> Dict:
        """Get query statistics."""
        return get_query_stats()
    
    # Ranking
    @staticmethod
    def get_citation_rankings() -> Dict:
        """Get citation-based rankings."""
        return calculate_citation_metrics()
    
    @staticmethod
    def get_paper_ranking(paper_id: str) -> Dict:
        """Get ranking for a paper."""
        return get_paper_rank(paper_id)
    
    # Search (Google Scholar-like)
    @staticmethod
    def search(
        query: Optional[str] = None,
        author: Optional[str] = None,
        year: Optional[int] = None,
        limit: int = 10
    ) -> Dict:
        """Google Scholar-like search."""
        return search_papers(query=query, author=author, year=year, limit=limit)
    
    @staticmethod
    def compare_papers(paper_ids: List[str]) -> Dict:
        """Compare papers side-by-side."""
        return compare_papers_side_by_side(paper_ids)
    
    @staticmethod
    def get_paper_scholar_style(paper_id: str) -> Dict:
        """Get paper details in Google Scholar format."""
        return get_paper_details_scholar_style(paper_id)


# Convenience instance
api = ScholarXAPI()

