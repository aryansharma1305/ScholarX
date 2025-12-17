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
from api.recommendations import (
    recommend_papers_based_on_history, recommend_papers_for_query,
    record_paper_view, record_query, get_trending_topics
)
from api.trends import analyze_topic_trends, get_field_popularity, predict_future_trends
from api.research_gaps import (
    identify_research_gaps, find_underexplored_combinations, suggest_research_directions
)
from api.exports import (
    export_to_bibtex, export_to_csv, export_to_json, export_to_markdown, export_rag_session
)
from api.query_intent import classify_query_intent, route_query_by_intent
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
    
    # Recommendations (NEW)
    @staticmethod
    def recommend_papers(limit: int = 10) -> List[Dict]:
        """Recommend papers based on user history."""
        return recommend_papers_based_on_history(limit)
    
    @staticmethod
    def recommend_for_query(query: str, limit: int = 10) -> List[Dict]:
        """Recommend papers for a specific query."""
        return recommend_papers_for_query(query, limit)
    
    @staticmethod
    def record_view(paper_id: str, paper_title: str = ""):
        """Record that a user viewed a paper."""
        record_paper_view(paper_id, paper_title)
    
    @staticmethod
    def record_user_query(query: str):
        """Record a user query for recommendations."""
        record_query(query)
    
    @staticmethod
    def get_trending_topics_api(days: int = 30) -> List[Dict]:
        """Get trending topics."""
        return get_trending_topics(days)
    
    # Trends (NEW)
    @staticmethod
    def analyze_trends(years: Optional[List[int]] = None) -> Dict:
        """Analyze topic trends over time."""
        return analyze_topic_trends(years)
    
    @staticmethod
    def get_field_trends(field: str) -> Dict:
        """Get popularity trend for a field."""
        return get_field_popularity(field)
    
    @staticmethod
    def predict_trends(field: str, years_ahead: int = 3) -> Dict:
        """Predict future trends for a field."""
        return predict_future_trends(field, years_ahead)
    
    # Research Gaps (NEW)
    @staticmethod
    def find_gaps(topic: str, min_papers: int = 5) -> Dict:
        """Identify research gaps in a topic."""
        return identify_research_gaps(topic, min_papers)
    
    @staticmethod
    def find_combination_gaps(concept1: str, concept2: str) -> Dict:
        """Find if a combination of concepts is underexplored."""
        return find_underexplored_combinations(concept1, concept2)
    
    @staticmethod
    def suggest_directions(topic: str) -> List[str]:
        """Suggest research directions based on gaps."""
        return suggest_research_directions(topic)
    
    # Exports (NEW)
    @staticmethod
    def export_bibtex(paper_ids: Optional[List[str]] = None, filename: str = "papers.bib") -> str:
        """Export papers to BibTeX."""
        return export_to_bibtex(paper_ids, filename)
    
    @staticmethod
    def export_csv(paper_ids: Optional[List[str]] = None, filename: str = "papers.csv") -> str:
        """Export papers to CSV."""
        return export_to_csv(paper_ids, filename)
    
    @staticmethod
    def export_json(paper_ids: Optional[List[str]] = None, filename: str = "papers.json") -> str:
        """Export papers to JSON."""
        return export_to_json(paper_ids, filename)
    
    @staticmethod
    def export_markdown(paper_ids: Optional[List[str]] = None, filename: str = "papers.md") -> str:
        """Export papers to Markdown."""
        return export_to_markdown(paper_ids, filename)
    
    @staticmethod
    def export_rag(query: str, answer: str, citations: List[Dict], filename: str = "rag_session.md") -> str:
        """Export a RAG session."""
        return export_rag_session(query, answer, citations, filename)
    
    # Query Intent (NEW)
    @staticmethod
    def classify_intent(query: str) -> Dict:
        """Classify query intent."""
        return classify_query_intent(query)
    
    @staticmethod
    def route_query(query: str) -> Dict:
        """Route query to appropriate handler."""
        return route_query_by_intent(query)


# Convenience instance
api = ScholarXAPI()

