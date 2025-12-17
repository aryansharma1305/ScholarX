"""Dataset loading and preparation for evaluation."""
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationQuery:
    """Single evaluation query with ground truth."""
    query: str
    expected_answer: str
    relevant_papers: List[str]
    relevant_chunks: List[str]
    domain: str
    difficulty: str  # 'easy', 'medium', 'hard'
    query_id: Optional[str] = None


@dataclass
class RetrievalQuery:
    """Query for retrieval evaluation."""
    query: str
    relevant_papers: List[str]
    relevance_map: Dict[str, float]  # paper_id -> relevance score (0-3)
    query_id: Optional[str] = None


class EvaluationDataset:
    """Evaluation dataset loader."""
    
    def __init__(self, dataset_path: Path):
        """
        Initialize dataset.
        
        Args:
            dataset_path: Path to dataset JSON file
        """
        self.dataset_path = Path(dataset_path)
        self.queries: List[EvaluationQuery] = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from JSON file."""
        if not self.dataset_path.exists():
            logger.warning(f"Dataset not found: {self.dataset_path}")
            return
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            query = EvaluationQuery(
                query=item['query'],
                expected_answer=item.get('expected_answer', ''),
                relevant_papers=item.get('relevant_papers', []),
                relevant_chunks=item.get('relevant_chunks', []),
                domain=item.get('domain', 'unknown'),
                difficulty=item.get('difficulty', 'medium'),
                query_id=item.get('query_id', None)
            )
            self.queries.append(query)
        
        logger.info(f"Loaded {len(self.queries)} evaluation queries")
    
    def get_queries_by_domain(self, domain: str) -> List[EvaluationQuery]:
        """Get queries filtered by domain."""
        return [q for q in self.queries if q.domain == domain]
    
    def get_queries_by_difficulty(self, difficulty: str) -> List[EvaluationQuery]:
        """Get queries filtered by difficulty."""
        return [q for q in self.queries if q.difficulty == difficulty]
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return self.queries[idx]


class RetrievalDataset:
    """Retrieval evaluation dataset."""
    
    def __init__(self, dataset_path: Path):
        """
        Initialize retrieval dataset.
        
        Args:
            dataset_path: Path to dataset JSON file
        """
        self.dataset_path = Path(dataset_path)
        self.queries: List[RetrievalQuery] = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from JSON file."""
        if not self.dataset_path.exists():
            logger.warning(f"Dataset not found: {self.dataset_path}")
            return
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            query = RetrievalQuery(
                query=item['query'],
                relevant_papers=item.get('relevant_papers', []),
                relevance_map=item.get('relevance_map', {}),
                query_id=item.get('query_id', None)
            )
            self.queries.append(query)
        
        logger.info(f"Loaded {len(self.queries)} retrieval queries")
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return self.queries[idx]


def create_sample_dataset(output_path: Path, num_queries: int = 20):
    """
    Create a sample evaluation dataset.
    
    Args:
        output_path: Path to save dataset JSON
        num_queries: Number of sample queries to create
    """
    sample_queries = [
        {
            "query": "What is the transformer architecture?",
            "expected_answer": "The transformer architecture is a neural network architecture based on self-attention mechanisms...",
            "relevant_papers": ["paper_1", "paper_2"],
            "relevant_chunks": ["chunk_1", "chunk_2"],
            "domain": "NLP",
            "difficulty": "easy",
            "query_id": "q1"
        },
        {
            "query": "How does attention mechanism work in neural networks?",
            "expected_answer": "Attention mechanisms allow models to focus on relevant parts of the input...",
            "relevant_papers": ["paper_2", "paper_3"],
            "relevant_chunks": ["chunk_3", "chunk_4"],
            "domain": "NLP",
            "difficulty": "medium",
            "query_id": "q2"
        },
        # Add more sample queries...
    ]
    
    # Extend to num_queries
    while len(sample_queries) < num_queries:
        sample_queries.append({
            "query": f"Sample query {len(sample_queries) + 1}",
            "expected_answer": f"Sample answer {len(sample_queries) + 1}",
            "relevant_papers": [],
            "relevant_chunks": [],
            "domain": "ML",
            "difficulty": "medium",
            "query_id": f"q{len(sample_queries) + 1}"
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_queries[:num_queries], f, indent=2)
    
    logger.info(f"Created sample dataset with {num_queries} queries at {output_path}")


def load_evaluation_dataset(dataset_path: Path) -> EvaluationDataset:
    """
    Load evaluation dataset from file.
    
    Args:
        dataset_path: Path to dataset JSON file
        
    Returns:
        EvaluationDataset object
    """
    return EvaluationDataset(dataset_path)


def load_retrieval_dataset(dataset_path: Path) -> RetrievalDataset:
    """
    Load retrieval dataset from file.
    
    Args:
        dataset_path: Path to dataset JSON file
        
    Returns:
        RetrievalDataset object
    """
    return RetrievalDataset(dataset_path)

