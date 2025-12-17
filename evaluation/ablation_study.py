"""Ablation study implementation for ScholarX RAG Pipeline."""
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from evaluation.metrics import calculate_retrieval_metrics, calculate_answer_quality_metrics
from evaluation.datasets import EvaluationDataset, RetrievalDataset, load_evaluation_dataset
from evaluation.statistical_analysis import compare_systems, calculate_statistics
from evaluation.run_evaluation import EvaluationRunner
from main import query_rag
from rag.pipeline import run_rag_pipeline, run_simple_rag_pipeline
from rag.hybrid_search import hybrid_search
from rag.retriever import retrieve_context
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class AblationConfig:
    """Configuration for ablation study."""
    
    def __init__(
        self,
        name: str,
        use_hybrid_search: bool = True,
        use_query_expansion: bool = True,
        use_reranking: bool = True,
        use_on_demand_fetching: bool = True
    ):
        """
        Initialize ablation configuration.
        
        Args:
            name: Name of configuration
            use_hybrid_search: Whether to use hybrid search
            use_query_expansion: Whether to use query expansion
            use_reranking: Whether to use re-ranking
            use_on_demand_fetching: Whether to fetch papers on-demand
        """
        self.name = name
        self.use_hybrid_search = use_hybrid_search
        self.use_query_expansion = use_query_expansion
        self.use_reranking = use_reranking
        self.use_on_demand_fetching = use_on_demand_fetching
    
    def __repr__(self):
        return (
            f"AblationConfig(name={self.name}, "
            f"hybrid={self.use_hybrid_search}, "
            f"expansion={self.use_query_expansion}, "
            f"rerank={self.use_reranking}, "
            f"fetch={self.use_on_demand_fetching})"
        )


# Standard ablation configurations
ABLATION_CONFIGS = [
    AblationConfig(
        name="Full ScholarX",
        use_hybrid_search=True,
        use_query_expansion=True,
        use_reranking=True,
        use_on_demand_fetching=True
    ),
    AblationConfig(
        name="ScholarX - Re-ranking",
        use_hybrid_search=True,
        use_query_expansion=True,
        use_reranking=False,
        use_on_demand_fetching=True
    ),
    AblationConfig(
        name="ScholarX - Hybrid Search",
        use_hybrid_search=False,
        use_query_expansion=True,
        use_reranking=True,
        use_on_demand_fetching=True
    ),
    AblationConfig(
        name="ScholarX - Query Expansion",
        use_hybrid_search=True,
        use_query_expansion=False,
        use_reranking=True,
        use_on_demand_fetching=True
    ),
    AblationConfig(
        name="ScholarX - All Enhancements",
        use_hybrid_search=False,
        use_query_expansion=False,
        use_reranking=False,
        use_on_demand_fetching=True
    ),
]


def run_ablation_retrieval(
    config: AblationConfig,
    query: str,
    top_k: int = 20
) -> List[str]:
    """
    Run retrieval with specific ablation configuration.
    
    Args:
        config: Ablation configuration
        query: User query
        top_k: Number of results to retrieve
        
    Returns:
        List of retrieved paper IDs
    """
    try:
        if config.use_hybrid_search:
            results = hybrid_search(query, top_k=top_k)
        else:
            results = retrieve_context(query, top_k=top_k)
        
        # Note: Re-ranking would be applied here if enabled
        # For now, we return the results as-is
        # In full implementation, you'd apply re-ranking if config.use_reranking
        
        return [r.paper_id for r in results]
    except Exception as e:
        logger.error(f"Error in ablation retrieval for {config.name}: {e}")
        return []


def run_ablation_answer_generation(
    config: AblationConfig,
    query: str,
    top_k: int = 5
) -> str:
    """
    Generate answer with specific ablation configuration.
    
    Args:
        config: Ablation configuration
        query: User query
        top_k: Number of context chunks
        
    Returns:
        Generated answer
    """
    try:
        # Use simple pipeline if all enhancements disabled
        if not config.use_hybrid_search and not config.use_query_expansion and not config.use_reranking:
            response = run_simple_rag_pipeline(query, top_k=top_k)
        else:
            # Use full pipeline with configuration
            response = run_rag_pipeline(
                query=query,
                top_k=top_k,
                fetch_papers=config.use_on_demand_fetching,
                use_hybrid_search=config.use_hybrid_search,
                use_reranking=config.use_reranking
            )
        
        return response.answer
    except Exception as e:
        logger.error(f"Error in ablation answer generation for {config.name}: {e}")
        return ""


class AblationStudy:
    """Ablation study runner."""
    
    def __init__(self, output_dir: Path, configs: List[AblationConfig] = None):
        """
        Initialize ablation study.
        
        Args:
            output_dir: Directory to save results
            configs: List of ablation configurations (default: standard configs)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.configs = configs or ABLATION_CONFIGS
        self.results = {}
    
    def run_retrieval_ablation(
        self,
        dataset: RetrievalDataset,
        metrics: List[str] = ['precision@10', 'recall@10', 'ndcg@10', 'map']
    ) -> Dict:
        """
        Run retrieval ablation study.
        
        Args:
            dataset: Retrieval dataset
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary with results for each configuration
        """
        logger.info("Starting retrieval ablation study")
        
        all_results = {}
        
        for config in self.configs:
            logger.info(f"Evaluating {config.name}")
            config_results = []
            
            for query in dataset:
                try:
                    # Run retrieval
                    retrieved = run_ablation_retrieval(config, query.query, top_k=20)
                    
                    # Calculate metrics
                    query_metrics = calculate_retrieval_metrics(
                        retrieved=retrieved,
                        relevant=query.relevant_papers,
                        relevance_map=query.relevance_map,
                        k_values=[5, 10, 20]
                    )
                    query_metrics['query_id'] = query.query_id
                    config_results.append(query_metrics)
                    
                except Exception as e:
                    logger.error(f"Error evaluating query {query.query_id}: {e}")
                    continue
            
            # Aggregate results
            aggregated = self._aggregate_metrics(config_results)
            all_results[config.name] = {
                'config': {
                    'use_hybrid_search': config.use_hybrid_search,
                    'use_query_expansion': config.use_query_expansion,
                    'use_reranking': config.use_reranking,
                    'use_on_demand_fetching': config.use_on_demand_fetching
                },
                'aggregated': aggregated,
                'per_query': config_results
            }
        
        # Statistical comparison
        comparison = self._compare_configurations(all_results, metrics)
        
        # Save results
        results_file = self.output_dir / "ablation_retrieval.json"
        with open(results_file, 'w') as f:
            json.dump({
                'configurations': all_results,
                'comparison': comparison,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Retrieval ablation study complete. Results saved to {results_file}")
        return all_results
    
    def run_answer_quality_ablation(
        self,
        dataset: EvaluationDataset,
        metrics: List[str] = ['bleu', 'rouge_l', 'semantic_similarity']
    ) -> Dict:
        """
        Run answer quality ablation study.
        
        Args:
            dataset: Evaluation dataset
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary with results for each configuration
        """
        logger.info("Starting answer quality ablation study")
        
        all_results = {}
        
        for config in self.configs:
            logger.info(f"Evaluating {config.name}")
            config_results = []
            
            for query in dataset:
                try:
                    # Generate answer
                    answer = run_ablation_answer_generation(config, query.query, top_k=5)
                    
                    # Calculate metrics
                    from processing.embeddings import generate_embedding
                    query_metrics = calculate_answer_quality_metrics(
                        candidate=answer,
                        reference=query.expected_answer,
                        embedding_fn=generate_embedding
                    )
                    query_metrics['query_id'] = query.query_id
                    config_results.append(query_metrics)
                    
                except Exception as e:
                    logger.error(f"Error evaluating query {query.query_id}: {e}")
                    continue
            
            # Aggregate results
            aggregated = self._aggregate_metrics(config_results)
            all_results[config.name] = {
                'config': {
                    'use_hybrid_search': config.use_hybrid_search,
                    'use_query_expansion': config.use_query_expansion,
                    'use_reranking': config.use_reranking,
                    'use_on_demand_fetching': config.use_on_demand_fetching
                },
                'aggregated': aggregated,
                'per_query': config_results
            }
        
        # Statistical comparison
        comparison = self._compare_configurations(all_results, metrics)
        
        # Save results
        results_file = self.output_dir / "ablation_answer_quality.json"
        with open(results_file, 'w') as f:
            json.dump({
                'configurations': all_results,
                'comparison': comparison,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Answer quality ablation study complete. Results saved to {results_file}")
        return all_results
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics across queries."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        metric_names = set()
        
        for m in metrics_list:
            metric_names.update(k for k in m.keys() if k != 'query_id')
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            if values:
                aggregated[metric_name] = calculate_statistics(values)
        
        return aggregated
    
    def _convert_to_retrieval_dataset(self, eval_dataset: EvaluationDataset) -> RetrievalDataset:
        """Convert evaluation dataset to retrieval dataset format."""
        from evaluation.datasets import RetrievalDataset
        import json
        
        retrieval_queries = []
        for query in eval_dataset:
            retrieval_queries.append({
                'query': query.query,
                'relevant_papers': query.relevant_papers,
                'relevance_map': {pid: 2.0 for pid in query.relevant_papers},  # Default relevance
                'query_id': query.query_id
            })
        
        # Create temporary dataset file
        temp_file = self.output_dir / 'temp_retrieval_dataset.json'
        with open(temp_file, 'w') as f:
            json.dump(retrieval_queries, f)
        
        from evaluation.datasets import load_retrieval_dataset
        return load_retrieval_dataset(temp_file)
    
    def _compare_configurations(
        self,
        all_results: Dict,
        metrics: List[str]
    ) -> Dict:
        """Compare configurations statistically."""
        comparison = {}
        
        for metric_name in metrics:
            scores_dict = {}
            for config_name, results in all_results.items():
                if metric_name in results['aggregated']:
                    # Get per-query scores
                    per_query = results.get('per_query', [])
                    scores = [q[metric_name] for q in per_query if metric_name in q]
                    if scores:
                        scores_dict[config_name] = scores
            
            if len(scores_dict) >= 2:
                comparison[metric_name] = compare_systems(scores_dict)
        
        return comparison
    
    def generate_ablation_table(self, output_file: Optional[Path] = None) -> str:
        """
        Generate LaTeX/Markdown table for ablation study.
        
        Args:
            output_file: Optional file to save table
            
        Returns:
            Table as string
        """
        # Load results
        retrieval_file = self.output_dir / "ablation_retrieval.json"
        answer_file = self.output_dir / "ablation_answer_quality.json"
        
        table = "# Ablation Study Results\n\n"
        table += "## Retrieval Performance\n\n"
        table += "| Configuration | Precision@10 | Recall@10 | NDCG@10 | MAP |\n"
        table += "|---------------|--------------|-----------|---------|-----|\n"
        
        if retrieval_file.exists():
            with open(retrieval_file, 'r') as f:
                data = json.load(f)
            
            for config_name, results in data['configurations'].items():
                agg = results['aggregated']
                p10 = agg.get('precision@10', {}).get('mean', 0)
                r10 = agg.get('recall@10', {}).get('mean', 0)
                ndcg = agg.get('ndcg@10', {}).get('mean', 0)
                map_score = agg.get('map', {}).get('mean', 0)
                
                table += f"| {config_name} | {p10:.3f} | {r10:.3f} | {ndcg:.3f} | {map_score:.3f} |\n"
        
        table += "\n## Answer Quality\n\n"
        table += "| Configuration | BLEU | ROUGE-L | Semantic Similarity |\n"
        table += "|---------------|------|---------|---------------------|\n"
        
        if answer_file.exists():
            with open(answer_file, 'r') as f:
                data = json.load(f)
            
            for config_name, results in data['configurations'].items():
                agg = results['aggregated']
                bleu = agg.get('bleu', {}).get('mean', 0)
                rouge = agg.get('rouge_l', {}).get('mean', 0)
                sem = agg.get('semantic_similarity', {}).get('mean', 0)
                
                table += f"| {config_name} | {bleu:.3f} | {rouge:.3f} | {sem:.3f} |\n"
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(table)
        
        return table


def main():
    """Main entry point for ablation study."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--dataset', type=str, required=True, help='Path to evaluation dataset')
    parser.add_argument('--output', type=str, default='ablation_results', help='Output directory')
    parser.add_argument('--mode', type=str, choices=['retrieval', 'answer', 'all'], default='all')
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_evaluation_dataset(Path(args.dataset))
    
    # Initialize ablation study
    study = AblationStudy(Path(args.output))
    
    # Convert to retrieval dataset
    retrieval_dataset = study._convert_to_retrieval_dataset(dataset)
    
    # Run studies
    if args.mode in ['retrieval', 'all']:
        study.run_retrieval_ablation(retrieval_dataset)
    
    if args.mode in ['answer', 'all']:
        study.run_answer_quality_ablation(dataset)
    
    # Generate table
    study.generate_ablation_table(study.output_dir / "ablation_table.md")
    
    print(f"\n✅ Ablation study complete! Results in {args.output}")


if __name__ == '__main__':
    main()

