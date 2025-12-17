"""Main evaluation runner script."""
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import argparse

from evaluation.metrics import (
    calculate_retrieval_metrics,
    calculate_answer_quality_metrics
)
from evaluation.baselines import run_baseline_system, compare_baselines
from evaluation.datasets import (
    EvaluationDataset,
    RetrievalDataset,
    load_evaluation_dataset,
    load_retrieval_dataset
)
from evaluation.statistical_analysis import compare_systems, calculate_statistics
from main import query_rag
from rag.pipeline import run_rag_pipeline
from processing.embeddings import generate_embedding
from utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationRunner:
    """Main evaluation runner."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize evaluation runner.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def evaluate_retrieval(
        self,
        dataset: RetrievalDataset,
        system_name: str = 'scholarx',
        top_k: int = 20
    ) -> Dict:
        """
        Evaluate retrieval performance.
        
        Args:
            dataset: Retrieval dataset
            system_name: Name of system to evaluate
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating retrieval for {system_name}")
        
        all_metrics = []
        
        for query in dataset:
            try:
                # Run retrieval
                if system_name == 'scholarx':
                    # Use full ScholarX pipeline
                    from rag.hybrid_search import hybrid_search
                    results = hybrid_search(query.query, top_k=top_k)
                    retrieved = [r.paper_id for r in results]
                else:
                    # Use baseline
                    baseline_result = run_baseline_system(system_name, query.query, top_k)
                    retrieved = baseline_result['retrieved']
                
                # Calculate metrics
                metrics = calculate_retrieval_metrics(
                    retrieved=retrieved,
                    relevant=query.relevant_papers,
                    relevance_map=query.relevance_map,
                    k_values=[5, 10, 20]
                )
                metrics['query_id'] = query.query_id
                all_metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Error evaluating query {query.query_id}: {e}")
                continue
        
        # Aggregate results
        aggregated = self._aggregate_metrics(all_metrics)
        
        # Save results
        results_file = self.output_dir / f"retrieval_{system_name}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'system': system_name,
                'aggregated': aggregated,
                'per_query': all_metrics
            }, f, indent=2)
        
        logger.info(f"Retrieval evaluation complete. Results saved to {results_file}")
        return aggregated
    
    def evaluate_answer_quality(
        self,
        dataset: EvaluationDataset,
        system_name: str = 'scholarx',
        top_k: int = 5
    ) -> Dict:
        """
        Evaluate answer quality.
        
        Args:
            dataset: Evaluation dataset
            system_name: Name of system to evaluate
            top_k: Number of context chunks to use
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating answer quality for {system_name}")
        
        all_metrics = []
        
        for query in dataset:
            try:
                # Generate answer
                if system_name == 'scholarx':
                    response = query_rag(
                        query.query,
                        top_k=top_k,
                        use_enhanced=True
                    )
                    answer = response['answer']
                else:
                    # Use baseline
                    baseline_result = run_baseline_system(system_name, query.query, top_k)
                    answer = baseline_result['answer']
                
                # Calculate metrics
                metrics = calculate_answer_quality_metrics(
                    candidate=answer,
                    reference=query.expected_answer,
                    embedding_fn=generate_embedding
                )
                metrics['query_id'] = query.query_id
                all_metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Error evaluating query {query.query_id}: {e}")
                continue
        
        # Aggregate results
        aggregated = self._aggregate_metrics(all_metrics)
        
        # Save results
        results_file = self.output_dir / f"answer_quality_{system_name}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'system': system_name,
                'aggregated': aggregated,
                'per_query': all_metrics
            }, f, indent=2)
        
        logger.info(f"Answer quality evaluation complete. Results saved to {results_file}")
        return aggregated
    
    def compare_all_systems(
        self,
        dataset: EvaluationDataset,
        systems: List[str] = None
    ) -> Dict:
        """
        Compare all systems on the same dataset.
        
        Args:
            dataset: Evaluation dataset
            systems: List of system names to compare
            
        Returns:
            Dictionary with comparison results
        """
        if systems is None:
            systems = ['scholarx', 'simple_semantic', 'basic_rag', 'hybrid_search']
        
        logger.info(f"Comparing systems: {systems}")
        
        # Evaluate each system
        system_results = {}
        
        # Convert evaluation dataset to retrieval dataset format
        retrieval_dataset = self._convert_to_retrieval_dataset(dataset)
        
        for system in systems:
            try:
                retrieval_metrics = self.evaluate_retrieval(
                    retrieval_dataset,
                    system_name=system
                )
                answer_metrics = self.evaluate_answer_quality(
                    dataset,
                    system_name=system
                )
                
                system_results[system] = {
                    'retrieval': retrieval_metrics,
                    'answer_quality': answer_metrics
                }
            except Exception as e:
                logger.error(f"Error evaluating system {system}: {e}")
                continue
        
        # Statistical comparison
        comparison_results = {}
        
        # Compare retrieval metrics
        for metric_name in ['precision@10', 'recall@10', 'ndcg@10', 'map']:
            scores_dict = {}
            for system, results in system_results.items():
                if metric_name in results['retrieval']:
                    scores_dict[system] = [
                        q[metric_name] for q in results.get('per_query', [])
                    ]
            
            if len(scores_dict) >= 2:
                comparison_results[f'retrieval_{metric_name}'] = compare_systems(scores_dict)
        
        # Compare answer quality metrics
        for metric_name in ['bleu', 'rouge_l', 'semantic_similarity']:
            scores_dict = {}
            for system, results in system_results.items():
                if metric_name in results['answer_quality']:
                    scores_dict[system] = [
                        q[metric_name] for q in results.get('per_query', [])
                    ]
            
            if len(scores_dict) >= 2:
                comparison_results[f'answer_{metric_name}'] = compare_systems(scores_dict)
        
        # Save comparison
        comparison_file = self.output_dir / "system_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump({
                'systems': system_results,
                'statistical_comparison': comparison_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"System comparison complete. Results saved to {comparison_file}")
        return comparison_results
    
    def _convert_to_retrieval_dataset(self, eval_dataset: EvaluationDataset) -> RetrievalDataset:
        """Convert evaluation dataset to retrieval dataset format."""
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
        
        return load_retrieval_dataset(temp_file)
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics across queries."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Get all metric names
        metric_names = set()
        for m in metrics_list:
            metric_names.update(k for k in m.keys() if k != 'query_id')
        
        # Calculate statistics for each metric
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            if values:
                aggregated[metric_name] = calculate_statistics(values)
        
        return aggregated
    
    def generate_report(self, output_file: Optional[Path] = None):
        """
        Generate evaluation report.
        
        Args:
            output_file: Path to save report (default: evaluation_report.md)
        """
        if output_file is None:
            output_file = self.output_dir / "evaluation_report.md"
        
        # Load all results
        results_files = list(self.output_dir.glob("*.json"))
        
        report = f"""# Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

This report contains evaluation results for the ScholarX RAG Pipeline.

## Results Files

"""
        
        for results_file in results_files:
            report += f"- `{results_file.name}`\n"
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report generated: {output_file}")


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description='Run evaluation for ScholarX RAG Pipeline')
    parser.add_argument('--dataset', type=str, required=True, help='Path to evaluation dataset')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--mode', type=str, choices=['retrieval', 'answer', 'compare', 'all'],
                       default='all', help='Evaluation mode')
    parser.add_argument('--systems', nargs='+', help='Systems to evaluate')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = EvaluationRunner(args.output)
    
    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    dataset = load_evaluation_dataset(dataset_path)
    
    # Run evaluation
    if args.mode in ['retrieval', 'all']:
        runner.evaluate_retrieval(dataset, system_name='scholarx')
    
    if args.mode in ['answer', 'all']:
        runner.evaluate_answer_quality(dataset, system_name='scholarx')
    
    if args.mode in ['compare', 'all']:
        runner.compare_all_systems(dataset, systems=args.systems)
    
    # Generate report
    runner.generate_report()


if __name__ == '__main__':
    main()

