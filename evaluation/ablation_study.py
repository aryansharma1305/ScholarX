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
        use_on_demand_fetching: bool = True,
        # Citation boost axes
        citation_boost_weight: float = 0.0,
        min_citation_anchor_hits: int = 1,
        use_2hop_citation: bool = False,
    ):
        self.name = name
        self.use_hybrid_search = use_hybrid_search
        self.use_query_expansion = use_query_expansion
        self.use_reranking = use_reranking
        self.use_on_demand_fetching = use_on_demand_fetching
        self.citation_boost_weight = citation_boost_weight
        self.min_citation_anchor_hits = min_citation_anchor_hits
        self.use_2hop_citation = use_2hop_citation
    
    def __repr__(self):
        return (
            f"AblationConfig(name={self.name}, "
            f"hybrid={self.use_hybrid_search}, "
            f"expansion={self.use_query_expansion}, "
            f"rerank={self.use_reranking}, "
            f"boost={self.citation_boost_weight}, "
            f"min_hits={self.min_citation_anchor_hits}, "
            f"2hop={self.use_2hop_citation})"
        )


# Standard ablation configurations (component removal study)
ABLATION_CONFIGS = [
    AblationConfig(
        name="Full ScholarX",
        use_hybrid_search=True, use_query_expansion=True,
        use_reranking=True, use_on_demand_fetching=True,
        citation_boost_weight=0.15,
    ),
    AblationConfig(
        name="ScholarX − Re-ranking",
        use_hybrid_search=True, use_query_expansion=True,
        use_reranking=False, use_on_demand_fetching=True,
        citation_boost_weight=0.15,
    ),
    AblationConfig(
        name="ScholarX − Hybrid Search",
        use_hybrid_search=False, use_query_expansion=True,
        use_reranking=True, use_on_demand_fetching=True,
        citation_boost_weight=0.15,
    ),
    AblationConfig(
        name="ScholarX − Query Expansion",
        use_hybrid_search=True, use_query_expansion=False,
        use_reranking=True, use_on_demand_fetching=True,
        citation_boost_weight=0.15,
    ),
    AblationConfig(
        name="Semantic Only (baseline)",
        use_hybrid_search=False, use_query_expansion=False,
        use_reranking=False, use_on_demand_fetching=False,
        citation_boost_weight=0.0,
    ),
]

# Citation-boost ablation sweep (7 rows, 3 independent axes)
# Run this after ABLATION_CONFIGS to get the full comparison table.
CITATION_ABLATION_CONFIGS = [
    # name,                        hybrid rerank  boost  min_hits  2hop
    AblationConfig("semantic_only",
        use_hybrid_search=False, use_reranking=False,
        citation_boost_weight=0.00, min_citation_anchor_hits=1, use_2hop_citation=False),
    AblationConfig("+ hybrid",
        use_hybrid_search=True,  use_reranking=False,
        citation_boost_weight=0.00, min_citation_anchor_hits=1, use_2hop_citation=False),
    AblationConfig("+ reranker",
        use_hybrid_search=True,  use_reranking=True,
        citation_boost_weight=0.00, min_citation_anchor_hits=1, use_2hop_citation=False),
    AblationConfig("+ citation (w=0.10)",
        use_hybrid_search=True,  use_reranking=True,
        citation_boost_weight=0.10, min_citation_anchor_hits=1, use_2hop_citation=False),
    AblationConfig("+ citation (w=0.15)",
        use_hybrid_search=True,  use_reranking=True,
        citation_boost_weight=0.15, min_citation_anchor_hits=1, use_2hop_citation=False),
    AblationConfig("+ citation (w=0.30)",
        use_hybrid_search=True,  use_reranking=True,
        citation_boost_weight=0.30, min_citation_anchor_hits=1, use_2hop_citation=False),
    AblationConfig("+ citation (min_hits=2)",
        use_hybrid_search=True,  use_reranking=True,
        citation_boost_weight=0.15, min_citation_anchor_hits=2, use_2hop_citation=False),
    AblationConfig("+ citation (2-hop)",
        use_hybrid_search=True,  use_reranking=True,
        citation_boost_weight=0.15, min_citation_anchor_hits=1, use_2hop_citation=True),
]


def run_ablation_retrieval(
    config: AblationConfig,
    query: str,
    top_k: int = 20
) -> List[str]:
    """
    Run retrieval with a specific ablation configuration.
    Temporarily patches settings with the config's citation params so the
    citation_graph_retriever reads the right values without forking the pipeline.
    """
    import contextlib

    @contextlib.contextmanager
    def _patch_settings(cfg: AblationConfig):
        """Temporarily override citation settings for this config."""
        old_weight = settings.citation_boost_weight
        old_min   = settings.min_citation_anchor_hits
        old_2hop  = settings.use_2hop_citation
        settings.citation_boost_weight   = cfg.citation_boost_weight
        settings.min_citation_anchor_hits = cfg.min_citation_anchor_hits
        settings.use_2hop_citation        = cfg.use_2hop_citation
        try:
            yield
        finally:
            settings.citation_boost_weight   = old_weight
            settings.min_citation_anchor_hits = old_min
            settings.use_2hop_citation        = old_2hop

    try:
        with _patch_settings(config):
            use_boost = config.citation_boost_weight > 0
            if config.use_hybrid_search:
                results = hybrid_search(query, top_k=top_k)
            else:
                results = retrieve_context(query, top_k=top_k)

            if config.use_reranking:
                from rag.reranker import rerank_results, ensure_diversity
                paper_ids = list(set(r.paper_id for r in results))
                results = rerank_results(results, {pid: {} for pid in paper_ids})
                results = ensure_diversity(results, max_per_paper=2)

            if use_boost:
                from rag.citation_graph_retriever import apply_citation_boost
                results = apply_citation_boost(results)

        return [r.paper_id for r in results]
    except Exception as e:
        logger.error(f"Error in ablation retrieval for {config.name}: {e}")
        return []


def run_ablation_answer_generation(
    config: AblationConfig,
    query: str,
    top_k: int = 5
) -> str:
    """Generate answer with a specific ablation configuration."""
    try:
        use_boost = config.citation_boost_weight > 0
        response = run_rag_pipeline(
            query=query,
            top_k=top_k,
            fetch_papers=config.use_on_demand_fetching,
            use_hybrid_search=config.use_hybrid_search,
            use_reranking=config.use_reranking,
            use_citation_boost=use_boost,
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


def run_citation_boost_sweep(
    queries_with_relevance: List[Dict],
    output_dir: Optional[Path] = None,
    top_k: int = 5,
) -> str:
    """
    Run the citation-boost ablation sweep across CITATION_ABLATION_CONFIGS.

    This is the primary function to call to produce the ablation table for
    the paper.  It iterates over the 8 citation configs, evaluates each on
    the provided queries, and returns a markdown table.

    Args:
        queries_with_relevance: List of dicts with keys:
            - "query": str
            - "query_id": str
            - "relevant_papers": List[str]  (ground-truth paper IDs)
            - "relevance_map": Dict[str, float]  (paper_id -> score, optional)
        output_dir: If provided, saves results JSON + markdown table here.
        top_k: Precision@K cutoff.

    Returns:
        Markdown table string ready to paste into README or paper.

    Example usage::

        from evaluation.ablation_study import run_citation_boost_sweep
        queries = [
            {
                "query_id": "q1",
                "query": "attention mechanisms in transformers",
                "relevant_papers": ["paper_a", "paper_b"],
                "relevance_map": {"paper_a": 2.0, "paper_b": 1.0},
            },
            # ... more queries
        ]
        table = run_citation_boost_sweep(queries, output_dir=Path("ablation_results"))
        print(table)
    """
    logger.info(
        "Starting citation boost sweep: %d configs × %d queries",
        len(CITATION_ABLATION_CONFIGS), len(queries_with_relevance),
    )

    all_results: Dict[str, Dict] = {}

    for config in CITATION_ABLATION_CONFIGS:
        logger.info("  → Config: %s", config.name)
        per_query_metrics = []

        for q in queries_with_relevance:
            try:
                retrieved = run_ablation_retrieval(config, q["query"], top_k=top_k * 4)
                relevant  = q.get("relevant_papers", [])
                rel_map   = q.get("relevance_map", {})

                qm = calculate_retrieval_metrics(
                    retrieved=retrieved,
                    relevant=relevant,
                    relevance_map=rel_map or None,
                    k_values=[top_k],
                )
                qm["query_id"] = q["query_id"]
                per_query_metrics.append(qm)
            except Exception as exc:
                logger.warning("  Query %s failed for %s: %s", q["query_id"], config.name, exc)

        # Aggregate
        def _mean(key: str) -> float:
            vals = [m[key] for m in per_query_metrics if key in m]
            return sum(vals) / len(vals) if vals else 0.0

        all_results[config.name] = {
            "config": {
                "citation_boost_weight":   config.citation_boost_weight,
                "min_citation_anchor_hits": config.min_citation_anchor_hits,
                "use_2hop_citation":       config.use_2hop_citation,
                "use_hybrid_search":       config.use_hybrid_search,
                "use_reranking":           config.use_reranking,
            },
            "metrics": {
                f"precision@{top_k}": _mean(f"precision@{top_k}"),
                f"recall@{top_k}":    _mean(f"recall@{top_k}"),
                f"ndcg@{top_k}":      _mean(f"ndcg@{top_k}"),
                "mrr":                _mean("mrr"),
                "map":                _mean("map"),
            },
            "per_query": per_query_metrics,
        }

    # ── Build markdown table ─────────────────────────────────────────────────
    k = top_k
    header = (
        f"| System | Hybrid | Rerank | CitBoost | MinHits | 2-Hop "
        f"| P@{k} | R@{k} | NDCG@{k} | MRR | MAP |\n"
        f"|--------|--------|--------|----------|---------|-------"
        f"|-------|-------|---------|-----|-----|\n"
    )
    rows = []
    for name, data in all_results.items():
        cfg = data["config"]
        m   = data["metrics"]
        rows.append(
            f"| **{name}** "
            f"| {'✓' if cfg['use_hybrid_search'] else '✗'} "
            f"| {'✓' if cfg['use_reranking'] else '✗'} "
            f"| {cfg['citation_boost_weight']:.2f} "
            f"| {cfg['min_citation_anchor_hits']} "
            f"| {'✓' if cfg['use_2hop_citation'] else '✗'} "
            f"| {m[f'precision@{k}']:.3f} "
            f"| {m[f'recall@{k}']:.3f} "
            f"| {m[f'ndcg@{k}']:.3f} "
            f"| {m['mrr']:.3f} "
            f"| {m['map']:.3f} |"
        )

    table = (
        f"## Citation-Graph Retrieval Ablation Study\n\n"
        f"Results over {len(queries_with_relevance)} queries · Precision@{k} cutoff\n\n"
        + header
        + "\n".join(rows)
        + "\n"
    )

    # ── Persist ──────────────────────────────────────────────────────────────
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results_file = out / "citation_ablation_results.json"
        table_file   = out / "citation_ablation_table.md"
        with open(results_file, "w") as f:
            json.dump({
                "results": all_results,
                "timestamp": datetime.now().isoformat(),
                "num_queries": len(queries_with_relevance),
                "top_k": top_k,
            }, f, indent=2)
        with open(table_file, "w") as f:
            f.write(table)
        logger.info("Citation ablation results saved to %s", output_dir)

    return table


if __name__ == '__main__':
    main()


