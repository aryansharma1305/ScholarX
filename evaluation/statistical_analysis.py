"""Statistical analysis for evaluation results."""
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from utils.logger import get_logger

logger = get_logger(__name__)


def paired_t_test(scores_a: List[float], scores_b: List[float]) -> Dict:
    """
    Perform paired t-test between two systems.
    
    Args:
        scores_a: Scores from system A
        scores_b: Scores from system B (paired with A)
        
    Returns:
        Dictionary with t-statistic, p-value, and effect size
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Scores must be paired (same length)")
    
    differences = [a - b for a, b in zip(scores_a, scores_b)]
    
    # Perform t-test
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    
    # Calculate effect size (Cohen's d)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'mean_difference': float(mean_diff),
        'significant': p_value < 0.05
    }


def wilcoxon_test(scores_a: List[float], scores_b: List[float]) -> Dict:
    """
    Perform Wilcoxon signed-rank test (non-parametric).
    
    Args:
        scores_a: Scores from system A
        scores_b: Scores from system B (paired with A)
        
    Returns:
        Dictionary with statistic, p-value, and effect size
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Scores must be paired (same length)")
    
    statistic, p_value = stats.wilcoxon(scores_a, scores_b)
    
    # Calculate effect size (r = z / sqrt(N))
    n = len(scores_a)
    z = stats.norm.ppf(p_value / 2) if p_value > 0 else 0
    effect_size = abs(z) / np.sqrt(n) if n > 0 else 0.0
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'effect_size': float(effect_size),
        'significant': p_value < 0.05
    }


def one_way_anova(scores_dict: Dict[str, List[float]]) -> Dict:
    """
    Perform one-way ANOVA across multiple systems.
    
    Args:
        scores_dict: Dictionary mapping system name to list of scores
        
    Returns:
        Dictionary with F-statistic, p-value, and post-hoc results
    """
    if len(scores_dict) < 2:
        raise ValueError("Need at least 2 systems for ANOVA")
    
    # Prepare data
    groups = list(scores_dict.values())
    group_names = list(scores_dict.keys())
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Post-hoc: Tukey HSD (if significant)
    post_hoc = {}
    if p_value < 0.05:
        # Simplified post-hoc (would need scipy.stats.tukey_hsd in newer versions)
        # For now, perform pairwise t-tests
        for i, name_a in enumerate(group_names):
            for name_b in group_names[i+1:]:
                t_result = paired_t_test(scores_dict[name_a], scores_dict[name_b])
                post_hoc[f"{name_a}_vs_{name_b}"] = {
                    'p_value': t_result['p_value'],
                    'significant': t_result['significant']
                }
    
    return {
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'post_hoc': post_hoc
    }


def calculate_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for mean.
    
    Args:
        scores: List of scores
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not scores:
        return (0.0, 0.0)
    
    mean = np.mean(scores)
    std_err = stats.sem(scores)
    
    h = std_err * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
    
    return (float(mean - h), float(mean + h))


def calculate_statistics(scores: List[float]) -> Dict:
    """
    Calculate descriptive statistics.
    
    Args:
        scores: List of scores
        
    Returns:
        Dictionary with mean, std, median, min, max, CI
    """
    if not scores:
        return {
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'ci_95': (0.0, 0.0)
        }
    
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1))
    median = float(np.median(scores))
    min_val = float(np.min(scores))
    max_val = float(np.max(scores))
    ci_95 = calculate_confidence_interval(scores, 0.95)
    
    return {
        'mean': mean,
        'std': std,
        'median': median,
        'min': min_val,
        'max': max_val,
        'ci_95': ci_95,
        'n': len(scores)
    }


def compare_systems(
    system_scores: Dict[str, List[float]],
    test_type: str = 'paired_t'
) -> Dict:
    """
    Compare multiple systems statistically.
    
    Args:
        system_scores: Dictionary mapping system name to list of scores
        test_type: Type of test ('paired_t', 'wilcoxon', 'anova')
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    if len(system_scores) < 2:
        return {'error': 'Need at least 2 systems for comparison'}
    
    system_names = list(system_scores.keys())
    
    # Calculate statistics for each system
    for name, scores in system_scores.items():
        results[name] = calculate_statistics(scores)
    
    # Perform comparisons
    if test_type == 'anova' and len(system_scores) > 2:
        anova_result = one_way_anova(system_scores)
        results['anova'] = anova_result
    else:
        # Pairwise comparisons
        comparisons = {}
        for i, name_a in enumerate(system_names):
            for name_b in system_names[i+1:]:
                scores_a = system_scores[name_a]
                scores_b = system_scores[name_b]
                
                if test_type == 'wilcoxon':
                    test_result = wilcoxon_test(scores_a, scores_b)
                else:  # paired_t
                    test_result = paired_t_test(scores_a, scores_b)
                
                comparisons[f"{name_a}_vs_{name_b}"] = test_result
        
        results['comparisons'] = comparisons
    
    return results

