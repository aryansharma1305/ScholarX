"""Evaluation metrics for retrieval and answer quality."""
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Precision@K.
    
    Args:
        retrieved: List of retrieved item IDs (ranked)
        relevant: Set of relevant item IDs
        k: Number of top results to consider
        
    Returns:
        Precision@K score (0-1)
    """
    if k == 0:
        return 0.0
    
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    
    if not retrieved_k:
        return 0.0
    
    relevant_retrieved = sum(1 for item in retrieved_k if item in relevant_set)
    return relevant_retrieved / len(retrieved_k)


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Recall@K.
    
    Args:
        retrieved: List of retrieved item IDs (ranked)
        relevant: Set of relevant item IDs
        k: Number of top results to consider
        
    Returns:
        Recall@K score (0-1)
    """
    if not relevant:
        return 0.0
    
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    
    relevant_retrieved = sum(1 for item in retrieved_k if item in relevant_set)
    return relevant_retrieved / len(relevant)


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved: List of retrieved item IDs (ranked)
        relevant: Set of relevant item IDs
        
    Returns:
        MRR score (0-1)
    """
    if not relevant:
        return 0.0
    
    relevant_set = set(relevant)
    
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / rank
    
    return 0.0


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at K.
    
    Args:
        relevance_scores: List of relevance scores (0-3)
        k: Number of results to consider
        
    Returns:
        DCG@K score
    """
    scores = relevance_scores[:k]
    dcg = 0.0
    
    for i, score in enumerate(scores, start=1):
        dcg += score / math.log2(i + 1)
    
    return dcg


def ndcg_at_k(retrieved: List[str], relevance_map: Dict[str, float], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    Args:
        retrieved: List of retrieved item IDs (ranked)
        relevance_map: Dict mapping item ID to relevance score (0-3)
        k: Number of results to consider
        
    Returns:
        NDCG@K score (0-1)
    """
    retrieved_k = retrieved[:k]
    
    # Get relevance scores for retrieved items
    relevance_scores = [relevance_map.get(item, 0.0) for item in retrieved_k]
    
    # Calculate DCG
    dcg = dcg_at_k(relevance_scores, k)
    
    # Calculate IDCG (ideal DCG)
    ideal_scores = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal_scores, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mean_average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """
    Calculate Mean Average Precision (MAP).
    
    Args:
        retrieved: List of retrieved item IDs (ranked)
        relevant: Set of relevant item IDs
        
    Returns:
        MAP score (0-1)
    """
    if not relevant:
        return 0.0
    
    relevant_set = set(relevant)
    relevant_retrieved = []
    
    for i, item in enumerate(retrieved):
        if item in relevant_set:
            relevant_retrieved.append(i + 1)  # 1-indexed rank
    
    if not relevant_retrieved:
        return 0.0
    
    # Calculate average precision
    precisions = []
    for i, rank in enumerate(relevant_retrieved, start=1):
        precisions.append(i / rank)
    
    return sum(precisions) / len(relevant) if relevant else 0.0


def bleu_score(candidate: str, reference: str, n: int = 4) -> float:
    """
    Calculate BLEU score (simplified version).
    
    Args:
        candidate: Generated answer
        reference: Reference answer
        n: Maximum n-gram order
        
    Returns:
        BLEU score (0-1)
    """
    candidate_tokens = candidate.lower().split()
    reference_tokens = reference.lower().split()
    
    if not candidate_tokens:
        return 0.0
    
    # Calculate precision for each n-gram order
    precisions = []
    
    for i in range(1, n + 1):
        candidate_ngrams = Counter([
            tuple(candidate_tokens[j:j+i])
            for j in range(len(candidate_tokens) - i + 1)
        ])
        reference_ngrams = Counter([
            tuple(reference_tokens[j:j+i])
            for j in range(len(reference_tokens) - i + 1)
        ])
        
        # Count matches
        matches = sum(
            min(candidate_ngrams[ngram], reference_ngrams[ngram])
            for ngram in candidate_ngrams
        )
        total = sum(candidate_ngrams.values())
        
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total)
    
    # Calculate brevity penalty
    if len(candidate_tokens) < len(reference_tokens):
        bp = math.exp(1 - len(reference_tokens) / len(candidate_tokens))
    else:
        bp = 1.0
    
    # Calculate geometric mean
    if any(p == 0 for p in precisions):
        return 0.0
    
    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / n)
    
    return bp * geometric_mean


def rouge_l(candidate: str, reference: str) -> float:
    """
    Calculate ROUGE-L (Longest Common Subsequence).
    
    Args:
        candidate: Generated answer
        reference: Reference answer
        
    Returns:
        ROUGE-L F1 score (0-1)
    """
    candidate_tokens = candidate.lower().split()
    reference_tokens = reference.lower().split()
    
    if not candidate_tokens or not reference_tokens:
        return 0.0
    
    # Calculate LCS length
    lcs_length = _lcs_length(candidate_tokens, reference_tokens)
    
    # Calculate precision, recall, F1
    precision = lcs_length / len(candidate_tokens) if candidate_tokens else 0.0
    recall = lcs_length / len(reference_tokens) if reference_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Calculate length of longest common subsequence."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def rouge_n(candidate: str, reference: str, n: int = 2) -> float:
    """
    Calculate ROUGE-N (n-gram overlap).
    
    Args:
        candidate: Generated answer
        reference: Reference answer
        n: N-gram order (1 or 2)
        
    Returns:
        ROUGE-N recall score (0-1)
    """
    candidate_tokens = candidate.lower().split()
    reference_tokens = reference.lower().split()
    
    if len(candidate_tokens) < n or len(reference_tokens) < n:
        return 0.0
    
    candidate_ngrams = Counter([
        tuple(candidate_tokens[i:i+n])
        for i in range(len(candidate_tokens) - n + 1)
    ])
    reference_ngrams = Counter([
        tuple(reference_tokens[i:i+n])
        for i in range(len(reference_tokens) - n + 1)
    ])
    
    # Count matches
    matches = sum(
        min(candidate_ngrams[ngram], reference_ngrams[ngram])
        for ngram in candidate_ngrams
    )
    total_reference = sum(reference_ngrams.values())
    
    if total_reference == 0:
        return 0.0
    
    return matches / total_reference


def semantic_similarity(candidate: str, reference: str, 
                       embedding_fn) -> float:
    """
    Calculate semantic similarity using embeddings.
    
    Args:
        candidate: Generated answer
        reference: Reference answer
        embedding_fn: Function that takes text and returns embedding
        
    Returns:
        Cosine similarity score (0-1)
    """
    try:
        candidate_emb = embedding_fn(candidate)
        reference_emb = embedding_fn(reference)
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(candidate_emb, reference_emb))
        norm_a = math.sqrt(sum(a * a for a in candidate_emb))
        norm_b = math.sqrt(sum(b * b for b in reference_emb))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    except Exception:
        return 0.0


def citation_accuracy(citations: List[str], correct_citations: List[str]) -> float:
    """
    Calculate citation accuracy.
    
    Args:
        citations: List of cited paper IDs
        correct_citations: List of correct paper IDs
        
    Returns:
        Citation accuracy (0-1)
    """
    if not citations:
        return 0.0
    
    citations_set = set(citations)
    correct_set = set(correct_citations)
    
    correct = sum(1 for c in citations if c in correct_set)
    return correct / len(citations) if citations else 0.0


def calculate_retrieval_metrics(
    retrieved: List[str],
    relevant: List[str],
    relevance_map: Optional[Dict[str, float]] = None,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Calculate all retrieval metrics.
    
    Args:
        retrieved: List of retrieved item IDs (ranked)
        relevant: List of relevant item IDs
        relevance_map: Optional dict mapping item ID to relevance score (0-3)
        k_values: List of K values for Precision@K and Recall@K
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Precision and Recall at K
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
    
    # MRR
    metrics["mrr"] = mean_reciprocal_rank(retrieved, relevant)
    
    # MAP
    metrics["map"] = mean_average_precision(retrieved, relevant)
    
    # NDCG at K
    if relevance_map:
        for k in k_values:
            metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevance_map, k)
    
    return metrics


def calculate_answer_quality_metrics(
    candidate: str,
    reference: str,
    embedding_fn=None
) -> Dict[str, float]:
    """
    Calculate all answer quality metrics.
    
    Args:
        candidate: Generated answer
        reference: Reference answer
        embedding_fn: Optional function for semantic similarity
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # BLEU
    metrics["bleu"] = bleu_score(candidate, reference)
    
    # ROUGE
    metrics["rouge_l"] = rouge_l(candidate, reference)
    metrics["rouge_1"] = rouge_n(candidate, reference, n=1)
    metrics["rouge_2"] = rouge_n(candidate, reference, n=2)
    
    # Semantic similarity
    if embedding_fn:
        metrics["semantic_similarity"] = semantic_similarity(
            candidate, reference, embedding_fn
        )
    
    return metrics

