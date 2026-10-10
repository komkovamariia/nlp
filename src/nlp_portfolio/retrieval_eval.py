"""Ranked retrieval metrics."""
import numpy as np


def reciprocal_rank(ranked_ids, relevant_ids):
    relevant = set(relevant_ids)
    return next((1.0 / i for i, key in enumerate(ranked_ids, 1) if key in relevant), 0.0)


def precision_at_k(ranked_ids, relevant_ids, k):
    if k < 1:
        raise ValueError("k must be positive")
    return len(set(ranked_ids[:k]) & set(relevant_ids)) / k


def ndcg_at_k(ranked_ids, relevance, k):
    scores = [float(relevance.get(key, 0.0)) for key in ranked_ids[:k]]
    ideal = sorted([float(v) for v in relevance.values()], reverse=True)[:k]
    discount = lambda vals: sum((2.0**v - 1) / np.log2(i + 2) for i, v in enumerate(vals))
    ideal_score = discount(ideal)
    return float(discount(scores) / ideal_score) if ideal_score else 0.0
