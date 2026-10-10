"""Ranked retrieval metrics."""
import numpy as np


def reciprocal_rank(ranked_ids, relevant_ids):
    relevant = set(relevant_ids)
    return next((1.0 / i for i, key in enumerate(ranked_ids, 1) if key in relevant), 0.0)


def precision_at_k(ranked_ids, relevant_ids, k):
    if k < 1:
        raise ValueError("k must be positive")
    return len(set(ranked_ids[:k]) & set(relevant_ids)) / k


def recall_at_k(ranked_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    if k < 1:
        raise ValueError("k must be positive")
    return len(set(ranked_ids[:k]) & relevant) / len(relevant) if relevant else 0.0


def average_precision(ranked_ids, relevant_ids):
    """Average precision with duplicate retrieved identifiers counted once."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    seen = set()
    precisions = []
    hits = 0
    for rank, item in enumerate(ranked_ids, 1):
        if item in seen:
            continue
        seen.add(item)
        if item in relevant:
            hits += 1
            precisions.append(hits / rank)
    return float(sum(precisions) / len(relevant))


def evaluate_run(run, qrels, k=5):
    """Macro-average retrieval metrics over query identifiers shared by qrels."""
    rows = []
    for query_id, relevant in qrels.items():
        ranking = run.get(query_id, [])
        rows.append({
            "query_id": query_id,
            "precision": precision_at_k(ranking, relevant, k),
            "recall": recall_at_k(ranking, relevant, k),
            "rr": reciprocal_rank(ranking, relevant),
            "ap": average_precision(ranking, relevant),
        })
    if not rows:
        raise ValueError("qrels must not be empty")
    return {name: float(np.mean([row[name] for row in rows]))
            for name in ("precision", "recall", "rr", "ap")}


def ndcg_at_k(ranked_ids, relevance, k):
    scores = [float(relevance.get(key, 0.0)) for key in ranked_ids[:k]]
    ideal = sorted([float(v) for v in relevance.values()], reverse=True)[:k]
    discount = lambda vals: sum((2.0**v - 1) / np.log2(i + 2) for i, v in enumerate(vals))
    ideal_score = discount(ideal)
    return float(discount(scores) / ideal_score) if ideal_score else 0.0
