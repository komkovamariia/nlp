"""Word-vector similarity, alignment and retrofitting."""
import numpy as np


def cosine(u, v):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.dot(u, v) / denominator) if denominator else 0.0


def align_orthogonal(source, target):
    """Orthogonal Procrustes map fitted on paired anchor word vectors."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("Expected paired matrices of identical shape")
    u, _, vt = np.linalg.svd(source.T @ target, full_matrices=False)
    return u @ vt


def retrofit(vectors, neighbors, iterations=10, anchor_weight=1.0):
    """Jacobi updates anchored at original vectors; symmetric neighbor sets preferred."""
    original = {k: np.asarray(v, dtype=float).copy() for k, v in vectors.items()}
    current = {k: v.copy() for k, v in original.items()}
    for _ in range(iterations):
        updated = {}
        for word, vector in original.items():
            local = [current[n] for n in neighbors.get(word, ()) if n in current and n != word]
            updated[word] = (anchor_weight * vector + sum(local, np.zeros_like(vector))) / (anchor_weight + len(local))
        current = updated
    return current
