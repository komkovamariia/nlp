"""Scaled dot-product attention."""
import numpy as np


def scaled_dot_product_attention(query, key, value, mask=None):
    q, k, v = (np.asarray(x, dtype=float) for x in (query, key, value))
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(k.shape[-1])
    if mask is not None:
        scores = np.where(np.asarray(mask, dtype=bool), scores, -1e9)
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs @ v, probs


def causal_mask(length):
    return np.tril(np.ones((length, length), dtype=bool))
