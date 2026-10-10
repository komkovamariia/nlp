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


def split_heads(x, num_heads):
    """Convert ``[tokens, model_dim]`` into ``[heads, tokens, head_dim]``."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] % num_heads:
        raise ValueError("model dimension must be divisible by num_heads")
    return x.reshape(x.shape[0], num_heads, x.shape[1] // num_heads).transpose(1, 0, 2)


def merge_heads(x):
    """Inverse of :func:`split_heads`."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 3:
        raise ValueError("expected [heads, tokens, head_dim]")
    return x.transpose(1, 0, 2).reshape(x.shape[1], -1)


def multi_head_attention(x, wq, wk, wv, wo, num_heads, mask=None):
    """Small NumPy reference implementation of multi-head self-attention."""
    x = np.asarray(x, dtype=float)
    q, k, v = (split_heads(x @ np.asarray(w), num_heads) for w in (wq, wk, wv))
    values = []
    weights = []
    for head in range(num_heads):
        output, probability = scaled_dot_product_attention(q[head], k[head], v[head], mask)
        values.append(output)
        weights.append(probability)
    return merge_heads(np.asarray(values)) @ np.asarray(wo), np.asarray(weights)


def rotary_embedding(x, positions=None):
    """Apply RoPE to the last dimension of a 2-D token matrix."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[-1] % 2:
        raise ValueError("RoPE requires an even feature dimension")
    positions = np.arange(len(x)) if positions is None else np.asarray(positions)
    half = x.shape[-1] // 2
    frequencies = 1.0 / (10000.0 ** (np.arange(half) / half))
    angles = positions[:, None] * frequencies[None, :]
    first, second = x[:, :half], x[:, half:]
    return np.concatenate(
        [
            first * np.cos(angles) - second * np.sin(angles),
            first * np.sin(angles) + second * np.cos(angles),
        ],
        axis=-1,
    )
