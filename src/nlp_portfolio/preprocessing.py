"""Tokenization and Zipf frequency statistics."""

from collections import Counter
import re
import numpy as np


def simple_tokens(text):
    """Unicode-aware word tokens with case folding; punctuation omitted."""
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.casefold(), flags=re.UNICODE)


def zipf_slope(tokens, min_count=2):
    """Fit log frequency to log rank for terms observed at least min_count times."""
    counts = Counter(tokens)
    values = sorted((v for v in counts.values() if v >= min_count), reverse=True)
    if len(values) < 2:
        raise ValueError("At least two distinct recurring terms are needed")
    ranks = np.arange(1, len(values) + 1, dtype=float)
    slope, intercept = np.polyfit(np.log(ranks), np.log(values), 1)
    return float(slope), float(intercept), ranks, np.asarray(values)
