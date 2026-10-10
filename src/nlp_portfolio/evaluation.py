"""Calibration and uncertainty utilities used by the notebooks."""
import numpy as np


def expected_calibration_error(y_true, probabilities, bins=10):
    """Expected calibration error for binary probabilities."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("expected paired one-dimensional arrays")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        include_right = index == bins - 1
        mask = (p >= edges[index]) & ((p <= edges[index + 1]) if include_right else (p < edges[index + 1]))
        if mask.any():
            result += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(result)


def bootstrap_interval(values, statistic=np.mean, confidence=0.95, samples=2000, seed=42):
    """Percentile bootstrap interval for independent observations."""
    x = np.asarray(values)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("values must be a nonempty vector")
    rng = np.random.default_rng(seed)
    estimates = [statistic(rng.choice(x, size=len(x), replace=True)) for _ in range(samples)]
    alpha = (1.0 - confidence) / 2
    return tuple(float(v) for v in np.quantile(estimates, [alpha, 1 - alpha]))


def paired_bootstrap_delta(first, second, samples=2000, seed=42):
    """Confidence interval for a paired mean difference."""
    a, b = np.asarray(first, float), np.asarray(second, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("expected nonempty paired vectors")
    return bootstrap_interval(b - a, samples=samples, seed=seed)
