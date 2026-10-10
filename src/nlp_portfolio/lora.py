"""Small numerical helpers for low-rank adaptation."""
import numpy as np


def lora_parameter_count(in_features, out_features, rank):
    if min(in_features, out_features, rank) <= 0:
        raise ValueError("dimensions and rank must be positive")
    return rank * (in_features + out_features)


def lora_linear(x, weight, a, b, alpha=1.0):
    """Frozen linear projection plus a scaled low-rank update."""
    x, weight, a, b = (np.asarray(value, dtype=float) for value in (x, weight, a, b))
    if a.shape[0] != weight.shape[1] or b.shape[1] != weight.shape[0] or a.shape[1] != b.shape[0]:
        raise ValueError("incompatible LoRA matrices")
    scale = alpha / a.shape[1]
    return x @ weight.T + scale * (x @ a @ b)


def model_memory_gib(parameter_count, bits_per_parameter, optimizer_multiplier=0.0):
    """Approximate parameter and optimizer-state memory in GiB."""
    if parameter_count < 0 or bits_per_parameter <= 0 or optimizer_multiplier < 0:
        raise ValueError("invalid memory estimate arguments")
    parameter_bytes = parameter_count * bits_per_parameter / 8
    return float(parameter_bytes * (1 + optimizer_multiplier) / 1024**3)
