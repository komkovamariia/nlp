"""Preference-optimization objectives for compact demonstrations."""

import numpy as np


def log_sigmoid(x):
    x = np.asarray(x, dtype=float)
    return -np.logaddexp(0.0, -x)


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    """Per-example DPO loss from sequence log probabilities."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    policy_margin = np.asarray(policy_chosen) - np.asarray(policy_rejected)
    reference_margin = np.asarray(reference_chosen) - np.asarray(reference_rejected)
    return -log_sigmoid(beta * (policy_margin - reference_margin))
