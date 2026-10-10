"""Causal language-model token surprisal."""

import numpy as np


def token_surprisals(text, tokenizer, model):
    """Negative log p(token_t | previous tokens); first token excluded without BOS."""
    import torch

    batch = tokenizer(text, return_tensors="pt").to(model.device)
    ids = batch["input_ids"]
    if ids.shape[1] < 2:
        return []
    model.eval()
    with torch.inference_mode():
        logits = model(**batch).logits[:, :-1, :]
        logprob = torch.log_softmax(logits, dim=-1)
        values = -logprob.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    tokens = tokenizer.convert_ids_to_tokens(ids[0, 1:].tolist())
    return list(zip(tokens, values[0].cpu().tolist()))


def calibrate_threshold(training_scores, training_labels):
    """Choose a high-surprisal typo threshold only on calibration data."""
    x, y = np.asarray(training_scores), np.asarray(training_labels)
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("Expected nonempty paired scores and labels")
    thresholds = np.unique(np.r_[x.min() - 1e-6, x, x.max() + 1e-6])
    return float(max(thresholds, key=lambda t: ((x >= t) == y).mean()))
