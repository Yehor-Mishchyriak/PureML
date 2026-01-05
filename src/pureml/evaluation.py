"""Evaluation helpers for PureML models.

Currently provides top-1 accuracy over a Dataset via DataLoader batching, respecting model
train/eval mode toggles, using `no_grad`, and handling logits/probs (argmax) or class-index
outputs as well as one-hot targets."""
from __future__ import annotations

from .base import BaseModel
from .training_utils import Dataset, DataLoader
from .machinery import no_grad, Tensor

def accuracy(mdl: BaseModel, test_set: Dataset, batch_size: int = 32) -> float:
    """
    Top-1 accuracy over `test_set`.

    - If model outputs logits/probs (B, C), we use argmax over the last axis.
    - If model outputs class indices (B,), we use them directly.
    - If targets Y are one-hot (B, C), we argmax them; if indices (B,), we use them.
    """
    # remember & set eval mode (so Dropout/BN behave)
    prev_mode = getattr(mdl, "training", None)
    if hasattr(mdl, "eval"):
        mdl.eval()

    total = 0
    correct = 0

    def _to_indices(t: Tensor) -> Tensor:
        # (B,) -> indices already; (B, C) -> argmax over classes
        return t if t.data.ndim == 1 else t.argmax(axis=-1)

    with no_grad():
        for X, Y in DataLoader(test_set, batch_size=batch_size):
            Y_hat = mdl(X)
            pred_idx = _to_indices(Y_hat)
            true_idx = _to_indices(Y)
            mask = pred_idx.eq(true_idx)
            batch = mask.data.shape[0] if mask.data.ndim >= 1 else 1
            total += batch
            correct += int(mask.data.sum())

    # restore previous mode
    if prev_mode is not None:
        if prev_mode and hasattr(mdl, "train"):
            mdl.train()
        elif (prev_mode is False) and hasattr(mdl, "eval"):
            mdl.eval()

    return correct / max(total, 1)


__all__ = [
    "accuracy"
]

if __name__ == "__main__":
    pass
