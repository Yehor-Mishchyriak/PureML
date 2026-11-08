"""Evaluation helpers for PureML models.

Provides top-1 accuracy and confusion-matrix counts over a Dataset via DataLoader batching, respecting model
train/eval mode toggles, using `no_grad`, and handling logits/probs (argmax) or class-index
outputs as well as one-hot targets."""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
import warnings

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


def confusion_matrix(
    mdl: BaseModel,
    test_set: Dataset,
    classes: Sequence[int],
    batch_size: int = 32,
) -> np.ndarray:
    """Count predictions by true class (rows) and predicted class (columns).

    Both axes follow the supplied, non-empty sequence of distinct integer class
    labels. Unobserved classes have zero counts. Labels outside ``classes`` raise
    ``ValueError`` instead of silently losing samples.

    Like :func:`accuracy`, accepts class-index outputs/targets (B,) or scores and
    one-hot targets (B, C), which are converted with argmax. For scores, class
    labels are the original column indices; ``classes`` only orders the matrix.
    Evaluation disables gradients and restores the model's prior mode, even when
    prediction or validation fails.
    """
    if not len(classes) or any(not isinstance(label, Integral) for label in classes):
        raise ValueError("classes must be a non-empty sequence of integer labels")
    positions = {label: i for i, label in enumerate(classes)}
    if len(positions) != len(classes):
        raise ValueError("classes must contain distinct labels")
    counts = np.zeros((len(classes), len(classes)), dtype=np.int64)

    def indices(t: Tensor) -> np.ndarray:
        if t.data.ndim == 1:
            return t.data
        if t.data.ndim == 2:
            return t.data.argmax(axis=-1)
        raise ValueError("expected class indices (B,) or class scores (B, C)")

    prev_mode = getattr(mdl, "training", None)
    try:
        if hasattr(mdl, "eval"):
            mdl.eval()
        with no_grad():
            for X, Y in DataLoader(test_set, batch_size=batch_size):
                pred_idx = indices(mdl(X))
                true_idx = indices(Y)
                if pred_idx.shape != true_idx.shape:
                    raise ValueError("prediction and target shapes must match")
                try:
                    rows = np.fromiter((positions[label] for label in true_idx), dtype=np.intp)
                    cols = np.fromiter((positions[label] for label in pred_idx), dtype=np.intp)
                except KeyError as exc:
                    raise ValueError(f"label {exc.args[0]!r} is not in classes") from exc
                # Repeated true/predicted pairs must accumulate, even within a batch.
                np.add.at(counts, (rows, cols), 1)
    finally:
        if prev_mode is not None:
            if prev_mode and hasattr(mdl, "train"):
                mdl.train()
            elif prev_mode is False and hasattr(mdl, "eval"):
                mdl.eval()
    return counts

def precision(mdl: BaseModel, test_set: Dataset, classes: Sequence[int], zero_division: float|str = 0.0, average:str|None=None,batch_size: int = 32) -> float|np.ndarray:
    cm = confusion_matrix(mdl, test_set, classes, batch_size=batch_size)

    if zero_division not in (0.0, 1.0, "warn"):
        raise ValueError("zero_division must be 0.0, 1.0, or 'warn'")
    if zero_division == 0.0 or zero_division == "warn":
        zero_division_val = 0.0
    elif zero_division == 1.0:
        zero_division_val = 1.0    
    if average not in (None, "micro", "macro", "weighted"):
        raise ValueError("average must be None, 'micro', 'macro', or 'weighted'")
    
    if average == "micro":
        tp = np.sum(np.diag(cm))
        fp = np.sum(cm) - tp
        if tp + fp == 0:
            if zero_division == "warn":
                warnings.warn("Precision is ill-defined and being set to 0.0 due to no positive predictions.", RuntimeWarning)
        if tp + fp == 0:
            return zero_division_val
        return tp / (tp + fp)
    precisions = np.zeros(len(classes), dtype=np.float64)
    for i in range(len(classes)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        if tp + fp == 0:
            if zero_division == "warn":
                warnings.warn(f"Precision for class {classes[i]} is ill-defined and being set to 0.0 due to no positive predictions.", RuntimeWarning)
            precisions[i] = zero_division_val
        else:
            precisions[i] = tp / (tp + fp)
    if average is None:
        return precisions
    elif average == "macro":
        return np.mean(precisions)
    elif average == "weighted":
        support = np.sum(cm, axis=1)
        total_support = np.sum(support)
        if total_support == 0:
            return zero_division_val
        return float(np.sum(precisions * support) / total_support)
__all__ = [
    "accuracy", "confusion_matrix", "precision"
]

if __name__ == "__main__":
    pass
