"""Data utilities: Dataset abstractions, Tensor-backed datasets, batching, and reproducible splits.

`TensorDataset` normalizes samples to Tensors; `DataLoader` batches/iterates with shuffle/drop_last,
slice-fast path when available, and seeds via `util.get_random_seed`. Includes helpers to stack/
collate samples, index arrays/Tensors safely, and basic math helpers for batching."""
from __future__ import annotations

# third party
import numpy as np
# built-in
import logging
from typing import Any, Callable, Iterator, Optional
from random import Random
from math import floor, ceil
# local
from .machinery import Tensor
from . import util

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#               CLASSES & HELPER FUNCTIONS
# *----------------------------------------------------*

class Dataset:
    """Dataset: implement __len__ and __getitem__ (int or slice)."""
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int | slice) -> Any:
        raise NotImplementedError

class TensorDataset(Dataset):
    """Wrap one or more arrays/Tensors; supports int and slice indexing.

    Notes:
        • __getitem__ (int or slice) ALWAYS returns Tensors (requires_grad=False),
          even when the underlying storage was a NumPy array. This keeps downstream
          code (e.g., layers expecting Tensor inputs) consistent.
    """
    def __init__(self, *arrays: np.ndarray | Tensor) -> None:
        if not arrays:
            raise ValueError("TensorDataset requires at least one tensor/array")
        n0 = _first_dim(arrays[0])
        for i, a in enumerate(arrays[1:], 1):
            if _first_dim(a) != n0:
                raise ValueError(f"arrays[0] and arrays[{i}] have different length")
        self._arrays = arrays # f.e. X, Y
        self._N = int(n0) # how many feature-vector-label pairs we have
        _logger.debug("TensorDataset: N=%d, fields=%d", self._N, len(arrays))

    def __len__(self) -> int:
        return self._N

    def __getitem__(self, idx: int | slice):
        return tuple(_index(a, idx) for a in self._arrays) # return X[idx], Y[idx], f.e.

def _first_dim(a: np.ndarray | Tensor) -> int:
    arr = a.data if isinstance(a, Tensor) else np.asarray(a)
    if arr.ndim == 0:
        raise ValueError("Inputs must have leading sample dimension")
    return arr.shape[0]

def _index(a: np.ndarray | Tensor, idx: int | slice):
    """Index along the leading dimension and ALWAYS return a no-grad Tensor."""
    base = a.data if isinstance(a, Tensor) else a
    view = np.asarray(base)[idx]
    return Tensor(view, requires_grad=False)

# *----------------------------------------------------*
#                   Data Organization
# *----------------------------------------------------*

def _stack(items: list[Any]):
    """Stack a list of homogeneous items along axis 0; else return the list."""
    x0 = items[0]
    if isinstance(x0, Tensor):
        return Tensor(np.stack([x.data for x in items], axis=0), requires_grad=False)
    if isinstance(x0, np.ndarray):
        return np.stack(items, axis=0)
    if isinstance(x0, (float, int, np.floating, np.integer)):
        return np.asarray(items)
    return items  

def combine_samples(samples: list[Any]):
    """
    Collate a list of samples into a mini-batch.

    Transforms a list of tuples into a tuple of stacked arrays/Tensors.

    Example (classification or regression):
        # samples is a list of per-sample tuples (x_i, y_i)
        samples = [(X1, Y1), (X2, Y2), ..., (XN, YN)]

        # 1) transpose list-of-tuples -> tuple of lists
        #    [(X1, Y1), (X2, Y2), ...]  ->  ([X1, X2, ...], [Y1, Y2, ...])

        # 2) stack each list along a new batch axis
        #    X_batch = stack([X1, X2, ..., XN])    # shape: (N, *x_shape)
        #    y_batch = stack([Y1, Y2, ..., YN])    # scalar Ys -> shape (N,)
        #
        # Result: (X_batch, y_batch)

    Notes:
        • Works when Y is scalar per-sample (0-D): stacking yields shape (N,).
        • If Y has shape (N, 1), stacked y_batch will be (N, 1).
        • If samples aren't tuples (e.g., only X), this stacks the list directly.

    """
    if not samples:
        return samples
    s0 = samples[0]
    if isinstance(s0, tuple):
        fields = list(zip(*samples))
        return tuple(_stack(list(col)) for col in fields)
    return _stack(samples)

# *----------------------------------------------------*
#                       Data Loader
# *----------------------------------------------------*

class DataLoader:
    """
    DataLoader.

    Args:
        dataset: implements __len__ and __getitem__ (int and/or slice).
        batch_size: items per batch.
        shuffle: shuffle indices each iteration.
        drop_last: drop incomplete final batch.
        combine_samples_fn: how to combine a list of samples; defaults to `combine_samples`.
        seed: optional random seed.
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        combine_samples_fn: Optional[Callable[[list[Any]], Any]] = None,
        seed: int = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.combine_samples_fn = combine_samples if combine_samples_fn is None else combine_samples_fn
        self.seed = int(seed) if seed is not None else util.get_random_seed()
        self._rng = Random(self.seed)

        # one-time probe for slice support
        self._sliceable = False
        try:
            _ = self.dataset[0:0]
            self._sliceable = True
        except Exception:
            pass

        self._N = len(self.dataset)
        _logger.debug(
            "DataLoader: N=%d, B=%d, shuffle=%s, drop_last=%s, sliceable=%s, seed=%d",
            self._N, self.batch_size, self.shuffle, self.drop_last, self._sliceable, self.seed
        )

    def __len__(self) -> int:
        return floor(self._N / self.batch_size) if self.drop_last else ceil(self._N / self.batch_size)

    def __iter__(self) -> Iterator[Any]:
        N, B = self._N, self.batch_size

        # Fast contiguous ranges if no shuffle and dataset supports slices
        if (not self.shuffle) and self._sliceable:
            for start, end in util.batches_of(range(N), B, shuffle=False, ranges=True):
                if self.drop_last and (end - start) < B:
                    break
                # TensorDataset[start:end] already returns batched fields -> yield directly
                yield self.dataset[start:end] # (X[start:end], Y[start:end]) f.e.
            return

        # General path: index lists (handles shuffle and non-sliceables)
        for idxs in util.batches_of(range(N), B, shuffle=self.shuffle, out_as=list, rng=self._rng):
            if self.drop_last and len(idxs) < B:
                break
            samples = [self.dataset[i] for i in idxs]
            yield self.combine_samples_fn(samples) # (X[start:end], Y[start:end]) f.e.

# *----------------------------------------------------*
#                    Helper functions
# *----------------------------------------------------*

def _asarray(x, *, dtype=None):
    """np.asarray that unwraps Tensor.data if needed."""
    base = x.data if isinstance(x, Tensor) else x
    return np.asarray(base, dtype=dtype)

def _wrap_like(out: np.ndarray, like):
    """Return a Tensor (no-grad) if `like` was a Tensor; else return ndarray."""
    if isinstance(like, Tensor):
        return Tensor(out, requires_grad=False)
    return out

def one_hot(dims: int, label) -> np.ndarray | Tensor:
    """Create one-hot vector(s).

    If `label` is scalar → returns shape `(dims,)`.
    If `label` has shape `S` → returns shape `S + (dims,)`.
    If `label` is a Tensor, returns a Tensor (requires_grad=False).
    """
    arr = _asarray(label, dtype=int)
    # scalar case
    if arr.ndim == 0:
        v = np.zeros(dims, dtype=np.float64)
        idx = int(arr)
        if 0 <= idx < dims:
            v[idx] = 1.0
        return _wrap_like(v, label)

    # array case
    out = np.zeros(arr.shape + (dims,), dtype=np.float64)
    valid = (arr >= 0) & (arr < dims)
    if valid.any():
        idxs = np.where(valid)                 # tuple of indices for all leading dims
        out[idxs + (arr[valid],)] = 1.0        # set along last axis
    return _wrap_like(out, label)

def multi_hot(dims: int, labels) -> np.ndarray | Tensor:
    """Create multi-hot vector(s).

    Accepted inputs:
      * 1D array/list/Tensor of ints         → returns `(dims,)`
      * 2D array/Tensor of shape `(B, K)`    → returns `(B, dims)`
      * Ragged list-of-lists length B        → returns `(B, dims)`
    If `labels` is a Tensor, returns a Tensor (requires_grad=False).
    """
    arr = _asarray(labels)  # dtype may be object for ragged python lists

    # Fast paths for numeric ndarrays/Tensors
    if arr.dtype != object:
        if arr.ndim == 1:
            v = np.zeros(dims, dtype=np.float64)
            idx = arr.astype(int, copy=False).ravel()
            idx = idx[(idx >= 0) & (idx < dims)]
            if idx.size:
                v[idx] = 1.0
            return _wrap_like(v, labels)

        if arr.ndim == 2:
            B, _ = arr.shape
            v = np.zeros((B, dims), dtype=np.float64)
            idx = arr.astype(int, copy=False)
            valid = (idx >= 0) & (idx < dims)
            rows, cols = np.where(valid)
            if rows.size:
                v[rows, idx[rows, cols]] = 1.0
            return _wrap_like(v, labels)

        raise ValueError("multi_hot: numeric input must be 1D (single) or 2D (batch).")

    # Ragged list-of-lists fallback (labels likely not a Tensor here)
    seq = list(labels)
    B = len(seq)
    v = np.zeros((B, dims), dtype=np.float64)
    for i, lab in enumerate(seq):
        idx = np.asarray(lab, dtype=int).ravel()
        idx = idx[(idx >= 0) & (idx < dims)]
        if idx.size:
            v[i, idx] = 1.0
    return _wrap_like(v, labels)


__all__ = [
    "Dataset", 
    "TensorDataset", 
    "DataLoader", 
    "combine_samples",
    "multi_hot",
    "one_hot"
]

if __name__ == "__main__":
    pass
