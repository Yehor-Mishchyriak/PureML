"""Autodiff-aware math utilities: EWMA helpers plus distance/stat ops backed by custom VJPs.

Provides Euclidean distance, mean/deviation/variance/std/sum with axis support and cached
intermediates, all exposed as `TensorValuedFunction` wrappers with shape-safe gradients."""
from __future__ import annotations

# third party
import numpy as np
# built in
import logging
# local
from .machinery import Tensor, TensorValuedFunction, _shape_safe_grad, _update_ctx

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                       UTILITIES
# *----------------------------------------------------*

def ewma(running: np.ndarray, current: np.ndarray, *, beta: float):
    return beta * running + (1-beta) * current

# *----------------------------------------------------*
#                        GENERAL
# *----------------------------------------------------*

def _euclidean_distance(x: np.ndarray, y: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    """Scalar Euclidean distance: ||x - y||_2 over all elements.

    Args:
        x: First vector/array.
        y: Second vector/array.
        context: Optional node cache dict (used to stash intermediates).

    Returns:
        Scalar numpy array (0-D) with Euclidean distance.
    """
    _logger.debug("euclid fwd: x.shape=%s, y.shape=%s, dtype=%s/%s",
                  getattr(x, "shape", None), getattr(y, "shape", None),
                  getattr(x, "dtype", None), getattr(y, "dtype", None))
    
    diff = x - y
    out = np.sqrt(np.sum(diff * diff))

    # cache intermediates if context is provided
    _update_ctx(context, diff=diff, out=out)

    _logger.debug("euclid fwd: diff.shape=%s, dist=%s",
                  getattr(diff, "shape", None), float(out))
    return out

@_shape_safe_grad
def _euclidean_distance_grad(
    upstream_grad: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    context: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """VJP for Euclidean distance d = ||x - y||_2.

    dL/dx = up * (x - y) / d
    dL/dy = -dL/dx

    Args:
        upstream_grad: Upstream scalar gradient `dL/dd`.
        x: Forward input `x`.
        y: Forward input `y`.
        context: Optional node cache dict (may contain 'diff' and 'out').

    Returns:
        Tuple of gradients `(dL/dx, dL/dy)` with shapes like `x` and `y`.
    """
    _logger.debug("euclid bwd: up.shape=%s, x.shape=%s, y.shape=%s",
                  getattr(upstream_grad, "shape", None),
                  getattr(x, "shape", None), getattr(y, "shape", None))

    ctx = context or {}
    diff = ctx.get("diff", x - y)
    d = ctx.get("out", np.sqrt(np.sum(diff * diff)))

    # avoid div-by-zero at x == y (subgradient = 0 vector)
    denom = d + 1e-12
    scale = upstream_grad / denom
    _logger.debug("euclid bwd: denom=%s", float(denom))

    dL_dx = scale * diff
    dL_dy = -dL_dx
    _logger.debug("euclid bwd: dL_dx.shape=%s, dL_dy.shape=%s",
                  getattr(dL_dx, "shape", None), getattr(dL_dy, "shape", None))
    return dL_dx, dL_dy

def euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
    """Public API wrapper for Euclidean distance."""
    _logger.debug("euclid API: Tensor x.shape=%s, y.shape=%s",
                  getattr(x, "shape", None), getattr(y, "shape", None))
    return TensorValuedFunction(_euclidean_distance, _euclidean_distance_grad)(x, y)

# *----------------------------------------------------------*
#                          STATISTICS
# *----------------------------------------------------------*

def _mean(X: np.ndarray, *, context: dict | None = None, axis: int | None = None) -> np.ndarray:
    """Mean over a chosen axis (default: last)."""
    _logger.debug("mean fwd: X.shape=%s, dtype=%s, axis=%s", getattr(X, "shape", None), getattr(X, "dtype", None), axis)
    out = np.mean(X, axis=axis)                                                           
    _update_ctx(context, axis=axis)
    _logger.debug("mean fwd: out.shape=%s", getattr(out, "shape", None))
    return out

@_shape_safe_grad
def _mean_grad(upstream: np.ndarray, X: np.ndarray, *, context: dict | None = None, axis: int | None = None) -> tuple[np.ndarray]:
    """VJP for mean over a chosen axis."""
    ax = (context or {}).get("axis", axis)
    if ax is None:
        N = X.size if X.size else 1
        grad = np.broadcast_to(upstream / N, X.shape)
        _logger.debug("mean bwd: up.shape=%s, X.shape=%s, axis=%s (None->all), N=%d",
                      getattr(upstream, "shape", None), getattr(X, "shape", None), ax, N)
        _logger.debug("mean bwd: grad.shape=%s", getattr(grad, "shape", None))
        return (grad,)
    if ax < 0: ax += X.ndim
    _logger.debug("mean bwd: up.shape=%s, X.shape=%s, axis=%s",
                  getattr(upstream, "shape", None), getattr(X, "shape", None), ax)
    N = X.shape[ax]
    up = np.expand_dims(upstream, axis=ax)
    grad = np.broadcast_to(up / N, X.shape)
    _logger.debug("mean bwd: grad.shape=%s", getattr(grad, "shape", None))
    return (grad,)

def mean(X: Tensor, *, axis: int | None = None) -> Tensor:
    _logger.debug("mean API: Tensor X.shape=%s, axis=%s", getattr(X, "shape", None), axis)
    return TensorValuedFunction(_mean, _mean_grad)(X, axis=axis)

def _deviation(X: np.ndarray, *, context: dict | None = None, axis: int = -1) -> np.ndarray:
    """Deviation from the mean over a chosen axis: dev = X - mean(X, axis)."""
    _logger.debug("dev fwd: X.shape=%s, dtype=%s, axis=%s",
                  getattr(X, "shape", None), getattr(X, "dtype", None), axis)
    mu = np.mean(X, axis=axis, keepdims=True)
    out = X - mu
    _update_ctx(context, mu=mu, dev=out, axis=axis)
    _logger.debug("dev fwd: mu.shape=%s, out.shape=%s", getattr(mu, "shape", None), getattr(out, "shape", None))
    return out

@_shape_safe_grad
def _deviation_grad(upstream: np.ndarray, X: np.ndarray, *, context: dict | None = None, axis: int | None = -1) -> tuple[np.ndarray]:
    """VJP for deviation: upstream - mean(upstream, axis)."""
    ax = (context or {}).get("axis", axis)
    _logger.debug("dev bwd: up.shape=%s, X.shape=%s, axis=%s",
                  getattr(upstream, "shape", None), getattr(X, "shape", None), ax)
    mean_up = np.mean(upstream, axis=ax, keepdims=True)
    out = upstream - mean_up
    _logger.debug("dev bwd: out.shape=%s", getattr(out, "shape", None))
    return (out,)

def deviation(X: Tensor, *, axis: int = -1) -> Tensor:
    """Public API wrapper for deviation-from-mean."""
    _logger.debug("dev API: Tensor X.shape=%s, axis=%s", getattr(X, "shape", None), axis)
    return TensorValuedFunction(_deviation, _deviation_grad)(X, axis=axis)

def _variance(X: np.ndarray, *, context: dict | None = None, axis: int = -1) -> np.ndarray:
    """Variance over a chosen axis: var = mean((X - mu)^2, axis)."""
    _logger.debug("var fwd: X.shape=%s, dtype=%s, axis=%s", getattr(X, "shape", None), getattr(X, "dtype", None), axis)
    mu = np.mean(X, axis=axis, keepdims=True)
    dev = X - mu
    var = np.mean(dev * dev, axis=axis)
    _update_ctx(context, mu=mu, dev=dev, var=var, axis=axis)
    _logger.debug("var fwd: var.shape=%s", getattr(var, "shape", None))
    return var

@_shape_safe_grad
def _variance_grad(upstream: np.ndarray, X: np.ndarray, *, context: dict | None = None, axis: int | None = -1) -> tuple[np.ndarray]:
    """VJP for variance over a chosen axis."""
    ctx = context or {}
    ax = ctx.get("axis", axis)
    if ax is None:
        _logger.debug("var bwd: up.shape=%s, X.shape=%s, axis=%s (None->all)",
                      getattr(upstream, "shape", None), getattr(X, "shape", None), ax)
        dev = ctx.get("dev", X - np.mean(X, axis=None, keepdims=True))
        N = X.size if X.size else 1
        grad = (2.0 / N) * dev * upstream
        _logger.debug("var bwd: grad.shape=%s", getattr(grad, "shape", None))
        return (grad,)
    if ax < 0: ax += X.ndim
    _logger.debug("var bwd: up.shape=%s, X.shape=%s, axis=%s",
                  getattr(upstream, "shape", None), getattr(X, "shape", None), ax)
    dev = ctx.get("dev", X - np.mean(X, axis=ax, keepdims=True))
    N = X.shape[ax]
    up = np.expand_dims(upstream, axis=ax)
    grad = (2.0 / N) * dev * up
    _logger.debug("var bwd: grad.shape=%s", getattr(grad, "shape", None))
    return (grad,)

def variance(X: Tensor, *, axis: int = -1) -> Tensor:
    _logger.debug("var API: Tensor X.shape=%s, axis=%s", getattr(X, "shape", None), axis)
    return TensorValuedFunction(_variance, _variance_grad)(X, axis=axis)

def _std(X: np.ndarray, *, context: dict | None = None, axis: int = -1) -> np.ndarray:
    """Standard deviation over a chosen axis: std = sqrt(var)."""
    _logger.debug("std fwd: X.shape=%s, dtype=%s, axis=%s", getattr(X, "shape", None), getattr(X, "dtype", None), axis)
    mu = np.mean(X, axis=axis, keepdims=True)
    dev = X - mu
    var = np.mean(dev * dev, axis=axis)
    std = np.sqrt(var + 1e-12)
    _update_ctx(context, mu=mu, dev=dev, var=var, out=std, axis=axis)
    _logger.debug("std fwd: std.shape=%s", getattr(std, "shape", None))
    return std

@_shape_safe_grad
def _std_grad(upstream: np.ndarray, X: np.ndarray, *, context: dict | None = None, axis: int | None = -1) -> tuple[np.ndarray]:
    """VJP for standard deviation over a chosen axis: d std / dX = dev / (N * std)."""
    ctx = context or {}
    ax = ctx.get("axis", axis)
    if ax is None:
        _logger.debug("std bwd: up.shape=%s, X.shape=%s, axis=%s (None->all)",
                      getattr(upstream, "shape", None), getattr(X, "shape", None), ax)
        N = X.size if X.size else 1
        std = ctx.get("out", _std(X, axis=None))
        dev = ctx.get("dev", X - np.mean(X, axis=None, keepdims=True))
        grad = upstream * (dev / (N * (std + 1e-12)))
        _logger.debug("std bwd: grad.shape=%s", getattr(grad, "shape", None))
        return (grad,)
    if ax < 0: ax += X.ndim
    _logger.debug("std bwd: up.shape=%s, X.shape=%s, axis=%s",
                  getattr(upstream, "shape", None), getattr(X, "shape", None), ax)
    N = X.shape[ax]
    std = ctx.get("out", _std(X, axis=ax))
    dev = ctx.get("dev", X - np.mean(X, axis=ax, keepdims=True))
    up = np.expand_dims(upstream, axis=ax)
    grad = up * (dev / (N * (std + 1e-12)))
    _logger.debug("std bwd: grad.shape=%s", getattr(grad, "shape", None))
    return (grad,)

def std(X: Tensor, *, axis: int = -1) -> Tensor:
    _logger.debug("std API: Tensor X.shape=%s, axis=%s", getattr(X, "shape", None), axis)
    return TensorValuedFunction(_std, _std_grad)(X, axis=axis)

def _sum(X: np.ndarray, *, context: dict | None = None, axis: int | None = -1) -> np.ndarray:
    """Sum over a chosen axis (default: last). If axis=None, sum over all elements."""
    _logger.debug("sum fwd: X.shape=%s, dtype=%s, axis=%s",
                  getattr(X, "shape", None), getattr(X, "dtype", None), axis)
    out = np.sum(X, axis=axis)
    _update_ctx(context, axis=axis)
    _logger.debug("sum fwd: out.shape=%s", getattr(out, "shape", None))
    return out

@_shape_safe_grad
def _sum_grad(upstream: np.ndarray, X: np.ndarray, *, context: dict | None = None, axis: int | None = -1) -> tuple[np.ndarray]:
    """VJP for sum: broadcast upstream back over the reduced axis/axes."""
    ax = (context or {}).get("axis", axis)
    if ax is None:
        up = upstream  # scalar or already broadcastable
    else:
        if ax < 0:
            ax += X.ndim
        up = np.expand_dims(upstream, axis=ax)
    grad = np.broadcast_to(up, X.shape)
    _logger.debug("sum bwd: grad.shape=%s", getattr(grad, "shape", None))
    return (grad,)

def sum(X: Tensor, *, axis: int | None = -1) -> Tensor:
    _logger.debug("sum API: Tensor X.shape=%s, axis=%s", getattr(X, "shape", None), axis)
    return TensorValuedFunction(_sum, _sum_grad)(X, axis=axis)


__all__ = [
    "euclidean_distance",
    "mean",
    "deviation",
    "variance",
    "std",
    "sum",
    "ewma"
]
