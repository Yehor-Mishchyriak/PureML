"""Activation functions with explicit forward/grad pairs (TensorValuedFunction-based).

Implements sigmoid, relu, tanh, softmax, and log_softmax with stable numerics, axis support,
and shape-safe vector-Jacobian products. Grad paths reuse cached forward outputs via node
contexts when available and fall back to recomputation otherwise."""
from __future__ import annotations

# third party
import numpy as np
# built-in
import logging
# local
from .machinery import Tensor, TensorValuedFunction, _shape_safe_grad, _update_ctx

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                Elementwise activations
# *----------------------------------------------------*

def _sigmoid(x: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    """Elementwise sigmoid; batch-safe via broadcasting."""
    _logger.debug("sigmoid fwd: x.shape=%s, dtype=%s", getattr(x, "shape", None), getattr(x, "dtype", None))
    out = 1.0 / (1.0 + np.exp(-x))
    _update_ctx(ctx=context, out=out)
    _logger.debug("sigmoid fwd: out.shape=%s", getattr(out, "shape", None))
    return out

@_shape_safe_grad
def _sigmoid_grad(upstream_grad: np.ndarray, x: np.ndarray, *, context: dict | None = None):
    """VJP for sigmoid: up * s * (1 - s)."""
    _logger.debug(
        "sigmoid bwd: up.shape=%s, x.shape=%s",
        getattr(upstream_grad, "shape", None), getattr(x, "shape", None)
    )
    s = (context or {}).get("out")
    if s is None:
        s = 1.0 / (1.0 + np.exp(-x))
    grad = upstream_grad * s * (1.0 - s)
    _logger.debug("sigmoid bwd: grad.shape=%s", getattr(grad, "shape", None))
    return (grad,)

def _relu(x: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    """Elementwise ReLU; batch-safe via broadcasting."""
    _logger.debug("relu fwd: x.shape=%s, dtype=%s", getattr(x, "shape", None), getattr(x, "dtype", None))
    out = np.maximum(0, x)  # elementwise: max(0, x_i)
    _update_ctx(ctx=context, out=out)
    _logger.debug("relu fwd: out.shape=%s", getattr(out, "shape", None))
    return out

@_shape_safe_grad
def _relu_grad(upstream_grad: np.ndarray, x: np.ndarray, *, context: dict | None = None):
    """VJP for ReLU: up * 1[x>0]."""
    _logger.debug(
        "relu bwd: up.shape=%s, x.shape=%s",
        getattr(upstream_grad, "shape", None), getattr(x, "shape", None)
    )
    out = (context or {}).get("out")
    if out is not None:
        mask = (out > 0).astype(x.dtype)  # same as (x>0), but reuse cached out
    else:
        mask = (x > 0).astype(x.dtype)
    grad = upstream_grad * mask
    _logger.debug("relu bwd: grad.shape=%s", getattr(grad, "shape", None))
    return (grad,)

def _tanh(x: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    """Elementwise tanh; batch-safe via broadcasting."""
    _logger.debug("tanh fwd: x.shape=%s, dtype=%s", getattr(x, "shape", None), getattr(x, "dtype", None))
    e_x = np.exp(x)
    e_neg_x = np.exp(-x)
    out = (e_x - e_neg_x) / (e_x + e_neg_x)
    _update_ctx(ctx=context, out=out)
    _logger.debug("tanh fwd: out.shape=%s", getattr(out, "shape", None))
    return out

@_shape_safe_grad
def _tanh_grad(upstream_grad: np.ndarray, x: np.ndarray, *, context: dict | None = None):
    """VJP for tanh: up * (1 - tanh(x)^2)."""
    _logger.debug(
        "tanh bwd: up.shape=%s, x.shape=%s",
        getattr(upstream_grad, "shape", None), getattr(x, "shape", None)
    )
    t = (context or {}).get("out")
    if t is None:
        e_x = np.exp(x)
        e_neg_x = np.exp(-x)
        t = (e_x - e_neg_x) / (e_x + e_neg_x)
    grad = upstream_grad * (np.ones_like(x) - t**2)
    _logger.debug("tanh bwd: grad.shape=%s", getattr(grad, "shape", None))
    return (grad,)

# *----------------------------------------------------*
#              Softmax family (batch-aware)
# *----------------------------------------------------*
# Built axis-parameterized closures so one can softmax over any axis
# while supporting arbitrary leading batch dims.

def _softmax_fwd(axis: int):
    def _softmax(x: np.ndarray, *, context: dict | None = None):
        """Stable softmax along `axis`, batch-safe over other dims."""
        _logger.debug("softmax fwd: x.shape=%s, axis=%d, dtype=%s", getattr(x, "shape", None), axis, getattr(x, "dtype", None))
        m = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - m)
        out = e / np.sum(e, axis=axis, keepdims=True)
        _update_ctx(ctx=context, out=out)
        _logger.debug("softmax fwd: out.shape=%s", getattr(out, "shape", None))
        return out
    return _softmax

def _softmax_bwd(axis: int):
    @_shape_safe_grad
    def _softmax_grad(upstream_grad: np.ndarray, x: np.ndarray, *, context: dict | None = None):
        """VJP for softmax: dX = (up - <up,s>) * s along `axis`."""
        _logger.debug(
            "softmax bwd: up.shape=%s, x.shape=%s, axis=%d",
            getattr(upstream_grad, "shape", None), getattr(x, "shape", None), axis
        )
        s = (context or {}).get("out")
        if s is None:
            m = np.max(x, axis=axis, keepdims=True)
            e = np.exp(x - m)
            s = e / np.sum(e, axis=axis, keepdims=True)
        dot = np.sum(upstream_grad * s, axis=axis, keepdims=True)
        grad = (upstream_grad - dot) * s
        _logger.debug("softmax bwd: grad.shape=%s", getattr(grad, "shape", None))
        return (grad,)
    return _softmax_grad

def _log_softmax_fwd(axis: int):
    def _log_softmax(x: np.ndarray, *, context: dict | None = None):
        """Stable log_softmax along `axis`, batch-safe over other dims."""
        _logger.debug("log_softmax fwd: x.shape=%s, axis=%d, dtype=%s", getattr(x, "shape", None), axis, getattr(x, "dtype", None))
        m = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - m)
        lse = np.log(np.sum(e, axis=axis, keepdims=True))
        out = x - m - lse
        _update_ctx(ctx=context, out=out)
        _logger.debug("log_softmax fwd: out.shape=%s", getattr(out, "shape", None))
        return out
    return _log_softmax

def _log_softmax_bwd(axis: int):
    @_shape_safe_grad
    def _log_softmax_grad(upstream_grad: np.ndarray, x: np.ndarray, *, context: dict | None = None):
        """VJP for log_softmax: dX = up - softmax(x) * sum(up, axis)."""
        _logger.debug(
            "log_softmax bwd: up.shape=%s, x.shape=%s, axis=%d",
            getattr(upstream_grad, "shape", None), getattr(x, "shape", None), axis
        )
        # prefer cached log-softmax; derive softmax from it
        logp = (context or {}).get("out")
        if logp is not None:
            s = np.exp(logp)  # stable: softmax = exp(log_softmax)
        else:
            m = np.max(x, axis=axis, keepdims=True)
            e = np.exp(x - m)
            s = e / np.sum(e, axis=axis, keepdims=True)
        sum_up = np.sum(upstream_grad, axis=axis, keepdims=True)
        grad = upstream_grad - sum_up * s
        _logger.debug("log_softmax bwd: grad.shape=%s", getattr(grad, "shape", None))
        return (grad,)
    return _log_softmax_grad

# *----------------------------------------------------*
#                      PUBLIC API
# *----------------------------------------------------*

def sigmoid(x: Tensor) -> Tensor:
    return TensorValuedFunction(_sigmoid, _sigmoid_grad)(x)

def relu(x: Tensor) -> Tensor:
    return TensorValuedFunction(_relu, _relu_grad)(x)

def tanh(x: Tensor) -> Tensor:
    return TensorValuedFunction(_tanh, _tanh_grad)(x)

def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Softmax over `axis` with stable forward and Jacobian-free VJP."""
    return TensorValuedFunction(_softmax_fwd(axis), _softmax_bwd(axis))(x)

def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Log-Softmax over `axis` with stable forward and Jacobian-free VJP."""
    return TensorValuedFunction(_log_softmax_fwd(axis), _log_softmax_bwd(axis))(x)


__all__ = [
    "sigmoid",
    "relu",
    "tanh",
    "softmax",
    "log_softmax"
]

if __name__ == "__main__":
    pass
