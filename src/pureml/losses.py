"""Supervised losses built on PureML's autodiff primitives.

Includes mean squared error, binary cross-entropy (optionally from logits with internal sigmoid),
and categorical cross-entropy (from logits or probs) with optional label smoothing. Each loss
is expressed as a `TensorValuedFunction` with cached forward outputs for stable, shape-safe
gradients (using `_shape_safe_grad` for unbroadcasting)."""
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

_EPS = 1e-12
_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                       UTILITIES
# *----------------------------------------------------*

def _smooth(Y: np.ndarray, eps: float, *, binary: bool = False) -> np.ndarray:
    """Return label-smoothed targets.

    Args:
        Y: Targets to smooth (any shape).
        eps: Smoothing factor in [0, 1).
        binary: If True, smooth toward 0.5; else toward uniform over last axis.

    Returns:
        Smoothed targets with the same shape as `Y`.
    """
    _logger.debug("smooth: eps=%.4f, binary=%s, Y.shape=%s",
                  float(eps), bool(binary), getattr(Y, "shape", None))
    if eps == 0.0:
        return Y
    if binary:
        return (1.0 - eps) * Y + 0.5 * eps
    K = Y.shape[-1]
    return (1.0 - eps) * Y + (eps / K)

# *----------------------------------------------------*
#                  Batch-aware losses
# *----------------------------------------------------*

# --------------------------- MEAN SQUARED ERROR ---------------------------
def _mse(Y: np.ndarray, Y_hat: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    """Mean squared error (MSE), averaged over all elements.

    Args:
        Y: Ground-truth array; shape and dtype broadcastable with Y_hat.
        Y_hat: Predictions; same shape as Y after broadcasting.

    Returns:
        Scalar numpy array (0-D) with the mean squared error.
    """
    _logger.debug("MSE fwd: Y.shape=%s, Y_hat.shape=%s, dtype=%s",
                  getattr(Y, "shape", None), getattr(Y_hat, "shape", None), getattr(Y_hat, "dtype", None))
    E = Y - Y_hat
    out = np.mean(E * E)
    _logger.debug("MSE fwd: loss=%s", float(out))

    _update_ctx(context, E=E)

    return out

@_shape_safe_grad
def _mse_grad(upstream_grad: np.ndarray, Y: np.ndarray, Y_hat: np.ndarray, *, context: dict | None = None):
    """VJP for MSE: dY = up * 2*(Y - Y_hat)/N, dY_hat = up * -2*(Y - Y_hat)/N.

    Args:
        upstream_grad: Scalar upstream gradient broadcasting over the loss.
        Y: Ground truth.
        Y_hat: Predictions.

    Returns:
        Tuple (dL/dY, dL/dY_hat) with the same shapes as inputs.
    """
    _logger.debug("MSE bwd: up.shape=%s, Y.shape=%s, Y_hat.shape=%s",
                  getattr(upstream_grad, "shape", None), getattr(Y, "shape", None), getattr(Y_hat, "shape", None))
    E = (context or {}).get("E", Y - Y_hat)
    N = E.size if E.size else 1
    gY     = upstream_grad * ( 2.0 * E / N)
    gY_hat = upstream_grad * (-2.0 * E / N)
    _logger.debug("MSE bwd: N=%d", N)
    return gY, gY_hat

def MSE(Y: Tensor, Y_hat: Tensor) -> Tensor:
    """Mean squared error (public wrapper).

    Computes `mean((Y - Y_hat)^2)` over all elements.

    Args:
        Y: Target tensor (broadcastable with `Y_hat`).
        Y_hat: Prediction tensor.

    Returns:
        Tensor: Scalar loss tensor.
    """
    _logger.debug("MSE call: Y.shape=%s, Y_hat.shape=%s",
                  getattr(Y, "shape", None), getattr(Y_hat, "shape", None))
    return TensorValuedFunction(_mse, _mse_grad)(Y, Y_hat)

# ------------------------ BINARY CROSS ENTROPY LOSS -----------------------
def _bce(Y: np.ndarray, Y_hat: np.ndarray, *, context: dict | None = None, label_smoothing: float = 0.0) -> np.ndarray:
    """Binary cross-entropy on probabilities."""
    Y = _smooth(Y, label_smoothing, binary=True)

    p = np.clip(Y_hat, _EPS, 1.0 - _EPS)  # <-- this just ensures no logs of 0 (or else boom)
    out = -np.mean(Y * np.log(p) + (1.0 - Y) * np.log(1.0 - p))
    _update_ctx(context, p=p, label_smoothing=label_smoothing)

    return out

@_shape_safe_grad # NOTE: context is required here! we must pass the smoothing factor!
def _bce_grad(upstream_grad: np.ndarray, Y: np.ndarray, Y_hat: np.ndarray, *, context: dict):
    """Backward for BCE(probs)."""
    ls = float(context.get("label_smoothing"))
    Y = _smooth(Y, ls, binary=True)

    p = context.get("p", np.clip(Y_hat, _EPS, 1.0 - _EPS))
    N = Y.size
    # dL/dp
    dLp = (-Y / p) + ((1.0 - Y) / (1.0 - p))
    gY_hat = upstream_grad * (dLp / N)
    # dL/dY
    gY = (1-ls) * upstream_grad * (-(np.log(p) - np.log(1.0 - p)) / N)

    return gY, gY_hat

def _sigmoid_bce(Y: np.ndarray, Z: np.ndarray, *,
                 context: dict | None = None, label_smoothing: float = 0.0) -> np.ndarray:
    """Binary cross-entropy on logits."""
    Y = _smooth(Y, label_smoothing, binary=True)

    # per-element: softplus(Z) - Y*Z, then mean over all elements
    # stable softplus
    per = np.maximum(Z, 0.0) - Y * Z + np.log1p(np.exp(-np.abs(Z)))
    out = np.mean(per)
    _update_ctx(context, S=lambda: 1.0 / (1.0 + np.exp(-Z)), label_smoothing=label_smoothing)

    return out

@_shape_safe_grad
def _sigmoid_bce_grad(upstream_grad: np.ndarray, Y: np.ndarray, Z: np.ndarray, *, context: dict):
    """Backward for BCE(logits)."""
    ls = float(context.get("label_smoothing"))
    Y = _smooth(Y, ls, binary=True)

    # N = number of elements (no class axis in binary case)
    N = Y.size
    S = context.get("S", 1.0 / (1.0 + np.exp(-Z)))     # sigmoid
    gZ = upstream_grad * ((S - Y) / N)
    # d/dY = -Z
    gY = (1-ls) * upstream_grad * (-Z / N)

    return gY, gZ

def BCE(Y: Tensor, Y_hat: Tensor, *, from_logits: bool = False,
        label_smoothing: float = 0.0) -> Tensor:
    """Binary cross-entropy.

    Computes BCE either from probabilities (`from_logits=False`) or logits
    (`from_logits=True`). Optional label smoothing pulls labels toward 0.5.

    Args:
        Y: Target tensor in [0, 1], same shape as `Y_hat`.
        Y_hat: Probabilities (if `from_logits=False`) or logits (if `from_logits=True`).
        from_logits: Interpret `Y_hat` as logits if True.
        label_smoothing: ε in [0, 1). 0 disables smoothing.

    Returns:
        Tensor: Scalar loss tensor.
    """
    _logger.debug("BCE call: from_logits=%s, label_smoothing=%.4f, Y.shape=%s, Y_hat.shape=%s",
                  bool(from_logits), float(label_smoothing),
                  getattr(Y, "shape", None), getattr(Y_hat, "shape", None))
    if from_logits:
        return TensorValuedFunction(_sigmoid_bce, _sigmoid_bce_grad)(
            Y, Y_hat, label_smoothing=label_smoothing)
    else:
        return TensorValuedFunction(_bce, _bce_grad)(
            Y, Y_hat, label_smoothing=label_smoothing)

# --------------------- CATEGORICAL CROSS ENTROPY LOSS ---------------------
def _cce(Y: np.ndarray, Y_hat: np.ndarray, *,
         context: dict | None = None, label_smoothing: float = 0.0) -> np.ndarray:
    """Categorical cross-entropy on probabilities."""
    Y = _smooth(Y, label_smoothing, binary=False)

    P = np.clip(Y_hat, _EPS, 1.0)  # last axis are class probs
    per = -np.sum(Y * np.log(P), axis=-1)  # per-sample loss
    out = np.mean(per)

    _update_ctx(context, P=P, label_smoothing=label_smoothing)

    return out

@_shape_safe_grad
def _cce_grad(upstream_grad: np.ndarray, Y: np.ndarray, P: np.ndarray, *, context: dict):
    """Backward for CCE(probs)."""
    ls = float(context.get("label_smoothing"))
    Y = _smooth(Y, ls, binary=False)

    P = np.clip(context.get("P", P), _EPS, 1.0)
    # number of samples = product of all non-class dims
    N = int(np.prod(Y.shape[:-1])) if Y.ndim > 1 else 1  # <-- note, you can't just let N be the batch axis
    # N = B doesn't work when each batch is also divided into, say, tokens, so you need to account for the loss
    # contributed by each missclassified token. So not just each sample from the batch, but tokens within each sample!
    gP = upstream_grad * (-(Y / P) / N)           # dL/dP
    gY = (1-ls) * upstream_grad * (-(np.log(P)) / N)       # dL/dY

    return gY, gP

def _softmax_cce(Y: np.ndarray, Z: np.ndarray, *,
                 context: dict | None = None, label_smoothing: float = 0.0) -> np.ndarray:
    """Categorical cross-entropy on logits."""
    Y = _smooth(Y, label_smoothing, binary=False)

    # log-softmax for stability
    m = np.max(Z, axis=-1, keepdims=True)
    e = np.exp(Z - m)
    lse = np.log(np.sum(e, axis=-1, keepdims=True))
    logp = (Z - m) - lse               # log softmax
    per = -np.sum(Y * logp, axis=-1)   # per-sample
    out = np.mean(per)

    # cache both logp and softmax probs S
    S = e / np.sum(e, axis=-1, keepdims=True)
    _update_ctx(context, logp=logp, S=S, label_smoothing=label_smoothing)

    return out

@_shape_safe_grad
def _softmax_cce_grad(upstream_grad: np.ndarray, Y: np.ndarray, Z: np.ndarray, *, context: dict):
    """Backward for CCE(logits)."""
    ls = float(context.get("label_smoothing"))
    Y = _smooth(Y, ls, binary=False)

    S = context.get("S", None)
    logp = context.get("logp", None)
    if S is None or logp is None:
        m = np.max(Z, axis=-1, keepdims=True)
        e = np.exp(Z - m)
        S = e / np.sum(e, axis=-1, keepdims=True)
        logp = (Z - m) - np.log(np.sum(e, axis=-1, keepdims=True))

    # normalization factor: all non-class positions
    N = int(np.prod(Y.shape[:-1])) if Y.ndim > 1 else 1
    # dL/dZ = (p - y)/N ; dL/dY = -log p / N
    gZ = upstream_grad * ((S - Y) / N)
    gY = (1-ls) * upstream_grad * (-(logp) / N)

    return gY, gZ

def CCE(Y: Tensor, Y_hat: Tensor, *, from_logits: bool = False,
        label_smoothing: float = 0.0) -> Tensor:
    """Categorical cross-entropy.

    Computes CCE along the last axis either from probabilities
    (`from_logits=False`) or from logits via a stable log-softmax
    (`from_logits=True`). Optional label smoothing mixes targets
    toward the uniform distribution over classes (ε/K).

    Args:
        Y: Target tensor on the simplex along the last axis (one-hot or soft).
        Y_hat: Probabilities (if `from_logits=False`) or logits (if `from_logits=True`)
            with the same shape as `Y`.
        from_logits: Interpret `Y_hat` as logits if True.
        label_smoothing: ε in [0, 1). 0 disables smoothing.

    Returns:
        Tensor: Scalar loss tensor.
    """
    _logger.debug("CCE call: from_logits=%s, label_smoothing=%.4f, Y.shape=%s, Y_hat.shape=%s",
                  bool(from_logits), float(label_smoothing),
                  getattr(Y, "shape", None), getattr(Y_hat, "shape", None))
    if from_logits:
        return TensorValuedFunction(_softmax_cce, _softmax_cce_grad)(
            Y, Y_hat, label_smoothing=label_smoothing)
    else:
        return TensorValuedFunction(_cce, _cce_grad)(
            Y, Y_hat, label_smoothing=label_smoothing)


__all__ = [
    "MSE",
    "BCE",
    "CCE"
]

if __name__ == "__main__":
    pass
