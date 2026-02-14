#  /----------------------- THINGS TO NOTE ----------------------\
# | Often we use <Tensor>.data.dtype or <Tensor>.data.shape, but  |
# | there's no need anymore because `Tensor` class has .shape,    |
# | .dtype and other useful properties. The reason older .data    |
# | interface is used in this code is because it was being        |
# | written gradually and along with `machinery` module.          |
# | In any case, <Tensor>.shape and .dtype are encouraged now.    |   
#  \-------------------------------------------------------------/

"""Layer stack built on the PureML autodiff core.

Provides a `Layer` base (training mode toggle, parameters/buffers, apply_state),
and concrete layers:
- Affine with Xavier init, bias toggle, and seed/buffer metadata (W stored (n, m))
- Dropout (1 and 2 -D) (inverted, cached mask/scale, seedable, mode-aware)
- BatchNorm1d with running stats buffers and EMA momentum
- Embedding with optional pad freezing and seedable init
All layers use `TensorValuedFunction` ops from `machinery` and RNG helpers in `util`."""
from __future__ import annotations

# third party
import numpy as np
# built-in
from abc import ABC, abstractmethod
from math import floor
import logging
from typing import Literal
# local
from .machinery import (
    Tensor, TensorValuedFunction, _shape_safe_grad, _update_ctx, sqrt
)
from . import general_math
from .util import rng_from_seed, validate_layer_contract

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#               CLASSES & HELPER FUNCTIONS
# *----------------------------------------------------*

# *----------------------------------------------------*
#                WEIGHT INITIALIZATIONS
# *----------------------------------------------------*

def xavier_glorot_normal(
    fan_in: int,
    fan_out: int,
    *,
    rng: np.random.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Initialize weights and bias using Xavier/Glorot normal.

    Args:
        fan_in: Input feature dimension (>0).
        fan_out: Output feature dimension (>0).
        rng: Optional NumPy Generator. If None, a fresh default_rng() is used.

    Returns:
        (W, b): W.shape == (fan_out, fan_in), b.shape == (fan_out,)
                both as Tensor with requires_grad=True.
    """
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError(f"fan_in and fan_out must be > 0 (got {fan_in=}, {fan_out=})")

    gen = rng or np.random.default_rng()
    std = np.sqrt(2.0 / (fan_in + fan_out))
    W = gen.normal(0.0, std, size=(fan_out, fan_in))
    b = np.zeros((fan_out,))

    return Tensor(W, requires_grad=True), Tensor(b, requires_grad=True)

# REFERENCE: https://docs.pytorch.org/docs/stable/nn.init.html
#                         | | |  (see: torch.nn.init.kaiming_normal_, and
#                         V V V        torch.nn.init.calculate_gain)

def calculate_gain(
        nonlinearity: Literal[
                        'Affine',
                        'Conv1D',
                        'Conv2D',
                        "sigmoid",
                        "tanh",
                        "relu",
                        "leaky_relu"],
        param:        int | float | None):
    raise NotImplementedError("The recommended gain value computation function has not yet been implemented")

def kaiming_normal(
    fan_in: int,
    fan_out: int,
    nonlinearity: str,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    *,
    rng: np.random.Generator | None = None
) -> tuple[Tensor, Tensor]:
    # The method is described in Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015).
    raise NotImplementedError("The kaiming_normal initialization has not yet been implemented")

# *----------------------------------------------------*

class Layer(ABC):
    """A module with (optional) trainable parameters and (optional) non-trainable buffers."""

    def __init__(self, *, training: bool = True) -> None:
        self._training = bool(training)

    @property
    def training(self) -> bool:
        # Fallback to True if subclass didn't call super().__init__
        return getattr(self, "_training", True)

    @training.setter
    def training(self, mode: bool) -> None:
        mode = bool(mode)
        prev = getattr(self, "_training", None)
        self._training = mode
        if prev is None or prev != mode:
            self.on_mode_change(mode)

    def train(self) -> Layer:
        """Put the module in training mode and return ``self``.

        This sets ``self.training = True`` (triggering ``on_mode_change``) and
        allows chaining, e.g., ``layer.train()``."""
        self.training = True
        return self

    def eval(self) -> Layer:
        """Put the module in evaluation mode and return ``self``.

        This sets ``self.training = False`` (triggering ``on_mode_change``) and
        allows chaining, e.g., ``layer.eval()``."""
        self.training = False
        return self

    def on_mode_change(self, training: bool) -> None:
        """Subclass hook called when `.training` flips.
        Override in layers like BatchNorm/Dropout if needed.
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> tuple[Tensor, ...]:
        """Return trainable parameters (possibly empty)."""
        raise NotImplementedError

    def named_buffers(self) -> dict[str, Tensor | np.ndarray]:
        """Return mapping of buffer-name -> Tensor/ndarray (non-trainable). Default: {}."""
        return {}

    def _validate_contract(self) -> None:
        validate_layer_contract(self, tensor_type=Tensor)

    def apply_state(
        self,
        *,
        tunable: tuple[np.ndarray, ...] | list[np.ndarray] = (),
        buffers: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Default in-place state load: writes arrays into `parameters` and `named_buffers()` Tensors."""
        self._validate_contract()
        # write trainables in-order
        if tunable:
            if len(tunable) != len(self.parameters):
                raise ValueError(
                    f"{self.__class__.__name__}.apply_state expected {len(self.parameters)} tunable arrays; got {len(tunable)}"
                )
            for t, arr in zip(self.parameters, tunable):
                if arr is None:
                    continue
                t.data = np.asarray(arr, dtype=t.data.dtype)

        # write buffers by name (only if buffer is a Tensor)
        if buffers:
            for name, v in self.named_buffers().items():
                if name in buffers and buffers[name] is not None and isinstance(v, Tensor):
                    v.data = np.asarray(buffers[name], dtype=v.data.dtype)

class Affine(Layer):
    """Affine (linear) layer implementing Y = X @ W + b.

    This layer stores the weight matrix with shape (n, m) so that a forward pass
    can compute `X @ W` directly when `X.shape == (B, n)` and `W.shape == (n, m)`.

    If a pre-initialized `W` is provided, either orientation is accepted:
    `(fan_in, fan_out)` or `(fan_out, fan_in)`. In the latter case, it is
    transposed to `(fan_in, fan_out)` before being stored internally as `(n, m)`.

    Args:
        fan_in (int): Input feature dimension `n`.
        fan_out (int): Output feature dimension `m`.
        method (str, optional): Initialization method. Supported: `"xavier-glorot-normal"`.
            Defaults to `"xavier-glorot-normal"`.
        W (Tensor | None): Optional pre-initialized weight tensor. May be shaped
            `(fan_in, fan_out)` or `(fan_out, fan_in)`; it will be converted to internal
            `(n, m)` storage.
        b (Tensor | None): Optional pre-initialized bias tensor of shape `(fan_out,)`.
        seed (int | None): Optional RNG seed used when initializing parameters.
        bias (bool): Optional. True by default. Indicates whether to add the bias term to the matmul.

    Attributes:
        W (Tensor): Weight matrix stored as shape `(n, m)`.
        b (Tensor): Bias vector stored as shape `(m,)`.

    Raises:
        ValueError: If `method` is unknown, or if provided `W`/`b` shapes are incompatible.
    """

    def __init__(self,
                 fan_in: int,
                 fan_out: int, 
                 method: Literal["xavier-glorot-normal"] = "xavier-glorot-normal",
                 W: Tensor | None = None, 
                 b: Tensor | None = None,
                 *,
                 bias: bool = True,
                 seed: int | None = None):
        super().__init__()

        self.method = method
        self._rng, self.seed = rng_from_seed(seed)
        self.use_bias = bool(bias)

        try:
            init_fn = {"xavier-glorot-normal": xavier_glorot_normal}[method]
        except KeyError as e:
            raise ValueError(f"Unknown init method '{method}'") from e

        W_init = b_init = None
        if (W is None) or (self.use_bias and b is None):
            Wi, bi = init_fn(fan_in, fan_out, rng=self._rng)
            W_init = Wi.T          # (n, m)
            b_init = bi            # (m,)

        # W accepts either (fan_in, fan_out) or (fan_out, fan_in)
        if W is None:
            self.W = W_init
        else:
            if W.data.shape == (fan_in, fan_out):
                self.W = W
            elif W.data.shape == (fan_out, fan_in):
                self.W = Tensor(W.data.T, requires_grad=True)
            else:
                raise ValueError(
                    f"Incompatible W shape {W.data.shape}; expected {(fan_in, fan_out)} or {(fan_out, fan_in)}"
                )
        self.W.requires_grad = True # MAKE SURE GRADS ARE ALWAYS TRACKED

        if not self.use_bias:
            if b is not None:
                raise ValueError("Received 'b' but bias=False. Either pass bias=True or drop 'b'.")
            self.b = Tensor(np.zeros((fan_out,), dtype=self.W.dtype), requires_grad=False) # <-- NOTE `requires_grad=False`
        else:
            if b is None:
                self.b = b_init
            else:
                if b.shape != (fan_out,):
                    raise ValueError(f"Incompatible b shape {b.shape}; expected {(fan_out,)}")
                self.b = b
            self.b.requires_grad = True # MAKE SURE GRADS ARE ALWAYS TRACKED

        _logger.debug(
            "Affine initialized: seed=%s, fan_in=%d, fan_out=%d, method=%s, use_bias=%s, W.shape=%s, b.shape=%s req_grad_b=%s",
            self.seed, fan_in, fan_out, self.method, self.use_bias,
            getattr(self.W, "shape", None), getattr(self.b, "shape", None), getattr(self.b, "requires_grad", None)
        )

    def named_buffers(self) -> dict[str, np.ndarray]:
        """Return non-trainable metadata buffers.

        Returns:
            dict[str, np.ndarray]: A mapping with:
                - method: initialization method as bytes (NumPy np.bytes_)
                - seed: RNG seed as uint64 (0 if unset)
                - use_bias: whether the bias term is active (1) or disabled (0).

        """
        seed_val = 0 if self.seed is None else self.seed
        return {
            "method": np.array(self.method.encode("utf-8"), dtype=np.bytes_),
            "seed":   np.asarray(seed_val, dtype=np.uint64),
            "use_bias": np.asarray(int(self.use_bias), dtype=np.int8)
        }

    def apply_state(self, *, tunable=(), buffers=None) -> None:
        """Load weights/bias and optional metadata into the layer.

        Args:
            tunable: Iterable with one or two arrays ``(W, b)`` or ``(W,)``. ``W`` may be shaped
                ``(n, m)`` or ``(m, n)`` and is transposed if needed; ``b`` must be
                ``(m,)`` if present.
            buffers: Optional mapping that may include:
                - ``"seed"`` (int or array-like): resets the RNG used by initializers
                - ``"method"`` (bytes/str): initialization method name
                - ``"use_bias"`` (int): whether to use bias term or not

        Raises:
            ValueError: If the number or shapes of arrays are incompatible."""
        self._validate_contract()

        if buffers:
            if "use_bias" in buffers and buffers["use_bias"] is not None:
                prev = self.use_bias
                self.use_bias = bool(int(np.asarray(buffers["use_bias"]).item()))
                if not self.use_bias:
                    # disabling while keeping the object's identity. We zero and freeze it
                    if self.b is not None:
                        self.b.data[...] = 0.0
                        self.b.requires_grad = False
                elif not prev:
                    # At this branch: self.use_bias is True, but previously was False, so we want to track grads now.
                    # Also note that we preserve the Tensor's contents (just toggle the `requires_grad` field).
                    if self.b is not None:
                        self.b.requires_grad = True

            if "seed" in buffers and buffers["seed"] is not None:
                seed_val = int(np.asarray(buffers["seed"]).item())
                self.seed = seed_val
                self._rng, _ = rng_from_seed(seed_val)

            if "method" in buffers and buffers["method"] is not None:
                val = buffers["method"]
                if isinstance(val, np.ndarray):
                    val = val.item()
                if isinstance(val, (bytes, bytearray)):
                    val = val.decode("utf-8", "ignore")
                self.method = str(val)

        if tunable:
            expected = 2 if self.use_bias else 1
            if len(tunable) != expected:
                raise ValueError(f"Affine.apply_state expected {expected} arrays (W{', b' if self.use_bias else ''}); got {len(tunable)}")
            
            W_arr = np.asarray(tunable[0])
            n, m = self.W.data.shape
            if W_arr.shape == (n, m):
                self.W.data = W_arr.astype(self.W.data.dtype, copy=False)
            elif W_arr.shape == (m, n):
                self.W.data = W_arr.T.astype(self.W.data.dtype, copy=False)
            else:
                raise ValueError(f"Incompatible W shape {W_arr.shape}; expected {(n, m)} or {(m, n)}")

            if self.use_bias:
                b_arr = np.asarray(tunable[1])
                if b_arr.shape != self.b.data.shape:
                    raise ValueError(f"Incompatible b shape {b_arr.shape}; expected {self.b.data.shape}")
                self.b.data = b_arr.astype(self.b.data.dtype, copy=False)

    @property
    def parameters(self) -> tuple[Tensor, ...]:
        """Return the trainable parameters"""
        return (self.W, self.b) if self.use_bias else (self.W,)

    @staticmethod
    def _affine(X: np.ndarray, W: np.ndarray, b: np.ndarray, *, context: dict | None = None) -> np.ndarray:
        """Compute the affine map `Y = X @ W + b`.

        Args:
            X (np.ndarray): Input data of shape `(B, n)` (or `(n,)` if unbatched).
            W (np.ndarray): Weight matrix of shape `(n, m)`.
            b (np.ndarray): Bias vector of shape `(m,)` (broadcast to `(B, m)`).

        Returns:
            np.ndarray: Output array `Y` with shape `(B, m)` (or `(m,)` if unbatched).
        """
        # X: (B, n)
        # W: (n, m)
        # b: (m,) <-- is broadcast to (B, m)
        _logger.debug("Affine forward: X.shape=%s, W.shape=%s, b.shape=%s", X.shape, W.shape, b.shape)
        out = X @ W + b
        _logger.debug("Affine forward: Y.shape=%s", out.shape)

        # cache useful intermediates for backward reuse (avoid extra transposes):
        # keep it lazy to avoid any cost if grads are disabled
        _update_ctx(context, WT=lambda: W.T, XT=lambda: X.T)

        return out

    @staticmethod
    @_shape_safe_grad
    def _affine_grad(upstream_grad: np.ndarray, X: np.ndarray, W: np.ndarray, b: np.ndarray, *, context: dict | None = None):
        """Compute gradients of `Y = X @ W + b`.

        Gradients are computed w.r.t. inputs `(X, W, b)` given `upstream_grad = dL/dY`.

        Args:
            upstream_grad (np.ndarray): Upstream gradient `dL/dY` with shape `(B, m)` for
                batched `X` or `(m,)` for single-sample.
            X (np.ndarray): Input `X` with shape `(B, n)` or `(n,)`.
            W (np.ndarray): Weight matrix with shape `(n, m)`.
            b (np.ndarray): Bias vector with shape `(m,)`.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - `grad_X`: Same shape as `X` -> `(B, n)` or `(n,)`
                - `grad_W`: Shape `(n, m)`
                - `grad_b`: Shape `(m,)`

        Raises:
            ValueError: If `X` is not 1D or 2D.
        """
        _logger.debug(
            "Affine backward: upstream_grad.shape=%s, X.shape=%s, W.shape=%s, b.shape=%s",
            getattr(upstream_grad, "shape", None), getattr(X, "shape", None),
            getattr(W, "shape", None), getattr(b, "shape", None)
        )

        ctx = context if context is not None else {}
        WT = ctx.get("WT", W.T); WT = WT() if callable(WT) else (W.T if WT is None else WT)
        XT = ctx.get("XT", X.T); XT = XT() if callable(XT) else (X.T if XT is None else XT)

        if X.ndim == 1:
            # Single sample: X:(n,), upstream_grad:(m,)
            _logger.debug("Affine backward path: single sample")
            g = upstream_grad
            grad_X = g @ WT
            grad_W = np.outer(X, g)
            grad_b = g
        elif X.ndim == 2:
            # Batched: X:(B,n), upstream_grad:(B,m)
            _logger.debug("Affine backward path: batched")
            G = upstream_grad
            grad_X = G @ WT          # (..., n)
            grad_W = XT @ G          # (n, m)
            grad_b = G.sum(axis=0)
        else:
            raise ValueError(f"X must be 1D or 2D, got {X.ndim}D")
        _logger.debug("Affine backward: grad_X.shape=%s, grad_W.shape=%s, grad_b.shape=%s",
                      getattr(grad_X, "shape", None), getattr(grad_W, "shape", None),
                      getattr(grad_b, "shape", None))
        return grad_X, grad_W, grad_b

    def __call__(self, X: Tensor) -> Tensor:
        """Apply the affine transform to input tensor `X`.

        Validates input dimensionality and delegates to `TensorValuedFunction`
        with `_affine` as the forward and `_affine_grad` as the backward.

        Args:
            X (Tensor): Input tensor with `X.data.ndim in {1, 2}`. If 1D, must
                have shape `(n,)`; if 2D, must have shape `(B, n)` where
                `n == self.W.shape[0]`.

        Returns:
            Tensor: Output tensor of shape `(m,)` for 1D input or `(B, m)` for 2D input.

        Raises:
            ValueError: If `X.data` is not 1D or 2D, or if the last dimension
                does not match `self.W.shape[0]`.
        """
        _logger.debug("Affine __call__: X.data.ndim=%s, X.data.shape=%s, W.shape=%s",
                      getattr(X.data, "ndim", None), getattr(X.data, "shape", None),
                      getattr(self.W, "shape", None))
        if X.data.ndim == 1:
            if X.data.shape != (self.W.shape[0],):
                raise ValueError(
                    f"Incompatible dimensions. Expected a {self.W.shape[0]}-dimensional tensor; "
                    f"received {X.data.shape[0]}"
                )
        elif X.data.ndim == 2:
            if X.data.shape[1] != self.W.shape[0]:
                raise ValueError(
                    f"Incompatible dims. Expected last dim {self.W.shape[0]}; "
                    f"received {X.data.shape[1]}"
                )
        else:
            raise ValueError(f"X must be 1D or 2D, got {X.data.ndim}D")
        out = TensorValuedFunction(self._affine, self._affine_grad)(X, self.W, self.b)
        _logger.debug("Affine __call__: output Tensor created")
        return out

class Dropout(Layer):
    """Inverted Dropout layer.

    During training, zeros out each element of the input with probability `p`
    and scales the survivors by `1/(1-p)` so that the expected activation
    stays constant. In eval mode, this is an identity map.

    Args:
        p (float): Drop probability in [0, 1]. Defaults to 0.5.
        seed (int | None): Optional RNG seed for reproducibility.
        training (bool): If True, applies dropout; otherwise acts as identity.
                         Defaults to True.
    """

    def __init__(self, p: float = 0.5, *, seed: int | None = None, training: bool = True) -> None:
        super().__init__(training=training)
        if not (0.0 <= float(p) <= 1.0):
            raise ValueError(f"Dropout p must be in [0, 1], got {p}")
        self.p: float = float(p)
        self._rng, self.seed = rng_from_seed(seed)
        _logger.debug("Dropout initialized: p=%.4f, training=%s, seed=%s",
              self.p, self.training, self.seed)

    def on_mode_change(self, training: bool):
        """Hook invoked when ``training`` flips.

        Used here only for logging; no state is altered beyond the mode itself."""
        if training:
            _logger.debug("Dropout set to training mode")
        else:
            _logger.debug("Dropout set to inference mode")

    @property
    def parameters(self) -> tuple[Tensor, ...]:
        return ()  # no trainables

    def named_buffers(self) -> dict[str, np.ndarray]:
        """Return non-trainable buffers for serialization.

        Returns:
            dict[str, np.ndarray]: A mapping with:
                - ``"p"``: drop probability as ``float64``
                - ``"seed"``: RNG seed as ``uint64`` (0 if unset)
                - ``"training"``: mode flag as ``int8`` (1 train, 0 eval)
        """
        seed_val = 0 if self.seed is None else self.seed
        return {
            "p":        np.asarray(float(self.p), dtype=np.float64),
            "seed":     np.asarray(seed_val, dtype=np.uint64),
            "training": np.asarray(int(self.training), dtype=np.int8),
        }
    
    def apply_state(self, *, tunable=(), buffers=None) -> None:
        """Restore dropout configuration from buffers.

        Args:
            tunable: Unused (dropout has no trainable parameters).
            buffers: Optional mapping with keys:
                - ``"p"`` (float): drop probability in ``[0, 1]``
                - ``"training"`` (int/bool): set module mode
                - ``"seed"`` (int): resets the RNG used to sample masks"""
        self._validate_contract()
        if buffers:
            if "p" in buffers:
                self.p = float(np.asarray(buffers["p"]).item())
            if "training" in buffers:
                self.training = bool(int(np.asarray(buffers["training"]).item()))
            if "seed" in buffers:
                seed_val = int(np.asarray(buffers["seed"]).item())
                self.seed = seed_val
                self._rng, _ = rng_from_seed(seed_val)

    @staticmethod
    def _dropout(X: np.ndarray, mask: np.ndarray, scale: np.ndarray, *, context: dict | None = None) -> np.ndarray:
        """Forward: elementwise masked scaling."""
        _logger.debug(
            "Dropout forward: X.shape=%s, mask.shape=%s, scale=%s",
            getattr(X, "shape", None), getattr(mask, "shape", None), getattr(scale, "item", lambda: scale)()
        )

        _update_ctx(context, mask=mask, scale=scale)

        return X * (mask * scale) 

    @staticmethod
    @_shape_safe_grad
    def _dropout_grad(upstream_grad: np.ndarray, X: np.ndarray, mask: np.ndarray, scale: np.ndarray, *, context: dict | None = None):
        """Backward: dL/dX = upstream * mask * scale. No grads for mask/scale."""
        _logger.debug(
            "Dropout backward: upstream_grad.shape=%s, X.shape=%s, mask.shape=%s, scale=%s",
            getattr(upstream_grad, "shape", None), getattr(X, "shape", None),
            getattr(mask, "shape", None), getattr(scale, "item", lambda: scale)()
        )
        # elementwise upstream mult is by the same logic as for, say, relu
        grad_X = upstream_grad * (mask * scale) # (mask * scale) is the local grad
        # mask/scale are non-differentiable constants for this op
        return grad_X, None, None

    def __call__(self, X: Tensor) -> Tensor:
        """Apply dropout to `X` in training mode; identity in eval mode.

        Supports 1D `(n,)` and 2D `(B, n)` inputs.
        """
        if not isinstance(X, Tensor):
            raise TypeError(f"Dropout expects a Tensor, got {type(X)}")

        x = X.data
        if x.ndim not in (1, 2):
            raise ValueError(f"Dropout only supports 1D/2D inputs, got {x.ndim}D")

        # Eval mode or p == 0 -> identity
        if (not self.training) or (self.p <= 0.0):
            _logger.debug("Dropout passthrough (eval mode or p<=0).")
            return X

        keep_p = 1.0 - self.p
        if keep_p <= 0.0:
            # degenerate case: drop everything
            _logger.warning("Dropout p=1.0: output will be all zeros.")
            mask_arr = np.zeros_like(x, dtype=x.dtype)
            scale_arr = np.asarray(1.0, dtype=x.dtype)  # irrelevant cuz output is zero anyway
        else:
            # elementwise Bernoulli mask and inverted scaling (note we sample uniformly between 0 & 1)
            mask_arr = (self._rng.random(x.shape) < keep_p).astype(x.dtype, copy=False)
            scale_arr = np.asarray(1.0 / keep_p, dtype=x.dtype)

        # wrap mask/scale as non-trainable Tensors so the autograd context saves them
        mask = Tensor(mask_arr, requires_grad=False)
        scale = Tensor(scale_arr, requires_grad=False)

        out = TensorValuedFunction(self._dropout, self._dropout_grad)(X, mask, scale)
        _logger.debug("Dropout __call__: output Tensor created with shape=%s", getattr(out.data, "shape", None))
        return out

class BatchNorm1d(Layer):
    """Batch Normalization for 2D inputs shaped (B, F).

    Normalizes each feature across the batch:
        y = gamma * (x - mu) / sqrt(var + eps) + beta

    Running statistics (EMA) are updated only in training mode:
        running = (1 - momentum) * running + momentum * batch_stat

    Args:
        num_features: Feature dimension F.
        eps: Small constant for numerical stability.
        momentum: EMA coefficient for running stats (PyTorch-style).
        gamma, beta: Optional trainable scale/shift (shape (F,)).
        running_variance, running_mean: Optional buffers to resume from.
        training: Initial mode.
    """

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        gamma: Tensor | None = None,
        beta: Tensor | None = None,
        running_variance: Tensor | None = None,
        running_mean: Tensor | None = None,
        training: bool = True,
    ) -> None:
        super().__init__(training=training)

        _logger.debug("BN1d.__init__: F=%d, eps=%g, momentum=%.3f, has_gamma=%s, has_beta=%s, "
                      "has_runvar=%s, has_runmean=%s, training=%s",
                      int(num_features), float(eps), float(momentum),
                      gamma is not None, beta is not None,
                      running_variance is not None, running_mean is not None, bool(training))

        self.num_features = int(num_features)
        self.momentum = float(momentum)

        # -- tuned --------------------------------------------------
        self.gamma = Tensor(np.ones((self.num_features,), dtype=np.float64)
                            if gamma is None else gamma.data,
                            requires_grad=True)

        self.beta  = Tensor(np.zeros((self.num_features,), dtype=np.float64)
                            if beta  is None else beta.data,
                            requires_grad=True)

        self.eps = Tensor(eps, requires_grad=False)
        # -------------------------------------------------------------

        # -- accumulated ----------------------------------------------
        self.running_variance = Tensor(np.ones((self.num_features,),  dtype=np.float64)
                                       if running_variance is None else running_variance.data,
                                       requires_grad=False)

        self.running_mean = Tensor(np.zeros((self.num_features,), dtype=np.float64)
                                   if running_mean is None else running_mean.data,
                                   requires_grad=False)
        # -------------------------------------------------------------

        _logger.debug("BN1d.__init__: gamma.shape=%s, beta.shape=%s, run_mean.shape=%s, run_var.shape=%s",
                      self.gamma.data.shape, self.beta.data.shape,
                      self.running_mean.data.shape, self.running_variance.data.shape)

    def on_mode_change(self, training: bool):
        """Hook invoked when ``training`` flips.

        BatchNorm behavior changes between using batch stats (train) and running
        stats (eval); this implementation logs the change."""
        if training:
            _logger.debug("BatchNorm1d set to training mode")
        else:
            _logger.debug("BatchNorm1d set to inference mode")

    @property
    def parameters(self) -> tuple[Tensor, ...]:
        """Return the trainable affine parameters.

        Returns:
            tuple[Tensor, Tensor]: ``(gamma, beta)`` of shape ``(F,)`` each."""
        return (self.gamma, self.beta)

    def named_buffers(self) -> dict[str, Tensor | np.ndarray]:
        """Return BN buffers and persisted configuration/state metadata.

        Returns:
            dict[str, Tensor | np.ndarray]: Mapping with:
                - ``"running_mean"``: EMA of per-feature means, shape ``(F,)``
                - ``"running_variance"``: EMA of per-feature variances, shape ``(F,)``
                - ``"eps"``: numerical stability constant (scalar Tensor)
                - ``"momentum"``: EMA coefficient as float64 scalar
                - ``"training"``: mode flag as int8 scalar (1 train, 0 eval)
                - ``"num_features"``: expected feature dimension as int64 scalar."""
        return {
            "running_mean": self.running_mean,
            "running_variance": self.running_variance,
            "eps": self.eps,
            "momentum": np.asarray(float(self.momentum), dtype=np.float64),
            "training": np.asarray(int(self.training), dtype=np.int8),
            "num_features": np.asarray(int(self.num_features), dtype=np.int64),
        }

    def apply_state(self, *, tunable=(), buffers=None) -> None:
        """Restore BN parameters and buffers from checkpoint payloads."""
        self._validate_contract()

        if buffers:
            # Restore Tensor buffers first: running stats + eps
            super().apply_state(tunable=(), buffers=buffers)

            if "num_features" in buffers and buffers["num_features"] is not None:
                loaded_f = int(np.asarray(buffers["num_features"]).item())
                if loaded_f != self.num_features:
                    raise ValueError(
                        f"BatchNorm1d num_features mismatch: checkpoint={loaded_f}, layer={self.num_features}"
                    )

            if "momentum" in buffers and buffers["momentum"] is not None:
                self.momentum = float(np.asarray(buffers["momentum"]).item())

            if "training" in buffers and buffers["training"] is not None:
                self.training = bool(int(np.asarray(buffers["training"]).item()))

        if tunable:
            if len(tunable) != 2:
                raise ValueError(f"BatchNorm1d.apply_state expected 2 arrays (gamma, beta); got {len(tunable)}")

            gamma_arr = np.asarray(tunable[0])
            beta_arr = np.asarray(tunable[1])
            if gamma_arr.shape != self.gamma.data.shape:
                raise ValueError(f"Incompatible gamma shape {gamma_arr.shape}; expected {self.gamma.data.shape}")
            if beta_arr.shape != self.beta.data.shape:
                raise ValueError(f"Incompatible beta shape {beta_arr.shape}; expected {self.beta.data.shape}")

            self.gamma.data = gamma_arr.astype(self.gamma.data.dtype, copy=False)
            self.beta.data = beta_arr.astype(self.beta.data.dtype, copy=False)

    def __call__(self, X: Tensor) -> Tensor:
        """Apply BN over the batch axis for (B, F) input.

        In training:
            - compute per-feature batch mean/var (axis=0)
            - update running stats via EMA
            - normalize using batch stats

        In eval:
            - normalize using running stats only
        """ 
        x = X.data
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(f"BatchNorm1d expects input of shape (B, {self.num_features}); got {x.shape}")

        _logger.debug("BN1d.__call__: training=%s, X.shape=%s", self.training, x.shape)

        if self.training:
            mu  = general_math.mean(X, axis=0)   # (F,)
            var = general_math.variance(X, axis=0)  # (F,)
            _logger.debug("BN1d.__call__: batch mu.shape=%s, var.shape=%s", mu.data.shape, var.data.shape)

            # EMA update: new = (1 - m)*old + m*current  -> ewma(old, current, beta=1-m)
            self.running_mean.data = general_math.ewma(self.running_mean.data,     mu.data,  beta=1.0 - self.momentum)
            self.running_variance.data = general_math.ewma(self.running_variance.data, var.data, beta=1.0 - self.momentum)
            _logger.debug("BN1d.__call__: updated running stats (momentum=%.3f)", self.momentum)

            used_mu, used_var = mu, var
        else:
            used_mu, used_var = self.running_mean, self.running_variance
            _logger.debug("BN1d.__call__: using running stats")

        X_hat = (X - used_mu) / sqrt(used_var + self.eps)
        out = X_hat * self.gamma + self.beta
        _logger.debug("BN1d.__call__: out.shape=%s", getattr(out.data, "shape", None))
        return out

class LayerNorm1d(Layer):

    def __init__(self,
                num_features: int,
                *,
                gamma: Tensor | None = None,
                beta: Tensor | None = None,
                eps: float = 1e-5,
                bias: bool = True,
                training: bool = True):
        
        super().__init__(training=training)

        if num_features <= 0:
            raise ValueError(f"num_features must be > 0, got {num_features}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.num_features = int(num_features)
        self.use_bias = bool(bias)

        if gamma is None:
            self.gamma = Tensor(np.ones((self.num_features,), dtype=np.float64), requires_grad=True)
        else:
            if gamma.shape != (self.num_features,):
                raise ValueError(f"Incompatible gamma shape {gamma.shape}; expected {(self.num_features,)}")
            self.gamma = gamma
            self.gamma.requires_grad = True

        if not self.use_bias:
            if beta is not None:
                raise ValueError("Received 'beta' but bias=False. Either pass bias=True or drop 'beta'.")
            self.beta = Tensor(np.zeros((self.num_features,), dtype=self.gamma.dtype), requires_grad=False)
        else:
            if beta is None:
                self.beta = Tensor(np.zeros((self.num_features,), dtype=np.float64), requires_grad=True)
            else:
                if beta.shape != (self.num_features,):
                    raise ValueError(f"Incompatible beta shape {beta.shape}; expected {(self.num_features,)}")
                self.beta = beta
                self.beta.requires_grad = True

        self.eps = Tensor(eps, requires_grad=False)
        _logger.debug(
            "LayerNorm1d initialized: num_features=%d, eps=%g, use_bias=%s, gamma.shape=%s, beta.shape=%s req_grad_beta=%s",
            self.num_features, float(eps), self.use_bias,
            getattr(self.gamma, "shape", None), getattr(self.beta, "shape", None), getattr(self.beta, "requires_grad", None)
        )

    def on_mode_change(self, training: bool):
        _logger.debug("LayerNorm1d mode changed: training=%s", bool(training))

    @property
    def parameters(self) -> tuple[Tensor, ...]:
        return (self.gamma, self.beta) if self.use_bias else (self.gamma,)

    def named_buffers(self) -> dict[str, Tensor | bool]:
        return {
            "eps": self.eps,
            "use_bias": self.use_bias
        }

    def apply_state(self, *, tunable=(), buffers=None) -> None:
        self._validate_contract()

        if buffers:
            # Delegate Tensor buffer restore (eps) to base implementation.
            super().apply_state(tunable=(), buffers=buffers)
            if "use_bias" in buffers and buffers["use_bias"] is not None:
                prev = self.use_bias
                self.use_bias = bool(int(np.asarray(buffers["use_bias"]).item()))
                if not self.use_bias:
                    self.beta.data[...] = 0.0
                    self.beta.requires_grad = False
                elif not prev:
                    self.beta.requires_grad = True

        if tunable:
            expected = 2 if self.use_bias else 1
            if len(tunable) != expected:
                raise ValueError(
                    f"LayerNorm1d.apply_state expected {expected} arrays "
                    f"(gamma{', beta' if self.use_bias else ''}); got {len(tunable)}"
                )

            gamma_arr = np.asarray(tunable[0])
            if gamma_arr.shape != self.gamma.data.shape:
                raise ValueError(f"Incompatible gamma shape {gamma_arr.shape}; expected {self.gamma.data.shape}")
            self.gamma.data = gamma_arr.astype(self.gamma.data.dtype, copy=False)

            if self.use_bias:
                beta_arr = np.asarray(tunable[1])
                if beta_arr.shape != self.beta.data.shape:
                    raise ValueError(f"Incompatible beta shape {beta_arr.shape}; expected {self.beta.data.shape}")
                self.beta.data = beta_arr.astype(self.beta.data.dtype, copy=False)

    def __call__(self, X: Tensor) -> Tensor:
        if not isinstance(X, Tensor):
            raise TypeError(f"LayerNorm1d expects a Tensor, got {type(X)}")
        if X.ndim < 1:
            raise ValueError(f"LayerNorm1d expects input with at least 1 dimension, got shape {X.shape}")
        if X.shape[-1] != self.num_features:
            raise ValueError(
                f"LayerNorm1d expects last dimension == num_features ({self.num_features}), got {X.shape}"
            )

        _logger.debug("LayerNorm1d __call__: training=%s, X.shape=%s", self.training, X.shape)
        mu = general_math.mean(X, axis=-1).unsqueeze(-1) # [a, b, ..., y, z] -> [a, b, ..., y, 1]
        sig2 = general_math.variance(X, axis=-1).unsqueeze(-1) # [a, b, ..., y, z] -> [a, b, ..., y, 1]
        X_hat = (X - mu) / sqrt(sig2+self.eps)
        if self.use_bias:
            out = self.gamma * X_hat + self.beta
        else:
            out = self.gamma * X_hat
        _logger.debug("LayerNorm1d __call__: out.shape=%s", getattr(out, "shape", None))
        return out

class Embedding(Layer):
    """Learned lookup table: returns rows of `W` for integer indices.

    Args:
        V (int): Vocabulary size (number of rows).
        D (int): Embedding size (number of columns).
        pad_idx (int | None): Optional padding index in `[0, V)`. If provided,
            that row is initialized to zeros and excluded from gradient updates
            (i.e., it remains a fixed "no-meaning" vector). Stored in checkpoints
            as `"padding_idx"`.
        method (str): Initialization method for `W`. Supported: `"xavier-glorot-normal"`.
        W (Tensor | None): Optional pre-initialized weight tensor of shape `(V, D)`.
            If provided, its shape is validated and used as-is.
        training (bool): Initial module mode flag.
        seed (int | None): Optional RNG seed used when initializing `W`.

    Attributes:
        W (Tensor): Embedding table of shape `(V, D)`.
        padding_idx (int | None): Index treated as padding (no gradient updates).

    Raises:
        ValueError: If `V <= 0` or `D <= 0`, if `method` is unknown, or if provided
            `W` has shape different from `(V, D)`.
    """

    def __init__(
        self,
        V: int,
        D: int,
        *,
        pad_idx: int | None = None,
        method: Literal["xavier-glorot-normal"] = "xavier-glorot-normal",
        W: Tensor | None = None,
        training: bool = True,
        seed: int | None = None
    ) -> None:
        super().__init__(training=training)

        self.method = method
        self._rng, self.seed = rng_from_seed(seed)

        init_fn = {
            "xavier-glorot-normal": xavier_glorot_normal
        }[method]

        if V <= 0 or D <= 0:
            raise ValueError(f"num_embeddings and embedding_dim must be positive, got {V=}, {D=}")
        self.V = int(V)
        self.D = int(D)

        self.padding_idx = None if pad_idx is None else int(pad_idx)

        if W is None:
            # NOTE: xavier_glorot_normal(fan_in, fan_out) -> W.shape == (fan_out, fan_in)
            # We need (V, D), so pass fan_in=D, fan_out=V.
            W_init, _ = init_fn(self.D, self.V, rng=self._rng)  # -> (V, D) as a Tensor
            if self.padding_idx is not None:
                if not (0 <= self.padding_idx < self.V):
                    raise ValueError(f"padding_idx must be in [0, {self.V}), got {self.padding_idx}")
                W_init.data[self.padding_idx, :] = 0.0
            self.W = W_init
        else:
            # enforce the expected shape
            if W.data.shape != (self.V, self.D):
                raise ValueError(f"W shape must be {(self.V, self.D)}, got {W.data.shape}")
            self.W = W
        self.W.requires_grad = True # MAKE SURE GRADS ARE ALWAYS TRACKED
        
        _logger.debug(
            "Embedding initialized: V=%d, D=%d, pad_idx=%s, seed=%s",
            self.V, self.D, self.padding_idx, self.seed
        )

    @property
    def parameters(self) -> tuple[Tensor, ...]:
        """Return the trainable embedding table.

        Returns:
            tuple[Tensor, ...]: A single-element tuple ``(W,)`` where
            ``W.shape == (V, D)``."""
        return (self.W,)

    def named_buffers(self) -> dict[str, np.ndarray]:
        """Return non-trainable buffers to persist in full-state checkpoints.

        Returns:
            dict[str, np.ndarray]: A mapping with:
                - "padding_idx": int64 (-1 if None)
                - "seed": uint64 RNG seed used for initialization (0 if unset)
                - "method": bytes (NumPy ``np.bytes_``) initialization method name
        """
        pid = -1 if self.padding_idx is None else int(self.padding_idx)
        seed_val = 0 if self.seed is None else self.seed
        return {
            "padding_idx": np.asarray(pid, dtype=np.int64),
            "seed":        np.asarray(seed_val, dtype=np.uint64),
            "method":      np.array(self.method.encode("utf-8"), dtype=np.bytes_),
        }

    def apply_state(self, *, tunable=(), buffers=None) -> None:
        """Restore parameters and buffers for the embedding layer.

        Expects exactly one tunable array for the weights and validates its shape.

        Args:
            tunable (tuple[np.ndarray, ...] | list[np.ndarray]): Must contain a single
                array with shape `(V, D)` to load into `self.W`.
            buffers (dict[str, np.ndarray] | None): Optional buffers to restore:
                `"padding_idx"` (int or array-like), `"seed"` (int), and `"method"`
                (bytes/str). Types are normalized internally.

        Raises:
            ValueError: If the number of tunables is not 1 or if the provided weight
                array does not have shape `(V, D)`.
        """
        super().apply_state(tunable=(), buffers=buffers)

        if tunable:
            if len(tunable) != 1:
                raise ValueError(f"Embedding.apply_state expected 1 array (W); got {len(tunable)}")
            W_arr = np.asarray(tunable[0])
            if W_arr.shape != (self.V, self.D):
                raise ValueError(f"Incompatible W shape {W_arr.shape}; expected {(self.V, self.D)}")
            self.W.data = W_arr.astype(self.W.data.dtype, copy=False)

        if buffers:
            if "padding_idx" in buffers and buffers["padding_idx"] is not None:
                pid = int(np.asarray(buffers["padding_idx"]).item())
                self.padding_idx = None if pid < 0 else pid

            if "seed" in buffers and buffers["seed"] is not None:
                seed_val = int(np.asarray(buffers["seed"]).item())
                self.seed = seed_val
                self._rng, _ = rng_from_seed(seed_val)

            if "method" in buffers and buffers["method"] is not None:
                val = buffers["method"]
                if isinstance(val, np.ndarray): val = val.item()
                if isinstance(val, (bytes, bytearray)): val = val.decode("utf-8", "ignore")
                self.method = str(val)

    @staticmethod
    def _gather(idx: np.ndarray, W: np.ndarray, *, context: dict | None = None) -> np.ndarray:
        """Forward: out = W[idx] with shape idx.shape + (D,)."""
        # idx is (B, T), where B is batch dim and T is tokens (tokenizer's output)
        if idx.dtype.kind not in "iu":  # integers or unsigned
            idx = idx.astype(np.int64, copy=False)

        V = context.get("V", None)
        if V is None:
            raise RuntimeError("Missing the vocabulary size during the forward pass through "
                               "the `Embedding` layer; Check <Embedding>.__call__ method "
                               "to ensure the vocabulary size `V` is passed to the fwd context.")

        if (idx < 0).any() or (idx >= V).any():
            bad = int(idx[(idx < 0) | (idx >= V)][0])
            raise IndexError(f"Embedding index {bad} out of range [0, {V})")
        
        out = W[idx] # RECALL: idx = [ [2, 3, 1, 5] (say `5` is <PAD> token btw)
        #                              [0, 1, 5, 5]
        #                              [4, 5, 5, 5] ] and W is a (V, D) matrix
        # Then: out == W[idx] == W[ [2, 3, 1, 5], [0, 1, 5, 5], [4, 5, 5, 5] ]
        # == [ [EMB_2, EMB_3, EMB_1, EMB_5]
        #       [EMB_0, EMB_1, EMB_5, EMB_5]
        #       [EMB_4, EMB_5, EMB_5, EMB_5] ] and each EMB_i is a (D,)-dimensional Tensor
        # SO: out.shape == (B, T, D), where batch is "padded sentences", T is number of tokens
        #     in each sentence (including the <PAD>'s), and D is the dimensionality of each embedding.
        
        # Cache flattened indices (lazy) for backward and pass-through padding_idx if present.
        _update_ctx(context,
            idx_flat=lambda: idx.reshape(-1), # to avoid extra work if gradients are disabled
            padding_idx=context.get("padding_idx", None)
        )
        _logger.debug("Embedding forward: idx.shape=%s -> out.shape=%s", idx.shape, out.shape)

        return out
        
    @staticmethod
    @_shape_safe_grad
    def _gather_grad(upstream_grad: np.ndarray, idx: np.ndarray, W: np.ndarray, *, context: dict | None = None) -> tuple[None, np.ndarray]:
        """Backward:
            dW[i] += sum_over_positions(up[pos]) where idx[pos] == i
            d(idx) = None (indices are non-differentiable).
        """
        # RECALL: for tunable layers' parameters like W (different from the layers' input like direct or processed sample `x`),
        #         you sum up upstream gradients for individual entries of the parameter
        #         to get the overall contribution of each entry toward the loss across ALL of the samples from the batch.
        # That is why we sum.

        ctx = context if context is not None else {}
        I = ctx.get("idx_flat")
        I = I() if callable(I) else I
        if I is None:
            I = idx.reshape(-1)

        D = W.shape[1]
        G = upstream_grad.reshape(-1, D)
        # The reason we also do np.add.at is due to repeated indices across samples of the batch (repeated tokens across "sentences");
        # Otherwise dW[I] += G has unpredictable behavior.
        # According to NumPy docs: 
        # np.add.at method is equivalent to a[indices] += b, except that results are accumulated for elements
        # that are indexed more than once.
        dW = np.zeros_like(W)
        np.add.at(dW, I, G)

        # If padding_idx is tracked in context, zero its grad
        pad = ctx.get("padding_idx", None)
        if pad is not None:
            p = int(pad)
            if 0 <= p < dW.shape[0]:
                dW[p, :] = 0.0

        return None, dW # grads w.r.t idx and W

    def __call__(self, indices: Tensor) -> Tensor:
        """Lookup embeddings for integer indices. Returns (..., D)."""
        if not isinstance(indices, Tensor):
            raise TypeError(f"Embedding expects a Tensor of indices, got {type(indices)}")

        fn = TensorValuedFunction(self._gather, self._gather_grad)
        out = fn(indices, self.W, context={"padding_idx": self.padding_idx, "V": self.V})

        _logger.debug("Embedding __call__: indices.ndim=%d -> out.shape=%s",
                      getattr(indices.data, "ndim", None), getattr(out.data, "shape", None))
        
        return out

# *----------------------------------------------------*
#          CNN HELPER FUNCTIONS (PRIVATE SCOPE)
# *----------------------------------------------------*

def _L_out(L_in: int, p: int, d: int, kL: int, s: int) -> int:
    return floor((L_in + 2*p - d*(kL - 1) - 1) / s) + 1

def _unfold2d(
        X: np.ndarray, # (B, C, H, W)
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        pad_with: float,
        *,
        context: dict | None = None) -> np.ndarray:
    """Unfold a 4D tensor into sliding local blocks (im2col layout).

    Args:
        X: Input array of shape ``(B, C, H, W)``.
        kernel_size: ``(kH, kW)`` kernel size.
        stride: ``(sH, sW)`` step of the sliding window.
        padding: ``(pH, pW)`` zero-padding on both spatial sides.
        dilation: ``(dH, dW)`` spacing between kernel taps.
        pad_with: Constant value used for padding.
        context: Optional cache used by ``_unfold2d_grad``.

    Returns:
        np.ndarray: Unfolded columns with shape ``(B, C*kH*kW, H_out*W_out)``.
    """
    _logger.debug(
        "_unfold2d forward: X.shape=%s, kernel_size=%s, stride=%s, padding=%s, dilation=%s, pad_with=%s",
        getattr(X, "shape", None), kernel_size, stride, padding, dilation, pad_with
    )
    if X.ndim != 4:
        raise ValueError(f"_unfold2d expects X with shape (B, C, H, W), got {X.shape}")
    for name, v in (("kernel_size", kernel_size), ("stride", stride), ("padding", padding), ("dilation", dilation)):
        if not (isinstance(v, tuple) and len(v) == 2):
            raise TypeError(f"{name} must be a tuple[int, int], got {v!r}")

    B, C, H, W = X.shape # B - batch size, C - input channels, H, W - input spatial size
    kH, kW = kernel_size # kH, kW - kernel spatial size
    sH, sW = stride      # sH, sW - stride size along the H and W axes
    dH, dW = dilation    # dH, dW - kernel dilation along the H and W axes
    pH, pW = padding     # pH, pW - padding at the ends of the H and W axes
    if kH <= 0 or kW <= 0:
        raise ValueError(f"kernel_size entries must be > 0, got {kernel_size}")
    if sH <= 0 or sW <= 0:
        raise ValueError(f"stride entries must be > 0, got {stride}")
    if dH <= 0 or dW <= 0:
        raise ValueError(f"dilation entries must be > 0, got {dilation}")
    if pH < 0 or pW < 0:
        raise ValueError(f"padding entries must be >= 0, got {padding}")

    X_pad = np.pad(X, pad_width=((0, 0), (0, 0), (pH, pH), (pW, pW)), mode="constant", constant_values=pad_with)
    # ^^^ shaped: (B, C, pH+H+pH, pW+W+pW)

    H_out = _L_out(H, pH, dH, kH, sH) # H length
    W_out = _L_out(W, pW, dW, kW, sW) # W length
    _logger.debug("_unfold2d forward: padded_shape=%s, H_out=%d, W_out=%d", X_pad.shape, H_out, W_out)
    if H_out <= 0 or W_out <= 0:
        raise ValueError(
            "Invalid unfold2d output size. "
            f"Got H_out={H_out}, W_out={W_out} from input={(H, W)}, "
            f"kernel={(kH, kW)}, stride={(sH, sW)}, padding={(pH, pW)}, dilation={(dH, dW)}"
        )

    # auxiliary constants:
    K = kH * kW # - kernel elements per channel
    P = H_out * W_out # - number of sliding positions

    H_start = np.arange(H_out) * sH # top-left y positions per output row
    W_start = np.arange(W_out) * sW # top-left x positions per output col
    H_taps  = np.arange(kH) * dH    # y offsets inside kernel with dilation
    W_taps  = np.arange(kW) * dW    # x offsets inside kernel with dilation

    H_centers, W_centers = np.meshgrid(H_start, W_start, indexing="ij")   # kernel centers (H_out, W_out), (H_out, W_out)
    # ^^^ all output center/start coordinates flattened later to P
    H_koords, W_koords   = np.meshgrid(H_taps, W_taps, indexing="ij")     # kernel taps around each center
    # ^^^ all kernel offsets flattened later to K
    H_centers = H_centers.reshape(-1); W_centers = W_centers.reshape(-1)  # flat kernel centers ([P,], [P,])
    H_koords  = H_koords.reshape(-1);  W_koords  = W_koords.reshape(-1)   # flat kernel taps    ([K,], [K,])
    # Now expand across the C channels:
    #                 (K, 1) +          (1, P) = (K, P)
    finalH = H_koords[..., None] + H_centers[None, ...]
    finalW = W_koords[..., None] + W_centers[None, ...]
    # ^^^ for each kernel element and each output position, compute absolute y/x in padded image

    c_idx = np.repeat(np.arange(C), K)[:, None]  # (C*K, 1) -- channel ids 0..C-1, each repeated K times (one row per kernel tap);
    #          ^ ^ ^ fancy indexing needs explicit per-row channel coordinates.
    h_idx = np.tile(finalH, (C, 1))              # (C*K, P) -- we do it for each output location so there is P
    w_idx = np.tile(finalW, (C, 1))              # (C*K, P) -- and this is within both H and W axes

    # !NOTE: here's what h_idx and w_idx are: “Which pixel in the padded image did this column entry (in `col`) come from?”

    # interpret the (C*K, P) shape as "K kernel taps per C channels for each output location"
    cols = X_pad[:, c_idx, h_idx, w_idx]         # (B, C*K, P)

    _update_ctx(context,
        c_idx=lambda: c_idx, h_idx=lambda: h_idx, w_idx=lambda: w_idx, # note lazy caching
        pH=pH, pW=pW, in_shape=X.shape, out=None, overwrite=False
    )

    _logger.debug("_unfold2d forward: cols.shape=%s", cols.shape)
    return cols

# -=-=-=-=-=-=-=- EXTRA np.add.at INFORMATION -=-=-=-=-=-=-=-
# g = np.zeros((3, 3), dtype=int)
#
# h_idx = np.array([[0, 0, 1],
#                   [0, 1, 1]])
# w_idx = np.array([[0, 1, 0],
#                   [0, 0, 1]])
#
# up = np.array([[1, 2, 3],
#                [4, 5, 6]])
#
# np.add.at(g, (h_idx, w_idx), up)
#    ^^^
# This means: for every position (i, j) in up,
# do g[h_idx[i,j], w_idx[i,j]] += up[i,j]
# -=-=-=-=-=-=-=--=-=-=-=-=-=-=--=-=-=-=-=-=-=--=-=-=-=-=-=-=

@_shape_safe_grad
def _unfold2d_grad(upstream_grad: np.ndarray, X: np.ndarray, *, context: dict | None = None):
    """Backward pass for ``_unfold2d`` via scatter-add into padded input space.

    Args:
        upstream_grad: Gradient w.r.t unfolded output with shape ``(B, R, P)``.
        X: Original forward input of shape ``(B, C, H, W)``.
        context: Forward cache containing ``c_idx/h_idx/w_idx`` and ``pH/pW``.

    Returns:
        tuple[np.ndarray]: A single-element tuple containing ``dL/dX``.
    """
    _logger.debug(
        "_unfold2d_grad backward: upstream_grad.shape=%s, X.shape=%s",
        getattr(upstream_grad, "shape", None), getattr(X, "shape", None)
    )
    if X.ndim != 4:
        raise ValueError(f"_unfold2d_grad expects X with shape (B, C, H, W), got {X.shape}")
    ctx = context if context is not None else {}
    if any(k not in ctx for k in ("c_idx", "h_idx", "w_idx", "pH", "pW")):
        raise RuntimeError("_unfold2d_grad requires c_idx/h_idx/w_idx/pH/pW in context")
    c_idx = ctx["c_idx"]; c_idx = c_idx() if callable(c_idx) else c_idx          # (C*K, 1) (note the call operator due to lazy caching)
    h_idx = ctx["h_idx"]; h_idx = h_idx() if callable(h_idx) else h_idx          # (C*K, P)
    w_idx = ctx["w_idx"]; w_idx = w_idx() if callable(w_idx) else w_idx          # (C*K, P)
    pH, pW = int(ctx["pH"]), int(ctx["pW"])
    if c_idx.ndim == 1:
        c_idx = c_idx[:, None]
    if h_idx.shape != w_idx.shape:
        raise ValueError(f"h_idx and w_idx must have same shape, got {h_idx.shape} vs {w_idx.shape}")

    B, C, H, W = X.shape
    R, P = h_idx.shape
    _logger.debug(
        "_unfold2d_grad backward: c_idx.shape=%s, h_idx.shape=%s, w_idx.shape=%s, pH=%d, pW=%d",
        getattr(c_idx, "shape", None), getattr(h_idx, "shape", None), getattr(w_idx, "shape", None), pH, pW
    )
    if upstream_grad.shape != (B, R, P):
        raise ValueError(f"Expected upstream_grad shape {(B, R, P)}, got {upstream_grad.shape}")
    gpad = np.zeros((B, C, H + 2*pH, W + 2*pW), dtype=upstream_grad.dtype) # gpad must be the same shape as X_pad

    # scatter-add (inverse of gather)
    np.add.at(gpad, (slice(None), c_idx, h_idx, w_idx), upstream_grad)

    # !NOTE: very importantly upstream_grad is shaped as [B, R, P] -- it's a batch of matrices.
    # Each column P is a receptive field's output. For a fixed batch b and position p, upstream_grad[b, :, p]
    # is a vector of length R -- gradient for every pixel that participated in that receptive field.
    # Each row = same kernel tap across all positions. For fixed r: upstream_grad[b, r, :]
    # is gradient for that specific kernel tap across all sliding windows.

    # Logic:
    # What np.add.at is doing:
    # FWD:
    # cols[b, r, p] = X_pad[b, c_idx[r], h_idx[r, p], w_idx[r, p]]
    # BWD:
    # gpad[b, c_idx[r], h_idx[r,p], w_idx[r,p]] += upstream_grad[b, r, p]
    # ^^^  which is what np.add.at(gpad, (slice(None), c_idx, h_idx, w_idx), upstream_grad) does.
    # This is necessary because many (r,p) can hit the same source pixel (overlap),
    # so gradients must be summed. add.at guarantees accumulation for repeated indices.

    # crop padding back to input shape
    gX = gpad[:, :, pH:pH+H, pW:pW+W]

    _logger.debug("_unfold2d_grad backward: gX.shape=%s", gX.shape)
    return (gX,)

# =-=-=-=-=-=-=- EXTRA INFORMATION ON UNFOLD2D -=-=-=-=-=-=-=
# (Input image) X = 
# ┌───┬───┬───┬───┐
# │ a │ b │ c │ d │
# ├───┼───┼───┼───┤
# │ e │ f │ g │ h │
# ├───┼───┼───┼───┤
# │ i │ j │ k │ l │
# ├───┼───┼───┼───┤
# │ m │ n │ o │ p │
# └───┴───┴───┴───┘
# Each 2×2 window becomes one column:
# p=0        p=1        p=2
# [a b]      [b c]      [c d]
# [e f]      [f g]      [g h]

# p=3        p=4        p=5
# [e f]      [f g]      [g h]
# [i j]      [j k]      [k l]

# p=6        p=7        p=8
# [i j]      [j k]      [k l]
# [m n]      [n o]      [o p]
# Flatten each window → column vector of length R = kH·kW = 4:
# cols = (R × P)
#
#       p0  p1  p2  p3  p4  p5  p6  p7  p8
# r0    a   b   c   e   f   g   i   j   k
# r1    b   c   d   f   g   h   j   k   l
# r2    e   f   g   i   j   k   m   n   o
# r3    f   g   h   j   k   l   n   o   p
# ^ ^ ^ -- Each entry came from one pixel in X.
#
# Now add batch + channels:
# If you have:
#   •	B images
#   •	C channels
# R = C × kH × kW
# P = H_out × W_out
# Then, cols.shape = (B, R, P), that is,
# Batch 0 → (R × P) matrix
# Batch 1 → (R × P) matrix
# Batch 2 → (R × P) matrix
# ...
#
# Now imagine gradients coming back from later layers:
# upstream_grad[b] =
#        p0    p1    p2    p3    p4    p5    p6    p7    p8
# r0   g00   g01   g02   g03   g04   g05   g06   g07   g08
# r1   g10   g11   g12   g13   g14   g15   g16   g17   g18
# r2   g20   g21   g22   g23   g24   g25   g26   g27   g28
# r3   g30   g31   g32   g33   g34   g35   g36   g37   g38

# ^ ^ ^ -- Each value must go back to the pixel it originally came from.
# Example:
# 	•	g₀₀ came from pixel a
# 	•	g₁₀ came from pixel b
# 	•	g₂₀ came from pixel e
# 	•	g₃₀ came from pixel f
#
# But pixel f appears in multiple windows → multiple grads hit it.
# So backward does:
# grad[f] += g₃₀ + g₁₁ + g₀₄ + ...
# Now, each 'f' pixel is identified by h and w _idx, so we do:
# grad[h_idx[r,p], w_idx[r,p]] += upstream_grad[r, p]
# 
# The following also helps understand the idea:
# upstream_grad is the same shape as cols, obviously
# So, upstream_grad[r, p] is contributions of each (r, p)th entry of cols.
# But here's the kicker: h_idx[r,p] gives all the heights where the pixel responsible
# for the (r, p) grad entry is located in the X_pad image height-wise, while w_idx[r,p] tells us
# the same but width-wise, so we take all the grad entries (contributions of this pixel) and add
# them together so that the overall shape matches the shape of the original (folded) image.
# -=-=-=-=-=-=-=--=-=-=-=-=-=-=--=-=-=-=-=-=-=--=-=-=-=-=-=-=

# *----------------------------------------------------*
def _unfold1d(
        X: np.ndarray, # (B, C, L)
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        pad_with: float,
        *,
        context: dict | None = None) -> np.ndarray:
    """Unfold a 3D tensor into 1D sliding windows.

    Args:
        X: Input array of shape ``(B, C, L)``.
        kernel_size: Kernel length ``kL``.
        stride: Sliding stride ``sL``.
        padding: Symmetric padding ``pL``.
        dilation: Kernel dilation ``dL``.
        pad_with: Constant value used for padding.
        context: Optional cache used by ``_unfold1d_grad``.

    Returns:
        np.ndarray: Unfolded columns with shape ``(B, C*kL, L_out)``.
    """
    _logger.debug(
        "_unfold1d forward: X.shape=%s, kernel_size=%s, stride=%s, padding=%s, dilation=%s, pad_with=%s",
        getattr(X, "shape", None), kernel_size, stride, padding, dilation, pad_with
    )
    if X.ndim != 3:
        raise ValueError(f"_unfold1d expects X with shape (B, C, L), got {X.shape}")
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")
    if dilation <= 0:
        raise ValueError(f"dilation must be > 0, got {dilation}")
    if padding < 0:
        raise ValueError(f"padding must be >= 0, got {padding}")

    B, C, L = X.shape
    kL = kernel_size
    sL = stride
    dL = dilation
    pL = padding

    X_pad = np.pad(
        X,
        pad_width=((0, 0), (0, 0), (pL, pL)),
        mode="constant",
        constant_values=pad_with
    )

    L_out = _L_out(L, pL, dL, kL, sL)
    _logger.debug("_unfold1d forward: padded_shape=%s, L_out=%d", X_pad.shape, L_out)
    if L_out <= 0:
        raise ValueError(
            "Invalid unfold1d output size. "
            f"Got L_out={L_out} from input={L}, kernel={kL}, stride={sL}, padding={pL}, dilation={dL}"
        )

    K = kL
    P = L_out

    L_start = np.arange(L_out) * sL
    L_taps = np.arange(kL) * dL

    finalL = L_taps[:, None] + L_start[None, :]  # (K, P)

    c_idx = np.repeat(np.arange(C), K)[:, None]  # (C*K, 1)
    l_idx = np.tile(finalL, (C, 1))              # (C*K, P)

    cols = X_pad[:, c_idx, l_idx]                # (B, C*K, P)

    _update_ctx(
        context,
        c_idx=lambda: c_idx,
        l_idx=lambda: l_idx,
        pL=pL,
        in_shape=X.shape,
        out=None,
        overwrite=False
    )

    _logger.debug("_unfold1d forward: cols.shape=%s", cols.shape)
    return cols

@_shape_safe_grad
def _unfold1d_grad(upstream_grad: np.ndarray, X: np.ndarray, *, context: dict | None = None):
    """Backward pass for ``_unfold1d`` via scatter-add into padded length axis.

    Args:
        upstream_grad: Gradient w.r.t unfolded output with shape ``(B, R, P)``.
        X: Original forward input of shape ``(B, C, L)``.
        context: Forward cache containing ``c_idx/l_idx`` and ``pL``.

    Returns:
        tuple[np.ndarray]: A single-element tuple containing ``dL/dX``.
    """
    _logger.debug(
        "_unfold1d_grad backward: upstream_grad.shape=%s, X.shape=%s",
        getattr(upstream_grad, "shape", None), getattr(X, "shape", None)
    )
    if X.ndim != 3:
        raise ValueError(f"_unfold1d_grad expects X with shape (B, C, L), got {X.shape}")
    ctx = context if context is not None else {}
    if any(k not in ctx for k in ("c_idx", "l_idx", "pL")):
        raise RuntimeError("_unfold1d_grad requires c_idx/l_idx/pL in context")
    c_idx = ctx["c_idx"]; c_idx = c_idx() if callable(c_idx) else c_idx
    l_idx = ctx["l_idx"]; l_idx = l_idx() if callable(l_idx) else l_idx
    pL = int(ctx["pL"])
    if c_idx.ndim == 1:
        c_idx = c_idx[:, None]

    B, C, L = X.shape
    R, P = l_idx.shape
    _logger.debug(
        "_unfold1d_grad backward: c_idx.shape=%s, l_idx.shape=%s, pL=%d",
        getattr(c_idx, "shape", None), getattr(l_idx, "shape", None), pL
    )
    if upstream_grad.shape != (B, R, P):
        raise ValueError(f"Expected upstream_grad shape {(B, R, P)}, got {upstream_grad.shape}")
    gpad = np.zeros((B, C, L + 2*pL), dtype=upstream_grad.dtype)

    np.add.at(gpad, (slice(None), c_idx, l_idx), upstream_grad)

    gX = gpad[:, :, pL:pL+L]

    _logger.debug("_unfold1d_grad backward: gX.shape=%s", gX.shape)
    return (gX,)

# *----------------------------------------------------*
#          CNN HELPER FUNCTIONS (PUBLIC SCOPE)
# *----------------------------------------------------*

def unfold2d(X: Tensor,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        pad_with: float = 0.0):
    """Extract sliding 2D patches from a tensor (im2col-style).

    For an input ``X`` with shape ``(B, C, H, W)``, this returns a tensor of
    shape ``(B, C*kH*kW, H_out*W_out)``, where:

    - ``H_out = floor((H + 2*pH - dH*(kH - 1) - 1) / sH) + 1``
    - ``W_out = floor((W + 2*pW - dW*(kW - 1) - 1) / sW) + 1``

    Args:
        X: Input tensor of shape ``(B, C, H, W)``.
        kernel_size: Kernel size ``(kH, kW)``.
        stride: Sliding stride ``(sH, sW)``.
        padding: Symmetric padding ``(pH, pW)`` on spatial dimensions.
        dilation: Dilation ``(dH, dW)`` for kernel taps.
        pad_with: Constant used to fill padded regions.

    Returns:
        Tensor: Tensor of shape ``(B, C*kH*kW, H_out*W_out)`` containing
        flattened receptive fields.

    Notes:
        The output layout is convenient for convolution-by-matmul workflows:
        each column is one receptive field, each row is one ``(channel, tap)`` pair.
    """
    return TensorValuedFunction(
            _unfold2d,
            _unfold2d_grad)(
                X, 
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                pad_with=pad_with)

def unfold1d(X: Tensor,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        pad_with: float = 0.0):
    """Extract sliding 1D patches from a tensor.

    For an input ``X`` with shape ``(B, C, L)``, this returns a tensor of
    shape ``(B, C*kL, L_out)``, where:

    - ``L_out = floor((L + 2*pL - dL*(kL - 1) - 1) / sL) + 1``

    Args:
        X: Input tensor of shape ``(B, C, L)``.
        kernel_size: Kernel length ``kL``.
        stride: Sliding stride ``sL``.
        padding: Symmetric padding ``pL`` on the length axis.
        dilation: Dilation ``dL`` for kernel taps.
        pad_with: Constant used to fill padded regions.

    Returns:
        Tensor: Tensor of shape ``(B, C*kL, L_out)`` containing flattened
        receptive fields along the length axis.

    Notes:
        The output layout mirrors ``unfold2d``: rows encode ``(channel, tap)``,
        and columns encode output positions.
    """
    return TensorValuedFunction(
            _unfold1d,
            _unfold1d_grad)(
                X,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                pad_with=pad_with)

# *----------------------------------------------------*
#                   CNN CORE LAYERS
# *----------------------------------------------------*

class Conv2D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        use_bias: bool = True,
        pad_with: float = 0.0,
        *,
        method: Literal["kaiming_normal"] = "kaiming_normal",
        training: bool = True,
        seed: int | None = None):

        """
        `kernel_size`, `stride`, `padding`, `dilation` can either be:
            1) int, in which case the same value is used for the height and width dimension; OR
            2) tuple[int, int], in which case, the first int is used for the height dimension, and the second int for the width dimension.
        """

        super().__init__(training=training)

        self.method = method
        self._rng, self.seed = rng_from_seed(seed)
        self.use_bias = bool(use_bias)

        try:
            init_fn = {"kaiming_normal": kaiming_normal}[method]
        except KeyError as e:
            raise ValueError(f"Unknown init method '{method}'") from e

        self.kernel_size = self._pair(kernel_size)
        self.stride = self._pair(stride)
        self.padding = self._pair(padding)
        self.dilation = self._pair(dilation)
        self.pad_with = pad_with

        # trainable parameters
        self.kernels: list[Tensor] = [
                init_fn(in_channels, out_channels, kernel_size)
                for _ in range(out_channels)
            ]
    
    @staticmethod
    def _pair(v: int | tuple[int, int]):
        if isinstance(v, int):
            return (v, v)
        return v

class Conv1D(Layer):
    pass

class MaxPool2D(Layer):
    pass

class MeanPool2D(Layer):
    pass

class AdaPool2D(Layer):
    pass

class MaxPool1D(Layer):
    pass

class MeanPool1D(Layer):
    pass

class AdaPool1D(Layer):
    pass

# *----------------------------------------------------*
#                    RNN CORE LAYERS
# *----------------------------------------------------*

class RNNCell(Layer):
    # some of my thoughts on the RNNs:
    # since we cannot process intro-sequence samples in parallel
    # -- only sequentially -- I was thinking about the following:
    # what if we use shared memory and load the batches into it
    # and then split the batch into k chunks -- 1 chunk per worker --
    # and then simply use multiprocessing with ProcessPoolExecutor.
    # We might need to use threadpoolctl though to avoid oversubscribing the CPU.

    # On a different point:
    # as for the cell's architecture, I am thinking about letting the user either pass in
    # NN objects for 1) R: x -> g, 2) W: (h, g) -> h, 3) U: h -> y networks or override the corresponding
    # methods of the class if they subclass from the RNNCell.
    # Then, this class will simply be an orhestrator in the sequence processeing task, properly
    # applying the R, W, U networks to sequences.
    # I reckon this is a good approach because then trailing dimensions of data samples can
    # be arbitrary: [B, T, ...] -- while the Batch and Token dims are of course required to be leading.
    pass

class LSTMCell(Layer):
    # Same thoughts on parallelism as for vanilla RNNCell's apply here.
    # NOTE: that's not an easy operation:
    # in order for us to avoid constantly pickling model weights and loading
    # them into processes we must keep them in the shared memory instance as well
    # and then update the buffer with gradient optimization. We will then write from
    # the SHAM back into the model in RAM. So some level of IPC will be required,
    # which means it'll pretty challenging.

    # As for the class organization, again, the same as for the RNNCell, but now
    # also passing in or overriding things like forget gate, input gate, etc.
    pass

# *----------------------------------------------------*
#                     Self-attention
# *----------------------------------------------------*

class Attention(Layer):
    pass

# *----------------------------------------------------*
#                     Miscellaneous
# *----------------------------------------------------*

class Sequential(Layer):
    pass

# *----------------------------------------------------*


__all__ = [
    "xavier_glorot_normal",
    "Layer",
    "Affine",
    "Dropout",
    "BatchNorm1d",
    "Embedding",
    "unfold2d",
    "unfold1d",
]

if __name__ == "__main__":
    pass
