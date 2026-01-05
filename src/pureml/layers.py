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
- Dropout (inverted, cached mask/scale, seedable, mode-aware)
- BatchNorm1d with running stats buffers and EMA momentum
- Embedding with optional pad freezing and seedable init
All layers use `TensorValuedFunction` ops from `machinery` and RNG helpers in `util`."""
from __future__ import annotations

# third party
import numpy as np
# built-in
from abc import ABC, abstractmethod
import logging
# local
from .machinery import (
    Tensor, TensorValuedFunction, _shape_safe_grad, _update_ctx, sqrt
)
from . import general_math
from .util import rng_from_seed

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#               CLASSES & HELPER FUNCTIONS
# *----------------------------------------------------*

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- WEIGHT INITIALIZATION STRATEGIES -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

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

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

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

    def apply_state(
        self,
        *,
        tunable: tuple[np.ndarray, ...] | list[np.ndarray] = (),
        buffers: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Default in-place state load: writes arrays into `parameters` and `named_buffers()` Tensors."""
        # write trainables in-order
        if tunable:
            for t, arr in zip(self.parameters, tunable):
                t.data = np.asarray(arr, dtype=t.data.dtype)

        # write buffers by name (only if buffer is a Tensor)
        if buffers:
            for name, v in self.named_buffers().items():
                if name in buffers and isinstance(v, Tensor):
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
                 method="xavier-glorot-normal", 
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

        ctx = context or {}
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
        # mask/scale are not trainable; return zeros of matching shapes
        return grad_X, np.zeros_like(mask), np.zeros_like(scale)

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

    def named_buffers(self) -> dict[str, Tensor]:
        """Return running statistics buffers.

        Returns:
            dict[str, Tensor]: A mapping with:
                - ``"running_mean"``: EMA of per-feature means, shape ``(F,)``
                - ``"running_variance"``: EMA of per-feature variances, shape ``(F,)``."""
        return {
            "running_mean": self.running_mean,
            "running_variance": self.running_variance,
        }

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
        method="xavier-glorot-normal",
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

        ctx = context or {}
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


__all__ = [
    "xavier_glorot_normal",
    "Layer",
    "Affine",
    "Dropout",
    "BatchNorm1d",
    "Embedding"
]

if __name__ == "__main__":
    pass
