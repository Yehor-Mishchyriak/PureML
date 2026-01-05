"""Autodiff core: `Tensor` wraps NumPy arrays, `TensorValuedFunction` represents graph nodes,
and reverse-mode backprop walks the creator graph with per-node forward context caching.
Backprop seeds ones by default, unbroadcasts gradients to input shapes, frees cached ctx,
and respects `no_grad` via a contextvar. Includes low-level math ops (elemwise, matmul,
reshape/flatten, slicing with scatter-add backward) plus helpers for shape-safe grads."""
from __future__ import annotations

# third party
import numpy as np
# built-in
from functools import wraps
from typing import Callable
import contextvars
import logging
import inspect
from typing import Any

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#               CLASSES & HELPER FUNCTIONS
# *----------------------------------------------------*

# added for clearer tracebacks
class GradientNotDefined(Exception):
    """Raised when a gradient function is not provided for a forward op.

    Args:
        func_name: Name of the forward function lacking a gradient.
    """

    def __init__(self, func_name: str) -> None:
        super().__init__(f"Gradient not defined for function '{func_name}'")


class Tensor:
    """A tiny autodiff-aware tensor wrapper around a NumPy array.

    Tracks data, gradient, and the function node that created it.

    Attributes:
        data: Backing NumPy array.
        requires_grad: Whether to track gradients for this tensor.
        grad: Accumulated gradient (same shape as `data`) or `None`.
        _creator: The `TensorValuedFunction` node that produced this tensor (or `None`).

    Notes:
        Grad recording is controlled by a contextvar; use `no_grad()` to
        temporarily disable graph building. During backprop, each creator
        node may provide a per-node forward context dict (`fn.fwd_ctx`)
        that can cache intermediates; if a node's grad function opts in
        (accepts a keyword-only parameter named `context`), that dict is
        passed into the grad function. The context is freed after the
        node's gradients are computed.
    """

    __hash__ = object.__hash__
    __eq__   = object.__eq__ 

    _grad_enabled_var = contextvars.ContextVar("grad_enabled", default=True)

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        *,
        dtype=None,
        copy: bool = False,
        ensure_writable: bool = True,
        coerce_float_if_grad: bool = True,
    ):
        """
        data: array-like | Tensor | np.ndarray
        requires_grad: track grads
        dtype: optional target dtype
        copy: force copy even if not needed
        ensure_writable: make a copy if input is read-only
        coerce_float_if_grad: ints/bools -> float if requires_grad=True
        """
        if isinstance(data, Tensor):
            arr = np.asarray(data.data, dtype=dtype)
        else:
            arr = np.asarray(data, dtype=dtype)

        if coerce_float_if_grad and requires_grad and arr.dtype.kind in "biu": # bool, sint, or usint
            arr = arr.astype(np.float64 if dtype is None else dtype, copy=False)

        if ensure_writable and not arr.flags.writeable:
            arr = arr.copy()

        if copy:
            arr = arr.copy()

        # no `obj` dtype
        if arr.dtype == np.dtype("O") and requires_grad:
            raise TypeError("Tensor with requires_grad=True does not support object dtype")

        self.data: np.ndarray = arr
        self.requires_grad: bool = bool(requires_grad)
        self.grad: np.ndarray | None = None
        self._creator: TensorValuedFunction | None = None

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    def __getitem__(self, index) -> Tensor:
        """Autodiff-aware slicing (NumPy semantics).

        Supports all standard NumPy indexing (ints, slices, ellipsis, None,
        boolean masks, and advanced integer arrays). Backward pass scatters
        upstream gradients into a zeros-like array of the input's shape.

        Example:
            x = Tensor(np.arange(6).reshape(2,3), requires_grad=True)
            y = x[:, 1:]          # view-like slice
            y.sum().backward()    # x.grad has 1s in the sliced region, 0 elsewhere
        """
        return TensorValuedFunction(_slice, _slice_grad)(self, index=index)

    @staticmethod
    def _build_topo(tensor: Tensor, *, topo: list, visited: set) -> None:
        """Depth-first build of a topological list of creator nodes.

        Args:
            tensor: Start tensor.
            topo: Output list to append creator nodes in topo order.
            visited: Set used to avoid revisiting nodes.
        """
        if tensor not in visited:
            visited.add(tensor)
            if tensor._creator:
                for inp_tensor in tensor._creator.inputs:
                    Tensor._build_topo(inp_tensor, topo=topo, visited=visited)
                topo.append(tensor._creator)

    def _creators_reverse_topo(self) -> list[TensorValuedFunction]:
        """List[TensorValuedFunction]: Creator nodes in reverse topological order."""
        topo = list()
        Tensor._build_topo(self, topo=topo, visited=set())
        topo.reverse()
        return topo

    def backward(self, grad: np.ndarray = None) -> None:
        """Run reverse-mode autodiff from this tensor.

        Performs a reverse topological traversal over creator nodes. If
        `grad` is None, seeds with a tensor of ones like `self.data`.

        For each node:
        1) Computes per-input gradients by calling the node's grad function
            with `(upstream_grad, *inputs)` (raw NumPy arrays). If the grad
            function's signature contains a `context` parameter, the node's
            forward context dict (`fn.fwd_ctx`) is also passed as `context=...`.
        2) Accumulates gradients into each input tensor's `.grad`.
        3) Frees the node's forward context via `fn._free_fwd_ctx()`.

        Args:
            grad: Upstream gradient to seed backprop (`dL/dself`). If omitted,
                uses `np.ones_like(self.data)`.
        """
        if not self.requires_grad:
            _logger.debug("backward() called on tensor with requires_grad=False; no-op.")
            return
        
        if grad is None:
            upstream_grad = np.ones_like(self.data) # dL/dL
            _logger.debug("backward(): no grad provided; using ones_like with shape=%s", upstream_grad.shape)
        else:
            upstream_grad = grad # dL/dself
            _logger.debug("backward(): using provided grad with shape=%s", upstream_grad.shape)

        self.grad = upstream_grad

        creators = self._creators_reverse_topo()
        _logger.debug("backward(): %d creator nodes to visit.", len(creators))

        for fn in creators: # fn: TensorValuedFunction
            # upstream * LG --> passing in the upstream and the inputs needed to compute LG and then perform (up * lg) inside .grad_fn
            fn_grads_wrt_inputs = _call_with_optional_context(fn.grad_fn, [fn.output.grad, *[inp.data for inp in fn.inputs]], fn.fwd_ctx)

            # ---- Error checks ----
            # 1) Must return a tuple
            if not isinstance(fn_grads_wrt_inputs, tuple):
                raise RuntimeError(
                    f"Grad function '{fn.forward_fn.__name__}' should return a tuple, "
                    f"but returned {type(fn_grads_wrt_inputs).__name__}"
                )

            # 2) Length must match number of inputs
            if len(fn_grads_wrt_inputs) != len(fn.inputs):
                raise RuntimeError(
                    f"Grad function '{fn.forward_fn.__name__}' returned {len(fn_grads_wrt_inputs)} gradients, "
                    f"but expected {len(fn.inputs)} (one per input)"
                )
            # ----------------------

            for fn_grad, inp in zip(fn_grads_wrt_inputs, fn.inputs):
                if inp.requires_grad:
                    before = None if inp.grad is None else "set"
                    inp.grad = fn_grad if inp.grad is None else inp.grad + fn_grad # accumulate the gradient
                    after = "set"
                    _logger.debug(
                        "Accumulated grad for input (requires_grad=True): before=%s, now=%s, shape=%s",
                        before, after, None if fn_grad is None else getattr(fn_grad, "shape", None)
                    )
                    # ^ a single tensor can feed into multiple operations, so we aggregate all of those contributions:
                    # ------------------------------------------------------------------------------------------------
                    #  f    |                           | This addition is what is coded above it
                    # / \   |                           V
                    # x  y  | So, df/dt = df/dx * dx/dt + df/dy * dy/dt;
                    # \ /   | Note, if inp.grad is None, we are computing df/dt without having computed df/dy,
                    #  t    | and then once we computed one branch (df/dt = df/dx * dx/dt -- fn_grad from above),
                    #       | we store it inside inp.grad, and then next time we are computing the other branch inp.grad is not None,
                    #       | so we add (df/dy * dy/dt) to it.
            fn._free_fwd_ctx()

    def zero_grad(self) -> None:
        """Set this tensor's `.grad` to `None`."""
        _logger.debug("zero_grad() on tensor with shape=%s", self.data.shape)
        self.grad = None

    def numpy(self, *, copy: bool = True, readonly: bool = False) -> np.ndarray:
        """
        Get the underlying array for external use.

        Args:
            copy: If True (default), return a defensive copy.
            readonly: If True, mark the returned array as non-writable
                      (only applies when copy=False or after the copy).

        Returns:
            np.ndarray
        """
        arr = self.data.copy() if copy else self.data.view()
        if readonly:
            # ensure the result cannot be mutated accidentally
            try:
                arr.setflags(write=False)
            except ValueError:
                pass
        return arr

    def detach(self) -> Tensor:
        """Return a new Tensor that shares storage but is not tracked."""
        # share storage (no copy), don't force writability changes here
        t = Tensor(self.data, requires_grad=False, copy=False, ensure_writable=False)
        # make sure it is a leaf with no history and no grad
        t._creator = None
        t.grad = None
        return t

    def detach_(self) -> Tensor:
        """In-place: stop autograd tracking for this Tensor."""
        self._creator = None
        self.requires_grad = False
        self.grad = None
        return self

    def requires_grad_(self, mode: bool) -> Tensor:
        """In-place toggle for requires_grad"""
        self.requires_grad = bool(mode)
        return self

    def _collect_graph(self) -> set[Tensor]:
        """Collect this tensor and all upstream tensors reachable via creators (ancestors)."""
        visited: set[Tensor] = set()
        stack = [self]
        while stack:
            t = stack.pop()
            if t in visited:
                continue
            visited.add(t)
            fn = t._creator
            if fn is not None:
                # walk to inputs (parents)
                stack.extend(fn.inputs)
        return visited

    def zero_grad_graph(self) -> None:
        """Set `.grad=None` for this tensor and all upstream tensors."""
        nodes = self._collect_graph()
        _logger.debug("zero_grad_graph() over %d tensors", len(nodes))
        for t in nodes:
            t.grad = None

    def detach_graph(self) -> None:
        """Detach this tensor and all upstream tensors: clear creators, grads, and stop tracking."""
        nodes = self._collect_graph()
        _logger.debug("detach_graph() over %d tensors", len(nodes))
        for t in nodes:
            fn = t._creator
            if fn is not None:
                # free any cached forward intermediates before breaking the link
                try:
                    fn._free_fwd_ctx()
                except Exception:
                    pass
            t._creator = None
            t.grad = None
            t.requires_grad = False  # make each a leaf that won't build history going forward

    def __repr__(self) -> str:
        """Return a concise representation including grad presence."""
        return (
            f"Tensor(data={self.data}, requires_grad={self.requires_grad}, "
            f"grad={'set' if self.grad is not None else 'None'})"
        )

    # COMPARISONS
    def eq(self, other: Tensor) -> Tensor:
        return Tensor(self.data == other.data, requires_grad=False)

    def ne(self, other: Tensor) -> Tensor:
        return Tensor(self.data != other.data, requires_grad=False)

    def lt(self, other: Tensor) -> Tensor:
        return Tensor(self.data < other.data, requires_grad=False)

    def le(self, other: Tensor) -> Tensor:
        return Tensor(self.data <= other.data, requires_grad=False)

    def gt(self, other: Tensor) -> Tensor:
        return Tensor(self.data > other.data, requires_grad=False)

    def ge(self, other: Tensor) -> Tensor:
        return Tensor(self.data >= other.data, requires_grad=False)

    # reductions
    def all(self, axis=None, keepdims=False) -> Tensor:
        return Tensor(np.all(self.data, axis=axis, keepdims=keepdims), requires_grad=False)

    def any(self, axis=None, keepdims=False) -> Tensor:
        return Tensor(np.any(self.data, axis=axis, keepdims=keepdims), requires_grad=False)

    # ARITHMETIC OPERATIONS
    def __add__(self, other: Tensor)      -> Tensor:
        """Elementwise addition."""
        return TensorValuedFunction(_add, _add_grad)(self, other)
    
    def __sub__(self, other: Tensor)      -> Tensor:
        """Elementwise subtraction."""
        return TensorValuedFunction(_sub, _sub_grad)(self, other)
    
    def __mul__(self, other: Tensor)      -> Tensor:
        """Elementwise multiplication."""
        return TensorValuedFunction(_mul, _mul_grad)(self, other)
    
    def __truediv__(self, other: Tensor)  -> Tensor:
        """Elementwise true division."""
        return TensorValuedFunction(_truediv, _truediv_grad)(self, other)
    
    def __pow__(self, other: Tensor)      -> Tensor:
        """Elementwise power."""
        return TensorValuedFunction(_pow, _pow_grad)(self, other)
    
    def __neg__(self)                     -> Tensor:
        """Unary negation."""
        return TensorValuedFunction(_neg, _neg_grad)(self)

    # MATRIX OPERATIONS
    @property
    def T(self)                           -> Tensor:
        """Matrix transpose (swap last two axes)."""
        return TensorValuedFunction(_transpose, _transpose_grad)(self)
    
    def __matmul__(self, other: Tensor)   -> Tensor:
        return TensorValuedFunction(_matmul, _matmul_grad)(self, other)
    
    # RESHAPING
    def reshape(self, *shape: int) -> Tensor:
        return reshape(self, *shape)

    def flatten(self, *, keep_batch: bool = True, sample_ndim: int | None = None) -> Tensor:
        return flatten(self, keep_batch=keep_batch, sample_ndim=sample_ndim)
    
    # AUXILIARY OPS
    def argmax(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Indices of max values along `axis` (non-differentiable)."""
        idx = np.argmax(self.data, axis=axis, keepdims=keepdims)
        return Tensor(np.asarray(idx), requires_grad=False)

class no_grad:
    """Context manager that disables gradient recording within its scope."""

    def __enter__(self):
        """Enter the no-grad context (disable grads)."""
        self.token = Tensor._grad_enabled_var.set(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the no-grad context (restore previous grad state)."""
        Tensor._grad_enabled_var.reset(self.token)

class TensorValuedFunction:
    """Computational graph node that wraps a forward function and its gradient.

    The forward function consumes raw NumPy arrays and returns a single
    NumPy array. The gradient function consumes `(upstream_grad, *inputs)`
    and returns a tuple of per-input gradients (NumPy arrays or `None`).

    Optional forward/grad context:
        Both `forward_fn` and `grad_fn` may *optionally* accept a keyword-only
        argument named `context` (a `dict`). When present in the function's
        signature, a per-node context dict (`fwd_ctx`) is passed as `context=...`.
        The forward call also stashes its output under `context["out"]` by default
        so the grad function can reuse it without recomputation. After the node's
        grad is computed, the context is cleared.

    Attributes:
        forward_fn: Callable that performs the forward computation.
        _grad_fn: Callable that computes local gradients.
        inputs: Tensors that were inputs when this node was applied.
        output: The output tensor produced by the last call, or `None`.
        _fwd_cache: Lazily-created dict used as the forward/grad context.
    """
    
    # forward_fn takes in any number of anything and outputs a single np.ndarray
    # grad_fn takes in any number of anything and outputs a variable number of np.ndarray's and potentially None's, so we add '| None'
    def __init__(self, forward_fn: Callable[..., np.ndarray], grad_fn: Callable[..., tuple[np.ndarray | None, ...]] = None) -> None:
        """Initialize a function node.

        Args:
            forward_fn: Forward computation function (arrays → array).
            grad_fn: Gradient function `(upstream, *inputs[, context=...]) -> tuple[grads]`.
                    If `grad_fn` is `None`, calling `.grad_fn` raises `GradientNotDefined`.
        """

        self.forward_fn = forward_fn
        self._grad_fn = grad_fn

        self.inputs: list[Tensor] = []
        self.output: Tensor | None = None # <-- None if wasn't called yet

        self.name = self.forward_fn.__name__
        self._fwd_cache = None

    @property
    def grad_fn(self) -> Callable[..., tuple[np.ndarray, ...]]:
        """Return the gradient function or raise if missing.

        Raises:
            GradientNotDefined: If no gradient function is provided.

        Returns:
            Callable: The gradient function for this node.
        """
        if self._grad_fn is None:
            raise GradientNotDefined(self.name)
        return self._grad_fn

    # note, __call__ takes in Tensors, but acts on t.data for each t from the 'inputs' Tensor tuple
    def __call__(self, *inputs: Tensor, **kwargs: Any) -> Tensor:
        """Apply the forward function to `inputs`, producing a `Tensor`.

        Keyword arguments:
            Any extra keyword args are *optionally* forwarded to the forward
            function (and later to the grad function via the saved `context`)
            but only if the target function's signature declares them. Unknown
            kwargs are silently ignored (not passed), avoiding TypeError.

        Behavior:
            • If any input requires gradients (and grad recording is enabled),
            build a new node:
                - Convert input tensors to raw arrays.
                - Call the forward function. If it has a `context` parameter,
                pass the node's `fwd_ctx` dict as `context=...`.
                - Forward only those kwargs accepted by the forward function.
                - Cache the raw forward result under `node.fwd_ctx["out"]`
                for use in backprop and store the produced `Tensor` as
                `node.output`.
            • If gradients are not required, just run the forward function
            (again forwarding only accepted kwargs) and return a
            non-differentiable `Tensor`.

        Returns:
            Tensor: The forward result (requires_grad mirrors the inputs).
        """
        requires_grad = is_grad_enabled() and any(inp.requires_grad for inp in inputs)

        # raw data to ndarray (to avoid memoryview issues)
        arrays = [np.asarray(tensor.data) for tensor in inputs]

        _logger.debug(
            "TensorValuedFunction.__call__: forward=%s, requires_grad=%s, n_inputs=%d",
            getattr(self.forward_fn, "__name__", str(self.forward_fn)), requires_grad, len(arrays)
        )

        if requires_grad:
            node = TensorValuedFunction(self.forward_fn, self._grad_fn)
            node.inputs = list(inputs)
            output_data = _call_with_optional_context(node.forward_fn, arrays, node.fwd_ctx, kwargs)
            node.fwd_ctx.setdefault("out", output_data)

            output_tensor = Tensor(
                data=output_data,
                requires_grad=True
            )

            output_tensor._creator = node
            node.output = output_tensor

            _logger.debug("Forward produced output with shape=%s (requires_grad=True)", output_data.shape)

        else:
            output_data = _call_with_optional_context(self.forward_fn, arrays, None, kwargs)
            output_tensor = Tensor(
                data=output_data,
                requires_grad=False
            )
            _logger.debug("Forward produced output with shape=%s (requires_grad=False)", output_data.shape)

        return output_tensor

    @property
    def fwd_ctx(self):
        """Return this node's forward/grad context dict.

        Created on first access; intended for caching intermediates (e.g.,
        softmax probabilities, masks, or the forward output under key "out").
        The engine calls `_free_fwd_ctx()` after the node's gradients are used.
        """
        if self._fwd_cache is None:
            self._fwd_cache = {}
        return self._fwd_cache

    def _free_fwd_ctx(self):
        """Clear this node's forward/grad context.

        Called by the engine after applying the node's gradient function to
        release cached intermediates and reduce peak memory.
        """
        self._fwd_cache = None

    def __repr__(self) -> str:
        """Return a readable representation of the function node."""
        gname = self._grad_fn.__name__ if self._grad_fn is not None else None
        return f"Function(forward_fn={self.forward_fn.__name__}, grad_fn={gname})"

def _update_ctx(
    ctx: dict | None = None,
    *,
    overwrite: bool = False,
    **variables: Any,
) -> None:
    """Update a node context dict with intermediates.

    If ctx is None, do nothing. If overwrite=False, existing keys are kept.

    Supports lazy values via callables:
        _update_ctx(ctx, foo=lambda: expensive_compute())

    Args:
        ctx: The context dict (or None).
        overwrite: Whether to replace existing keys.
        **variables: Key/value pairs to cache; values may be callables.
    """
    if ctx is None:
        return
    for k, v in variables.items():
        if not overwrite and k in ctx:
            continue
        ctx[k] = v() if callable(v) else v

def _call_with_optional_context(fn, arrays, ctx, extra_kwargs=None):
    """Invoke `fn(*arrays)`, optionally passing `context=ctx` and accepted kwargs."""
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

        kw = {}

        # If caller supplied a 'context' dict, merge it into the node ctx
        supplied_ctx = None
        if extra_kwargs and "context" in extra_kwargs and isinstance(extra_kwargs["context"], dict):
            supplied_ctx = extra_kwargs["context"]
            if ctx is None:
                ctx = {}
            # merge (ctx was supplied by the engine, so it's empty at first)
            for k, v in supplied_ctx.items():
                if k not in ctx:
                    ctx[k] = v

        # Always pass the node context if the fn accepts it (or **kwargs)
        if "context" in params or accepts_var_kw:
            kw["context"] = ctx

        # Forward any other accepted kwargs, but NEVER override 'context'
        if extra_kwargs:
            if accepts_var_kw:
                # copy but drop 'context' to avoid clobbering cuz we already merged the node ctx and the supplied one
                for k, v in extra_kwargs.items():
                    if k != "context":
                        kw[k] = v
            else:
                for k, v in extra_kwargs.items():
                    if k == "context":
                        continue
                    if k in params:
                        kw[k] = v

        return fn(*arrays, **kw) if kw else fn(*arrays)
    except (ValueError, TypeError):
        return fn(*arrays)

def is_grad_enabled() -> bool:
    """Return whether gradient recording is currently enabled.

    Returns:
        bool: True if gradients are enabled in the current context.
    """
    return Tensor._grad_enabled_var.get()

# *----------------------------------------------------*
#                       CONVENIENCE
# *----------------------------------------------------*

# NumPy broadcasting happens during the forward pass through the NN;
# therefore, all the gradients have their dimensions corresponding to the outputs -- not the input dimensions we need;
# they are basically of stretched input shapes, so we need to collapse them back
def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Reduce a broadcast-shaped gradient back to the original `target_shape`.

    Drops inserted broadcast-only dims (but never the leading batch axis when it already matches)
    and sums over axes that were broadcast (size-1 in `target_shape` along aligned axes).

    Args:
        grad: Gradient aligned with the *output* (broadcasted) shape.
        target_shape: Original input shape to reduce to.

    Returns:
        np.ndarray: Gradient with shape exactly `target_shape`.
    """
    _logger.debug("unbroadcast: in.shape=%s target=%s", getattr(grad, "shape", None), target_shape)

    g = np.asarray(grad)

    # Fast paths
    if g.shape == tuple(target_shape):
        return g
    if len(target_shape) == 0:
        # input was scalar → sum everything
        return np.asarray(g.sum(), dtype=g.dtype)
    if g.ndim == 0 and len(target_shape) > 0:
        # upstream is scalar (e.g., reduction) → broadcast it to the input
        return np.ones(target_shape, dtype=g.dtype) * g

    # IMPORTANT: the leading dimension is the batch dimension.
    # If it matches, we must **not** reduce it. We'll only reduce the "tail".
    protect_batch = (g.ndim >= 1 and len(target_shape) >= 1 and g.shape[0] == target_shape[0])

    # Step 1) Normalize rank: drop *inserted* broadcast-only axes by summing,
    # but never the protected batch axis.
    if protect_batch:
        # number of tail dims on each
        g_tail = g.ndim - 1
        t_tail = len(target_shape) - 1
        # If grad has extra tail dims, reduce them at axis=1 (right after batch)
        extra_tail = g_tail - t_tail
        for _ in range(max(extra_tail, 0)):
            g = g.sum(axis=1, keepdims=True)
    else:
        # No protected batch: reduce extra leading axes at axis=0
        extra_lead = g.ndim - len(target_shape)
        for _ in range(max(extra_lead, 0)):
            g = g.sum(axis=0, keepdims=True)

    # Step 2) Right-align the *tails* and sum where the target had size 1 but grad had >1.
    # We compute over the tail (skip batch if protected) to preserve axis 0.
    g_start = 1 if protect_batch else 0
    t_start = 1 if protect_batch else 0

    tail_len = min(g.ndim - g_start, len(target_shape) - t_start)
    for i in range(tail_len):
        gi = g.shape[g_start + i]
        ti = target_shape[t_start + i]
        if ti == 1 and gi != 1:
            g = g.sum(axis=g_start + i, keepdims=True)

    # Step 3) Final reshape to drop size-1 axes that don't exist in target.
    # This is safe because we've summed all broadcasted positions.
    out = g.reshape(target_shape)

    _logger.debug("unbroadcast: out.shape=%s", out.shape)
    return out

def _shape_safe_grad(grad_fn):
    """Decorator ensuring per-input grads match original input shapes.

    Wraps a raw grad function, forwards any keyword arguments (including
    `context=...`), converts the return to a tuple, and then “unbroadcasts”
    each gradient back to the corresponding input's original shape.

    Contract:
        The wrapped grad function must return one gradient per input
        (or `None` for inputs that do not receive a gradient).

    Args:
        grad_fn: Raw gradient function `(upstream, *inputs[, context=...]) -> grads`.

    Returns:
        Callable: Wrapped grad function with shape safety.
    """
    @wraps(grad_fn)
    def wrapper(upstream, *inputs, **kwargs):
        # call the “raw” grad function
        grads = grad_fn(upstream, *inputs, **kwargs)
        # ensure it's a tuple
        if not isinstance(grads, tuple):
            grads = (grads,)
        # Unbroadcast each gradient to match its input’s shape
        safe = []
        for g, inp in zip(grads, inputs):
            if g is None:
                safe.append(None)
            else:
                # allow Python scalars and 0-D arrays
                g_arr = np.asarray(g)
                safe.append(_unbroadcast(g_arr, getattr(inp, "shape", ())))
        _logger.debug("_shape_safe_grad: processed %d grads", len(safe))
        return tuple(safe)
    return wrapper

# *----------------------------------------------------*
#                       TENSOR MATH
# *----------------------------------------------------*

def _add(x: np.ndarray, y: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    """Elementwise addition."""
    return x + y

@_shape_safe_grad
def _add_grad(upstream_grad: np.ndarray, x: np.ndarray, y: np.ndarray, *, context: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of addition: upstream flows to both inputs."""
    return upstream_grad, upstream_grad

def _mul(x: np.ndarray, y: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    """Elementwise multiplication."""
    return x * y

@_shape_safe_grad
def _mul_grad(upstream_grad: np.ndarray, x: np.ndarray, y: np.ndarray, *, context: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of multiply: (up*y, up*x)."""
    return upstream_grad * y, upstream_grad * x

def _sub(x: np.ndarray, y: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    """Elementwise subtraction."""
    return x - y

@_shape_safe_grad
def _sub_grad(upstream_grad: np.ndarray, x: np.ndarray, y: np.ndarray, *, context: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of subtraction: (up, -up)."""
    return upstream_grad, -upstream_grad

def _truediv(x: np.ndarray, y: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    """Elementwise true division."""
    return x / y

@_shape_safe_grad
def _truediv_grad(upstream_grad: np.ndarray, x: np.ndarray, y: np.ndarray, *, context: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of division: (up*(1/y), up*(-x/y^2))."""
    return upstream_grad * (1.0 / y), upstream_grad * (-x / (y * y))

def _pow(x: np.ndarray, y: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    """Elementwise power."""
    return x ** y

@_shape_safe_grad
def _pow_grad(upstream_grad: np.ndarray, x: np.ndarray, y: np.ndarray, *, context: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of power: (up*y*x^(y-1), up*x^y*log(x))."""
    return upstream_grad * (y * x ** (y - 1)), upstream_grad * (x ** y * np.log(x))

def _log2(x: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    """Base-2 logarithm."""
    return np.log2(x)

@_shape_safe_grad
def _log2_grad(upstream_grad: np.ndarray, x: np.ndarray, *, context: dict | None = None) -> tuple[np.ndarray]:
    """Gradient of log2: up / (x * ln(2))."""
    return (upstream_grad / (x * np.log(2)),)

def log2(x: Tensor, *, context: dict | None = None) -> Tensor:
    """Apply base-2 logarithm elementwise to a `Tensor`."""
    return TensorValuedFunction(_log2, _log2_grad)(x)

def _ln(x: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    """Natural logarithm."""
    return np.log(x)

@_shape_safe_grad
def _ln_grad(upstream_grad: np.ndarray, x: np.ndarray, *, context: dict | None = None) -> tuple[np.ndarray]:
    """Gradient of ln: up / x."""
    return (upstream_grad / x,)

def ln(x: Tensor) -> Tensor:
    """Apply natural logarithm elementwise to a `Tensor`."""
    return TensorValuedFunction(_ln, _ln_grad)(x)

def _sqrt(x: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    out = np.sqrt(x)
    _update_ctx(context, out=out)
    return out

@_shape_safe_grad
def _sqrt_grad(upstream_grad: np.ndarray, x: np.ndarray, *, context: dict | None = None):
    s = (context or {}).get("out")
    if s is None:
        s = np.sqrt(x)
    return (0.5 * upstream_grad / (s + 1e-12),)

def sqrt(x: Tensor) -> Tensor:
    return TensorValuedFunction(_sqrt, _sqrt_grad)(x)

def _neg(x: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    """Unary negation."""
    return -x

@_shape_safe_grad
def _neg_grad(upstream_grad: np.ndarray, x: np.ndarray, *, context: dict | None = None) -> tuple[np.ndarray]:
    """Gradient of negation: -up."""
    return (-upstream_grad,)

def _transpose(x: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    """Swap the last two axes (matrix transpose). No-op for 0D/1D.

    Args:
        x: Input array.

    Returns:
        np.ndarray: Transposed array (last two axes swapped).
    """
    if x.ndim < 2:
        return x
    return np.swapaxes(x, -1, -2)

@_shape_safe_grad
def _transpose_grad(upstream_grad: np.ndarray, x: np.ndarray, *, context: dict | None = None) -> tuple[np.ndarray]:
    """Gradient of transpose is transpose (swap the same two axes).

    Args:
        upstream_grad: Upstream gradient `dL/dY`.
        x: Original input array to the transpose op.

    Returns:
        Tuple[np.ndarray]: A one-element tuple with `dL/dX`.
    """
    if x.ndim < 2:
        return (upstream_grad,)
    return (np.swapaxes(upstream_grad, -1, -2),)

def _matmul(A: np.ndarray, B: np.ndarray, *, context: dict | None = None) -> np.ndarray:
    # Supports A:(..., i, k), B:(..., k, j) -> (..., i, j) with broadcasting over leading dims
    return A @ B

@_shape_safe_grad
def _matmul_grad(up: np.ndarray, A: np.ndarray, B: np.ndarray, *, context: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    # up:(..., i, j) ; A:(..., i, k) ; B:(..., k, j)
    dA = up @ np.swapaxes(B, -1, -2)            # (..., i, k)
    dB = np.swapaxes(A, -1, -2) @ up            # (..., k, j)
    return dA, dB

def _reshape_fwd(new_shape: tuple[int, ...]):
    def _reshape(x: np.ndarray, *, context: dict | None = None) -> np.ndarray:
        _update_ctx(context, in_shape=x.shape)
        return np.reshape(x, new_shape)
    return _reshape

def _reshape_bwd():
    @_shape_safe_grad
    def _reshape_grad(up: np.ndarray, x: np.ndarray, *, context: dict | None = None):
        in_shape = (context or {}).get("in_shape", x.shape)
        return (up.reshape(in_shape),)
    return _reshape_grad

def reshape(x: Tensor, *shape: int) -> Tensor:
    """Return a view-like reshape with autograd support."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return TensorValuedFunction(_reshape_fwd(tuple(shape)), _reshape_bwd())(x)

def flatten(x: Tensor, *, keep_batch: bool = True, sample_ndim: int | None = None) -> Tensor:
    """
    Flatten `x` while optionally preserving leading dims.

    Modes:
      • sample_ndim is not None:
          Keep all leading dims, collapse the last `sample_ndim` dims.
          Example: (B, T, C, H, W) with sample_ndim=3 -> (B, T, C*H*W)
      • else, keep_batch:
          If x.ndim >= 2, keep first dim (batch) and collapse the rest.
          Example: (B, ...) -> (B, prod(...))
      • else:
          Collapse everything to 1D.
    """
    shp = x.data.shape

    if sample_ndim is not None:
        if not (0 <= sample_ndim <= x.data.ndim):
            raise ValueError(f"sample_ndim must be in [0, {x.data.ndim}], got {sample_ndim}")
        if sample_ndim == 0:
            return x  # nothing to flatten
        lead = shp[:-sample_ndim]
        tail = shp[-sample_ndim:]
        new_shape = lead + (int(np.prod(tail)),)
        return reshape(x, new_shape)

    if keep_batch and x.data.ndim >= 2:
        new_shape = (shp[0], int(np.prod(shp[1:])))
        return reshape(x, new_shape)

    # collapse everything
    new_shape = (int(np.prod(shp)),)
    return reshape(x, new_shape)

def _slice(x: np.ndarray, *, index, context: dict | None = None) -> np.ndarray:
    """Forward: x[index]."""
    _update_ctx(context, in_shape=x.shape, index=index)
    return x[index]

@_shape_safe_grad
def _slice_grad(upstream_grad: np.ndarray, x: np.ndarray, *, context: dict | None = None):
    """Backward: scatter-add upstream_grad into a zeros_like(x) at `index`."""
    if context is None or "index" not in context:
        raise RuntimeError("Embedding/slice grad: missing `index` in context")
    index = context["index"]
    g = np.zeros_like(x)
    np.add.at(g, index, upstream_grad)
    return (g,)

__all__ = [
    "GradientNotDefined",
    "Tensor",
    "TensorValuedFunction",
    "_update_ctx",
    "_shape_safe_grad",
    "is_grad_enabled",
    "no_grad",
    "ln",
    "log2",
    "sqrt"
]

if __name__ == "__main__":
    pass
