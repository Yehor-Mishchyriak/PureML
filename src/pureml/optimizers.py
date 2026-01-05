"""Optimizers and schedulers for PureML tensors with checkpointable state.

Implements SGD, AdaGrad, RMSProp, and Adam plus LR schedulers (StepLR, ExponentialLR,
CosineAnnealingLR). Optim states (hypers and per-param slots) persist via ArrayStorage
alongside current parameters to guarantee reproducible restarts."""
from __future__ import annotations

# third party
import numpy as np
# built-in
import logging
from typing import Any
from pathlib import Path
# local
from .machinery import Tensor
from .general_math import ewma
from . import util

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                      Optimizers
# *----------------------------------------------------*

class Optim:
    """Base optimizer.

    Checkpointing:
        Subclasses declare:
          - HYPERS: tuple of JSON-safe scalar field names on `self` to save (e.g., "lr", "beta1").
          - SLOTS:  tuple of per-parameter slot list names on `self`, each aligned with `model_params`
                    (e.g., "_v", "_r"). Each list entry is either an ndarray state tensor or None.

        Methods:
          - `.state` returns (hypers, slots) in normalized form.
          - `.save_state(path)` writes a single archive with:
                attrs["optim.meta"] = {"class", "n_params", "hypers"}
                blocks "optim.<slot>.<i>" for slot arrays
          - `.load_state(path, strict=True)` restores hypers and slots (shape-checked per parameter).
    """

    # what to checkpoint (subclasses override these)
    HYPERS: tuple[str, ...] = ("lr",)
    SLOTS:  tuple[str, ...] = ()

    def __init__(self, model_params: list[Tensor], lr: float) -> None:
        self.model_params = model_params
        self.lr = float(lr)

    def zero_grad(self) -> None:
        """Set all parameter gradients to zero in-place."""
        _logger.debug("Optim.zero_grad: params=%d", len(self.model_params))
        for param in self.model_params:
            param.zero_grad()

    def step(self) -> None:
        """Apply a single optimization step (in-place on model parameters)."""
        raise NotImplementedError
    
    @property
    def state(self) -> tuple[dict[str, Any], dict[str, list[np.ndarray | None]]]:
        hypers: dict[str, Any] = {}
        for name in self.HYPERS:
            if not hasattr(self, name):
                continue
            v = getattr(self, name)
            if isinstance(v, (bool, int, float, str, type(None))):
                hypers[name] = v
            else:
                try:
                    hypers[name] = v.item()
                except Exception:
                    _logger.debug("Skipping non-scalar hyper '%s' (type=%s)", name, type(v).__name__)

        n = len(self.model_params)
        slots: dict[str, list[np.ndarray | None]] = {}
        for slot_name in self.SLOTS:
            lst = getattr(self, slot_name, None)
            if not isinstance(lst, list):
                _logger.debug("Slot '%s' is not a list; initializing empty list", slot_name)
                lst = [None] * n
            # normalize length to n
            if len(lst) != n:
                _logger.debug("Slot '%s' len=%d != n_params=%d; resizing", slot_name, len(lst), n)
                norm = [None] * n
                for i in range(min(len(lst), n)):
                    norm[i] = lst[i]
                lst = norm
            slots[slot_name] = lst

        return hypers, slots

    def save_state(self, pth: Path | str, *, compression_level: int = 3) -> None:
        pth = Path(pth).with_suffix(".pureml.zip")
        hypers, slots = self.state
        _logger.info("Saving optimizer state (%s) to %s", type(self).__name__, pth)

        with util.ArrayStorage.compress_and_cleanup(pth, compression_level) as storage:
            # write all hypers as a single attr under 'optim.meta'
            meta = {
                "class": type(self).__name__,
                "n_params": len(self.model_params),
                "hypers": hypers,
            }
            storage.add_attr("optim.meta", meta)

            # write slot arrays per-parameter under 'optim.<slot>.<i>'
            for slot_name, lst in slots.items():
                for i, arr in enumerate(lst):
                    if arr is None:
                        continue
                    if not isinstance(arr, np.ndarray):
                        raise RuntimeError(f"Slot '{slot_name}[{i}]' is not an ndarray (got {type(arr).__name__})")
                    storage.write([arr], to_block_named=f"optim.{slot_name}.{i}", arrays_per_chunk=1)
                    _logger.debug("Wrote optim.%s.%d shape=%s dtype=%s",
                                slot_name, i, arr.shape, arr.dtype)

            # also write current parameter values to guarantee next-step reproducibility
            for i, param in enumerate(self.model_params):
                w = param.data
                storage.write([w], to_block_named=f"optim.param.{i}", arrays_per_chunk=1)
                _logger.debug("Wrote optim.param.%d shape=%s dtype=%s", i, w.shape, w.dtype)

    def load_state(self, pth: Path | str, *, strict: bool = True) -> None:
        """Restore optimizer hypers and per-param slots previously saved by save_state()."""
        _logger.info("Loading optimizer state into %s from %s", type(self).__name__, pth)
        n = len(self.model_params)

        with util.ArrayStorage(pth, mode="r") as storage:
            # read meta/hypers
            try:
                meta = storage.get_attr("optim.meta") or {}
            except Exception as e:
                if strict:
                    raise
                _logger.debug("No optim.meta (strict=False): %s", e)
                meta = {}

            # apply hypers (only those declared in HYPERS)
            src_hypers = meta.get("hypers", {})
            for name in self.HYPERS:
                if name in src_hypers:
                    cur = getattr(self, name, None)
                    val = src_hypers[name]
                    if isinstance(cur, (bool, int, float)) or cur is None:
                        try:
                            val = type(cur)(val) if cur is not None else val
                        except Exception:
                            pass
                    setattr(self, name, val)

            # read slots into new lists aligned to params
            for slot_name in self.SLOTS:
                dst_list: list[np.ndarray | None] = [None] * n
                for i in range(n):
                    key = f"optim.{slot_name}.{i}"
                    try:
                        arr = storage.read(key, 0)
                        # shape check against corresponding param
                        pshape = self.model_params[i].data.shape
                        if arr.shape != pshape:
                            msg = f"Slot '{slot_name}[{i}]' shape {arr.shape} != param {pshape}"
                            if strict:
                                raise ValueError(msg)
                            _logger.debug("%s (strict=False): skipping", msg)
                            arr = None
                        elif arr.dtype != self.model_params[i].data.dtype:
                            # coerce dtype to param dtype
                            arr = arr.astype(self.model_params[i].data.dtype, copy=False)
                        # ensure contiguous, owned copy to avoid view/mmapped subtleties
                        if arr is not None:
                            arr = np.array(arr, copy=True)
                        dst_list[i] = arr
                    except Exception as e:
                        if strict:
                            raise
                        _logger.debug("Missing %s (strict=False): %s", key, e)
                        dst_list[i] = None
                setattr(self, slot_name, dst_list)

            # restore parameters themselves to ensure identical starting point for next step
            for i in range(n):
                key = f"optim.param.{i}"
                try:
                    w = storage.read(key, 0)
                    p = self.model_params[i]
                    if w.shape != p.data.shape:
                        msg = f"Param '{key}' shape {w.shape} != expected {p.data.shape}"
                        if strict:
                            raise ValueError(msg)
                        _logger.debug("%s (strict=False): skipping", msg)
                        continue
                    if w.dtype != p.data.dtype:
                        w = w.astype(p.data.dtype, copy=False)
                    p.data = np.array(w, copy=True)
                except Exception as e:
                    if strict:
                        raise
                    _logger.debug("Missing %s (strict=False): %s", key, e)

class SGD(Optim):
    """Stochastic Gradient Descent with optional momentum and weight decay."""
    # --- checkpoint declaration ---
    HYPERS = ("lr", "beta", "weight_decay", "decoupled_wd")
    SLOTS  = ("_v",)

    def __init__(
        self,
        model_params: list[Tensor],
        lr: float,
        *,
        beta: float = 0.0,
        weight_decay: float = 0.0,
        decoupled_wd: bool = True
    ) -> None:  
        super().__init__(model_params, lr)

        self.beta = float(beta)
        self.weight_decay = float(weight_decay)
        self.decoupled_wd = bool(decoupled_wd)

        _logger.debug(
            "SGD init: params=%d, lr=%s, beta=%s, weight_decay=%s",
            len(self.model_params), self.lr, self.beta, self.weight_decay
        )

        self._v: list[np.ndarray | None] = [None] * len(self.model_params)

    def step(self) -> None:
        """Perform one in-place SGD (optionally with momentum/weight decay) update."""
        _logger.debug(
            "SGD.step: params=%d, lr=%s, beta=%s, weight_decay=%s, decoupled_wd=%s",
            len(self.model_params), self.lr, self.beta, self.weight_decay, self.decoupled_wd
        )
        use_mom = self.beta > 0.0
        wd = self.weight_decay
        lr = self.lr

        for i, param in enumerate(self.model_params):
            g = getattr(param, "grad", None)
            if g is None:
                _logger.debug("SGD.step: idx=%d has no grad; skipping", i)
                continue

            if wd > 0.0 and not self.decoupled_wd: # weight decay
                g = g + wd * param.data

            # momentum
            if use_mom:
                v = self._v[i]
                if (v is None) or (v.shape != param.data.shape):
                    v = np.zeros_like(param.data, dtype=param.data.dtype)
                    self._v[i] = v
                    _logger.debug("SGD.step: idx=%d, init v with shape=%s", i, v.shape)
                v[:] = ewma(v, g, beta=self.beta)
                update = v
            else:
                update = g

            if wd > 0.0 and self.decoupled_wd:
                param.data -= lr * wd * param.data

            param.data -= lr * update

class AdaGrad(Optim):
    """AdaGrad optimizer (per-parameter adaptive learning rates).

    Accumulates squared gradients r_i and scales updates by 1/sqrt(r_i + δ).

    Weight decay:
      - If decoupled_wd=False (default): classic L2 → g ← g + wd * w  (coupled)
      - If decoupled_wd=True: AdamW-style → w ← w - lr * wd * w, then gradient step (decoupled)
    """
    # --- checkpoint declaration ---
    HYPERS = ("lr", "weight_decay", "delta", "decoupled_wd")
    SLOTS  = ("_r",)

    def __init__(
        self,
        model_params: list[Tensor],
        lr: float,
        *,
        weight_decay: float = 0.0,
        delta: float = 1e-7,
        decoupled_wd: bool = True,
    ) -> None:
        """Initialize AdaGrad.

        Args:
            model_params: Trainable parameters to update.
            lr: Base learning rate.
            weight_decay: L2 coefficient.
            delta: Small epsilon for numerical stability inside sqrt.
            decoupled_wd: If True, apply AdamW-style decoupled weight decay.
        """
        super().__init__(model_params, lr)
        self.weight_decay = float(weight_decay)
        self.delta = float(delta)
        self.decoupled_wd = bool(decoupled_wd)
        self._r: list[np.ndarray | None] = [None] * len(self.model_params)

        _logger.debug(
            "AdaGrad init: params=%d, lr=%s, weight_decay=%s, delta=%s, decoupled_wd=%s",
            len(self.model_params), self.lr, self.weight_decay, self.delta, self.decoupled_wd
        )

    def step(self) -> None:
        """Apply one AdaGrad step to all parameters in-place."""
        wd = self.weight_decay
        lr = self.lr

        _logger.debug(
            "AdaGrad.step: params=%d, lr=%s, weight_decay=%s, delta=%s, decoupled_wd=%s",
            len(self.model_params), lr, wd, self.delta, self.decoupled_wd
        )

        for i, param in enumerate(self.model_params):
            g = getattr(param, "grad", None)
            if g is None:
                _logger.debug("AdaGrad.step: idx=%d has no grad; skipping", i)
                continue

            if wd > 0.0 and not self.decoupled_wd:
                _logger.debug("AdaGrad.step: idx=%d applying coupled L2 to grad", i)
                g = g + wd * param.data

            # ------- sq grads updates -------
            r = self._r[i]
            if (r is None) or (r.shape != param.data.shape):
                r = np.zeros_like(param.data, dtype=param.data.dtype)
                self._r[i] = r
                _logger.debug("AdaGrad.step: idx=%d, init r with shape=%s", i, r.shape)
            r[:] = r + g * g
            # --------------------------------

            if wd > 0.0 and self.decoupled_wd:
                _logger.debug("AdaGrad.step: idx=%d applying decoupled WD", i)
                param.data -= lr * wd * param.data

            # param update
            update = g / (np.sqrt(r) + self.delta)
            param.data -= lr * update

            _logger.debug(
                "AdaGrad.step: idx=%d, |g|=%s, mean(r)=%s",
                i,
                float(np.linalg.norm(g)),
                float(np.mean(r)) if r.size else 0.0,
            )

class RMSProp(Optim):
    """RMSProp optimizer.

    Keeps an exponential moving average of squared gradients r_i and scales
    each parameter's update by 1/sqrt(r_i + delta).

    Weight decay:
      - If decoupled_wd=False (default): classic L2 → g ← g + wd * w  (coupled)
      - If decoupled_wd=True: AdamW-style → w ← w - lr * wd * w, then gradient step (decoupled)
    """
    # --- checkpoint declaration ---
    HYPERS = ("lr", "weight_decay", "beta", "delta", "decoupled_wd")
    SLOTS  = ("_r",)

    def __init__(
        self,
        model_params: list[Tensor],
        lr: float,
        *,
        weight_decay: float = 0.0,
        beta: float = 0.9,
        delta: float = 1e-6,
        decoupled_wd: bool = True,
    ) -> None:
        """Initialize RMSProp.

        Args:
            model_params: Trainable parameters to update.
            lr: Base learning rate.
            weight_decay: L2 coefficient.
            beta: EMA decay for squared gradients.
            delta: Small epsilon for numerical stability.
            decoupled_wd: If True, apply AdamW-style decoupled weight decay.
        """
        super().__init__(model_params, lr)

        self.weight_decay = float(weight_decay)
        self.beta = float(beta)
        self.delta = float(delta)
        self.decoupled_wd = bool(decoupled_wd)
        self._r: list[np.ndarray | None] = [None] * len(self.model_params)

        _logger.debug(
            "RMSProp init: params=%d, lr=%s, beta=%s, weight_decay=%s, delta=%s, decoupled_wd=%s",
            len(self.model_params), self.lr, self.beta, self.weight_decay, self.delta, self.decoupled_wd
        )

    def step(self) -> None:
        """Apply one RMSProp step to all parameters in-place."""
        wd = self.weight_decay
        lr = self.lr

        _logger.debug(
            "RMSProp.step: params=%d, lr=%s, beta=%s, weight_decay=%s, delta=%s, decoupled_wd=%s",
            len(self.model_params), lr, self.beta, wd, self.delta, self.decoupled_wd
        )

        for i, param in enumerate(self.model_params):
            g = getattr(param, "grad", None)
            if g is None:
                _logger.debug("RMSProp.step: idx=%d has no grad; skipping", i)
                continue

            if wd > 0.0 and not self.decoupled_wd:
                _logger.debug("RMSProp.step: idx=%d applying coupled L2 to grad", i)
                g = g + wd * param.data

            # ------- sq grads updates -------
            r = self._r[i]
            if (r is None) or (r.shape != param.data.shape):
                r = np.zeros_like(param.data, dtype=param.data.dtype)
                self._r[i] = r
                _logger.debug("RMSProp.step: idx=%d init r with shape=%s", i, r.shape)
            g_sq = g * g
            r[:] = ewma(r, g_sq, beta=self.beta)
            # --------------------------------

            if wd > 0.0 and self.decoupled_wd:
                _logger.debug("RMSProp.step: idx=%d applying decoupled WD", i)
                param.data -= lr * wd * param.data

            # param update
            update = g / (np.sqrt(r) + self.delta)
            param.data -= lr * update

            _logger.debug(
                "RMSProp.step: idx=%d, grad_norm=%s, mean(r)=%s",
                i, float(np.linalg.norm(g)), float(np.mean(r)) if r.size else 0.0
            )

class Adam(Optim):
    """Adam optimizer with optional decoupled weight decay (AdamW).

    Maintains first moment (v) and second moment (r) estimates with bias correction.

    Weight decay:
      - If decoupled_wd=False (default): classic L2 → g ← g + wd * w  (coupled)
      - If decoupled_wd=True: AdamW-style → w ← w - lr * wd * w, then Adam update (decoupled)
    """
    # --- checkpoint declaration ---
    HYPERS = ("lr", "weight_decay", "beta1", "beta2", "delta", "decoupled_wd", "_t")
    SLOTS  = ("_v", "_r")

    def __init__(
        self,
        model_params: list[Tensor],
        lr: float,
        *,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        delta: float = 1e-8,
        decoupled_wd: bool = True,
    ) -> None:
        """Initialize Adam/AdamW.

        Args:
            model_params: Trainable parameters to update.
            lr: Base learning rate.
            weight_decay: L2 coefficient.
            beta1: Exponential decay for first moment.
            beta2: Exponential decay for second moment.
            delta: Numerical stability epsilon.
            decoupled_wd: If True, apply AdamW-style decoupled weight decay.
        """
        super().__init__(model_params, lr)
        self.weight_decay = float(weight_decay)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.delta = float(delta)
        self.decoupled_wd = bool(decoupled_wd)
        self._v: list[np.ndarray | None] = [None] * len(self.model_params)  # first moment
        self._r: list[np.ndarray | None] = [None] * len(self.model_params)  # second moment
        self._t: int = 0  # persistent global step counter

        _logger.debug(
            "Adam init: params=%d, lr=%s, beta1=%s, beta2=%s, weight_decay=%s, delta=%s, decoupled_wd=%s",
            len(self.model_params), self.lr, self.beta1, self.beta2, self.weight_decay, self.delta, self.decoupled_wd
        )

    def step(self) -> None:
        """Apply one Adam/AdamW step to all parameters in-place."""
        wd = self.weight_decay
        lr = self.lr

        # increment global step once per optimizer step
        self._t += 1
        t = self._t

        _logger.debug(
            "Adam.step: t=%d, params=%d, lr=%s, beta1=%s, beta2=%s, weight_decay=%s, delta=%s, decoupled_wd=%s",
            t, len(self.model_params), lr, self.beta1, self.beta2, wd, self.delta, self.decoupled_wd
        )

        for i, param in enumerate(self.model_params):
            g = getattr(param, "grad", None)
            if g is None:
                _logger.debug("Adam.step: idx=%d has no grad; skipping", i)
                continue

            if wd > 0.0 and not self.decoupled_wd:
                _logger.debug("Adam.step: idx=%d applying coupled L2 to grad", i)
                g = g + wd * param.data

            # --- first moment (v_t) ---
            v = self._v[i]
            if (v is None) or (v.shape != param.data.shape):
                v = np.zeros_like(param.data, dtype=param.data.dtype)
                self._v[i] = v
                _logger.debug("Adam.step: idx=%d init v with shape=%s", i, v.shape)
            v[:] = ewma(v, g, beta=self.beta1)

            # --- second moment (r_t) ---
            r = self._r[i]
            if (r is None) or (r.shape != param.data.shape):
                r = np.zeros_like(param.data, dtype=param.data.dtype)
                self._r[i] = r
                _logger.debug("Adam.step: idx=%d init r with shape=%s", i, r.shape)
            r[:] = ewma(r, g * g, beta=self.beta2)

            # bias correction
            v_hat = v / (1.0 - self.beta1**t)
            r_hat = r / (1.0 - self.beta2**t)

            if wd > 0.0 and self.decoupled_wd:
                _logger.debug("Adam.step: idx=%d applying decoupled WD", i)
                param.data -= lr * wd * param.data

            # parameter update
            param.data -= lr * (v_hat / (np.sqrt(r_hat) + self.delta))

# *----------------------------------------------------*
#                     LR Schedulers
# *----------------------------------------------------*

class LRScheduler:
    def __init__(self, optim: Optim, *, last_step: int = -1) -> None:
        self.optim = optim
        self.base_lr = float(optim.lr)
        self.last_step = int(last_step)
        self.lr = float(optim.lr)

    def compute_lr(self, step: int) -> float:
        raise NotImplementedError

    def step(self, n: int = 1) -> float:
        self.last_step += int(n)
        new_lr = float(self.compute_lr(self.last_step))
        self.optim.lr = new_lr
        self.lr = new_lr
        return new_lr

    def state_dict(self) -> dict:
        return {
            "last_step": self.last_step,
            "base_lr": self.base_lr,
            "class": type(self).__name__,
        }

    def load_state_dict(self, state: dict) -> None:
        self.last_step = int(state.get("last_step", self.last_step))
        self.base_lr = float(state.get("base_lr", self.base_lr))
        # keep `self.lr` consistent with current step
        self.lr = float(self.compute_lr(self.last_step))
        self.optim.lr = self.lr

    def save_state(self, pth: Path | str, *, compression_level: int = 3) -> None:
        """Save scheduler state_dict() into an ArrayStorage archive."""
        pth = Path(pth).with_suffix(".pureml.zip")
        _logger.info("Saving LR scheduler state (%s) to %s", type(self).__name__, pth)
        with util.ArrayStorage.compress_and_cleanup(pth, compression_level) as storage:
            storage.add_attr("sched.meta", self.state_dict())

    def load_state(self, pth: Path | str, *, strict: bool = True) -> None:
        """Load scheduler state saved by save_state()."""
        _logger.info("Loading LR scheduler state into %s from %s", type(self).__name__, pth)
        with util.ArrayStorage(pth, mode="r") as storage:
            try:
                meta = storage.get_attr("sched.meta") or {}
            except Exception as e:
                if strict:
                    raise
                _logger.debug("No sched.meta (strict=False): %s", e)
                return
            cls = meta.get("class")
            if strict and cls and cls != type(self).__name__:
                raise ValueError(f"Scheduler class mismatch: saved={cls}, current={type(self).__name__}")
            self.load_state_dict(meta)

class StepLR(LRScheduler):
    """Step decay: lr = base_lr * gamma ** floor(step / step_size)."""

    def __init__(self, optim: Optim, step_size: int, *, gamma: float = 0.1, last_step: int = -1) -> None:
        """Args:
            optim: Optimizer whose lr will be scheduled.
            step_size: Number of steps between decays (> 0).
            gamma: Multiplicative decay factor applied every `step_size` steps.
            last_step: Last processed step; -1 means “start before first step”.
        """
        super().__init__(optim, last_step=last_step)
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        _logger.debug("StepLR init: base_lr=%s, step_size=%d, gamma=%s, last_step=%d",
                      self.base_lr, self.step_size, self.gamma, self.last_step)

    def compute_lr(self, step: int) -> float:
        """Compute LR at a given step using piecewise-constant step decay."""
        k = step // self.step_size if step >= 0 else -1  # keep base lr until first step
        lr = self.base_lr * (self.gamma ** max(k, 0))
        _logger.debug("StepLR.compute_lr: step=%d, k=%d, lr=%s", step, k, lr)
        return lr

class ExponentialLR(LRScheduler):
    """Exponential decay: lr = base_lr * gamma ** step (clamped for step<0)."""

    def __init__(self, optim: Optim, *, gamma: float, last_step: int = -1) -> None:
        """Args:
            optim: Optimizer whose lr will be scheduled.
            gamma: Per-step multiplicative factor.
            last_step: Last processed step; -1 means “start before first step”.
        """
        super().__init__(optim, last_step=last_step)
        self.gamma = float(gamma)
        _logger.debug("ExponentialLR init: base_lr=%s, gamma=%s, last_step=%d",
                      self.base_lr, self.gamma, self.last_step)

    def compute_lr(self, step: int) -> float:
        """Compute LR at a given step using smooth exponential decay."""
        lr = self.base_lr * (self.gamma ** max(step, 0))
        _logger.debug("ExponentialLR.compute_lr: step=%d, lr=%s", step, lr)
        return lr

class CosineAnnealingLR(LRScheduler):
    """Cosine annealing from base_lr to eta_min over T_max steps (no restarts)."""

    def __init__(self, optim: Optim, *, T_max: int, eta_min: float = 0.0, last_step: int = -1) -> None:
        """Args:
            optim: Optimizer whose lr will be scheduled.
            T_max: Number of steps to anneal over (> 0).
            eta_min: Final/minimum learning rate at the end of the anneal.
            last_step: Last processed step; -1 means “start before first step”.
        """
        super().__init__(optim, last_step=last_step)
        if T_max <= 0:
            raise ValueError("T_max must be > 0")
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        _logger.debug("CosineAnnealingLR init: base_lr=%s, T_max=%d, eta_min=%s, last_step=%d",
                      self.base_lr, self.T_max, self.eta_min, self.last_step)

    def compute_lr(self, step: int) -> float:
        """Compute LR at a given step using half-cosine schedule (clamped after T_max)."""
        s = 0 if step <= 0 else (self.T_max if step >= self.T_max else step)
        # progress in [0, 1]
        x = s / self.T_max
        lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + np.cos(np.pi * x))
        _logger.debug("CosineAnnealingLR.compute_lr: step=%d, s=%d, x=%.6f, lr=%s",
                      step, s, x, lr)
        return lr

__all__ = [
    "SGD",
    "AdaGrad",
    "RMSProp",
    "Adam",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR"
]

if __name__ == "__main__":
    pass
