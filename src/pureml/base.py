"""Model bases and state helpers.

`BaseModel`/`NN` define the fit/predict contract, training/eval mode propagation across contained
`Layer`s, collection of parameters/buffers, and ArrayStorage-backed save/load of params or full
state (including literals). Helpers gather named params/buffers and provide convenience loaders
to resume from `.pureml.zip` artifacts."""
from __future__ import annotations

# built-in
from pathlib import Path
import logging
from typing import Any, overload
from collections.abc import Mapping, Sequence
# local
from .training_utils import Dataset
from .machinery import Tensor
from .layers import Layer
from . import util

_logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for models.
    Subclasses must implement .fit and .predict.
    """

    @overload
    def fit(self, X: Tensor, Y: Tensor, *args, **kwargs) -> None: ...
    
    @overload
    def fit(self, dataset: Dataset, *args, **kwargs) -> None: ...

    def fit(self, *args, **kwargs) -> None:
        """Train the model on (X, Y) or on a Dataset."""
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def state(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Return (literals, layers) where:
            literals: {field_name: JSON-safe literal} for top-level, non-layer fields
            layers: {
                "<attr>": {
                    "tunable": tuple[np.ndarray, ...],
                    "buffers": { "<bname>": np.ndarray, ... }
                }, ...
            }
        """
        literals: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Layer):
                continue
            if util.is_json_literal(v):
                literals[k] = v

        layers: dict[str, dict[str, Any]] = {}
        for lname, layer in ((name, obj) for name, obj in self.__dict__.items() if isinstance(obj, Layer)):
            tensor_arrays = tuple(t.data for t in layer.parameters)
            buffer_arrays: dict[str, Any] = {}
            for bname, buf in layer.named_buffers().items():
                buffer_arrays[bname] = (buf.data if isinstance(buf, Tensor) else buf)
            layers[lname] = {"tunable": tensor_arrays, "buffers": buffer_arrays}
        return literals, layers
    
    def save_state(self, pth: Path | str, *, compression_level: int = 3) -> None:
        raise NotImplementedError

    def load_state(self, pth: Path | str, **kwargs) -> BaseModel:
        raise NotImplementedError

def get_mdl_params(mdl: BaseModel) -> dict[str, Tensor]:
    """Collect learnable parameters from all `Layer` attributes on `mdl`."""
    params = {}
    for p_name, p in mdl.__dict__.items():
        if isinstance(p, Layer):
            for i, t in enumerate(p.parameters):
                params[f"{p_name}.{i}"] = t
    _logger.debug("Collected %d parameters from model %s", len(params), mdl.__class__.__name__)
    return params

def get_mdl_named_buffers(mdl: BaseModel) -> dict[str, Any]:
    """Collect non-trainable buffers from `Layer` attributes on `mdl`.

    Returns:
        dict: flat mapping "<attr>.buf.<name>" -> Tensor/ndarray
    """
    bufs: dict[str, Any] = {}
    for lname, layer in ((name, obj) for name, obj in mdl.__dict__.items() if isinstance(obj, Layer)):
        for bname, buf in layer.named_buffers().items():
            bufs[f"{lname}.buf.{bname}"] = buf
    _logger.debug("Collected %d buffers from model %s", len(bufs), mdl.__class__.__name__)
    return bufs

def save_mdl_params(params_or_mdl, pth: Path | str, *, compression_level: int = 3):
    """Save model parameters to a compressed ArrayStorage zip.
    Accepts either a dict[str, Tensor/ndarray] or a model instance.
    """
    pth = Path(pth).with_suffix(".pureml.zip")
    if isinstance(params_or_mdl, BaseModel):
        params = get_mdl_params(params_or_mdl)
    else:
        params = params_or_mdl
    _logger.info("Saving %d parameters to %s (clevel=%d)", len(params), pth, compression_level)
    with util.ArrayStorage.compress_and_cleanup(pth, compression_level) as storage:
        for p_name, p in params.items():
            arr = p.data if isinstance(p, Tensor) else p
            storage.write([arr], to_block_named=p_name, arrays_per_chunk=1)
            _logger.debug("Wrote parameter block '%s' with shape=%s dtype=%s",
                          p_name, getattr(arr, "shape", "?"), getattr(arr, "dtype", "?"))

def save_full_state(mdl: BaseModel, pth: Path | str, *, compression_level: int = 3) -> None:
    """Save trainable params, non-trainable buffers, and top-level literals."""
    pth = Path(pth).with_suffix(".pureml.zip")
    literals, layers = mdl.state()

    n_params = sum(len(pack["tunable"]) for pack in layers.values())
    n_bufs   = sum(len(pack["buffers"]) for pack in layers.values())
    _logger.info("Saving FULL state of %s to %s (params=%d, buffers=%d)",
                 mdl.__class__.__name__, pth, n_params, n_bufs)
    with util.ArrayStorage.compress_and_cleanup(pth, compression_level) as storage:
        
        for lname, pack in layers.items():
            for i, arr in enumerate(pack["tunable"]):
                storage.write([arr], to_block_named=f"{lname}.param.{i}", arrays_per_chunk=1)
            for bname, arr in pack["buffers"].items():
                storage.write([arr], to_block_named=f"{lname}.buf.{bname}", arrays_per_chunk=1)

        storage.add_attr("meta", {"kind": "NNState"}) # can add more if need be in the future
        storage.add_attr("model_class", mdl.__class__.__name__)
        storage.add_attr("literals", literals)

def load_state(mdl: BaseModel, pth: Path | str, *, strict: bool = True, load_literals: bool = True) -> BaseModel:
    """Module-level loader.
    If the archive has meta.kind=='NNState' → delegate to NN.load_state (full).
    Otherwise → treat as params-only and call NN.load_params.
    """ 
    pth = Path(pth)
    meta = None
    try:
        with util.ArrayStorage(pth, mode="r") as storage:
            meta = storage.get_attr("meta")
    except Exception as e:
        _logger.debug("load_state: could not read meta (%s); assuming params-only", e)
    if isinstance(meta, dict) and meta.get("kind") == "NNState":
        return mdl.load_state(pth, strict=strict, load_literals=load_literals)
    return mdl.load_params(pth)

class NN(BaseModel):
    """Neural network base class."""

    @property
    def training(self) -> bool:
        # default to True if never explicitly set
        return getattr(self, "_training", True)

    def train(self) -> NN:
        """Switch the model (and all child Layers) to training mode."""
        self._set_mode(True)
        return self

    def eval(self) -> NN:
        """Switch the model (and all child Layers) to eval mode."""
        self._set_mode(False)
        return self

    def _set_mode(self, mode: bool) -> None:
        self._training = bool(mode)
        # propagate to all Layer instances found in attributes/containers
        for layer in self._iter_layers():
            layer.training = mode  # triggers each layer's on_mode_change hook

    def _iter_layers(self):
        """Yield all Layer instances reachable from self.__dict__ (1-level containers OK)."""
        def walk(obj):
            if isinstance(obj, Layer):
                yield obj
            elif isinstance(obj, Mapping):
                for v in obj.values():
                    yield from walk(v)
            elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
                for v in obj:
                    yield from walk(v)
        for v in self.__dict__.values():
            yield from walk(v)

    def load_params(self, param_pth: Path | str) -> NN:
        """Load parameters from an ArrayStorage into this model."""
        _logger.info("Loading parameters for %s from %s", self.__class__.__name__, param_pth)
        with util.ArrayStorage(param_pth, mode="r") as storage:
            for p_name, p in self.__dict__.items():
                if isinstance(p, Layer):
                    for i, t in enumerate(p.parameters):
                        data = storage.read(f"{p_name}.{i}", 0)
                        _logger.debug("Loaded '%s' with shape=%s dtype=%s",
                                      f"{p_name}.{i}", data.shape, data.dtype)
                        t.data = data
        return self

    @property
    def parameters(self):
        return [
            param
            for v in self.__dict__.values()
            if isinstance(v, Layer)
            for param in v.parameters
        ]

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.predict(*args, **kwargs)

    def save(self, pth: Path | str, *, compression_level: int = 3) -> None:
        """Save ONLY trainable parameters (back-compat)."""
        _logger.info("Saving model %s parameters to %s", self.__class__.__name__, pth)
        save_mdl_params(get_mdl_params(self), pth, compression_level=compression_level)

    def save_state(self, pth: Path | str, *, compression_level: int = 3) -> None:
        """Save params + buffers + top-level literals into one archive."""
        save_full_state(self, pth, compression_level=compression_level)

    def load_state(
        self,
        pth: Path | str,
        *,
        strict: bool = True,
        load_literals: bool = True,
    ) -> NN:
        """Load a full state previously saved by `save_state`."""
        _logger.info("Loading FULL state for %s from %s", self.__class__.__name__, pth)
        with util.ArrayStorage(pth, mode="r") as storage:
            
            # 1) literals
            if load_literals:
                try:
                    lits = storage.get_attr("literals") or {}
                    for k, v in lits.items():
                        if hasattr(self, k):
                            setattr(self, k, v)
                        else:
                            # Missing literals (e.g., optional attributes) should not block loading.
                            _logger.debug("Skipping literal '%s' (no attr on target)", k)
                except Exception as e:
                    if strict:
                        raise
                    _logger.debug("No/invalid literals (strict=False): %s", e)

            # 2) per-layer params & buffers
            for lname, layer in ((name, obj) for name, obj in self.__dict__.items() if isinstance(obj, Layer)):
                
                # params
                p_arrays: list[object] = []
                num_params = len(layer.parameters)
                for i in range(num_params):
                    key = f"{lname}.param.{i}"
                    try:
                        arr = storage.read(key, 0)
                        # shape guard before accepting the array
                        exp = layer.parameters[i].data.shape
                        if arr.shape != exp:
                            msg = f"{key} shape {arr.shape} != expected {exp}"
                            if strict:
                                raise ValueError(msg)
                            _logger.debug("%s (strict=False): skipping", msg)
                            arr = None
                        p_arrays.append(arr)
                    except Exception as e:
                        if strict:
                            raise
                        _logger.debug("Missing %s (strict=False): %s", key, e)
                        p_arrays.append(None)
                
                # buffers (only those declared by the layer)
                buf_arrays: dict[str, object] = {}
                for bname in layer.named_buffers().keys():
                    key = f"{lname}.buf.{bname}"
                    try:
                        arr = storage.read(key, 0)
                        # optional buffer shape guard
                        cur = layer.named_buffers().get(bname)
                        if hasattr(cur, "data"):
                            exp = cur.data.shape
                            if arr.shape != exp:
                                msg = f"{key} shape {arr.shape} != expected {exp}"
                                if strict:
                                    raise ValueError(msg)
                                _logger.debug("%s (strict=False): skipping", msg)
                                arr = None
                        buf_arrays[bname] = arr
                    except Exception as e:
                        if strict:
                            raise
                        _logger.debug("Missing %s (strict=False): %s", key, e)
                        buf_arrays[bname] = None
                
                # delegate to layer
                layer.apply_state(tunable=tuple(p_arrays), buffers=buf_arrays)

        return self


__all__ = [
    "BaseModel",
    "NN",
    "get_mdl_params",
    "get_mdl_named_buffers",
    "save_mdl_params",
    "save_full_state",
    "load_state"
]
