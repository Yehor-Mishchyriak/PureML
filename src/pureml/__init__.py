"""PureML: NumPy-only autodiff, layers, and minimal DL utilities with bundled datasets/models.

Exposes the core Tensor/Function engine, helper utilities (storage, batching, RNG), reference
layers/models/datasets, and logging helpers. Modules are lazily imported via __getattr__."""
# pureml/__init__.py
from .machinery import (
    Tensor, TensorValuedFunction, GradientNotDefined,
    is_grad_enabled, no_grad,
)
from .util import ArrayStorage, compose_steps, batches_of

__version__ = "1.2.8"

# expose subpackages on demand
def __getattr__(name):
    if name in {
        "layers","activations","losses","optimizers",
        "base","general_math","training_utils","datasets","models"
    }:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)

__all__ = [
    # core
    "Tensor","TensorValuedFunction","GradientNotDefined",
    "is_grad_enabled","no_grad",
    # utils
    "ArrayStorage","compose_steps","batches_of",
    # namespaces (lazy) (see above)
    "layers","activations","losses","optimizers",
    "base","general_math","training_utils","datasets","models",
]
