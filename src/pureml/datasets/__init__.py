"""Dataset wrappers packaged with PureML (currently MNIST via bundled Zarr archive)."""
from .MNIST import MnistDataset

__all__ = [
    "MnistDataset"
]
