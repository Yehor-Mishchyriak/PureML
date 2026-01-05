"""MNIST dataset loader packaged with PureML (uses bundled Zarr zip via importlib.resources)."""
from .dataset import MnistDataset

__all__ = [
    "MnistDataset"
]
