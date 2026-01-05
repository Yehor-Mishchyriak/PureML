"""MNIST dataset reader backed by the packaged Zarr archive.

Uses `importlib.resources.files` to locate the zipped store bundled in the wheel, opens it via
ArrayStorage in read-only mode, and returns Tensor pairs: images scaled to float32 in [0,1] and
one-hot labels for train (class indices for test). Implements context manager to close the store."""
import numpy as np
from typing import Literal
from importlib.resources import files, as_file

from ...training_utils import Dataset, one_hot
from ...util import ArrayStorage
from ...machinery import Tensor


class MnistDataset(Dataset):
    def __init__(self, mode: Literal["train", "test"] = "train"):
        self.mode = mode

        if self.mode == "train":
            self.x_key, self.y_key = "train_images", "train_labels"
        elif self.mode == "test":
            self.x_key, self.y_key = "test_images", "test_labels"
        else:
            raise ValueError(f"'mode' must be 'train' or 'test'; Instead, {mode} was given")

        self._mnist_res = files("pureml.datasets.MNIST.files").joinpath(
            "mnist-28x28_uint8.zarr.zip"
        )
        self._mnist_cm = as_file(self._mnist_res)
        self._mnist_path = str(self._mnist_cm.__enter__())

        self.store = ArrayStorage(self._mnist_path, mode="r")
        self.N = self.store.root[self.x_key].shape[0]

        self.X = self.store.read(self.x_key)
        self.y = self.store.read(self.y_key)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return (
            Tensor(X.astype(np.float32) / 255.0, requires_grad=False),
            one_hot(dims=10, label=Tensor(y, requires_grad=False)) if self.mode == "train"
            else Tensor(y, requires_grad=False)
        )

    def close(self):
        self.store.close()
        if hasattr(self, "_mnist_cm") and self._mnist_cm is not None:
            self._mnist_cm.__exit__(None, None, None)
            self._mnist_cm = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


__all__ = ["MnistDataset"]
