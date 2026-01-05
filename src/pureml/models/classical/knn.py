"""Simple k-Nearest Neighbors classifier built on PureML tensors.

Supports configurable k, distance function (default Euclidean from `general_math`),
and optional per-feature standardization. Uses `no_grad` for inference and resolves
ties by closest distance among tied labels."""
from __future__ import annotations

# third party
import numpy as np
# built in
from typing import Callable
# local
from ...machinery import Tensor, no_grad
from ...general_math import euclidean_distance, mean, std
from ...base import BaseModel

class KNN(BaseModel):

    def __init__(self,
                 k: int,
                 d: Callable[[Tensor, Tensor], Tensor] = euclidean_distance,
                 standardize_features: bool = True) -> None:
        """Initializes a KNN classifier

        Args:
            k (int): number of nearest neighbors voting
            d (Callable, optional): distance function for finding nearest neighbors.
                Defaults to Euclidean distance (Tensor -> Tensor).
            standardize_features (bool, optional): if True, z-score per feature. Defaults to True.
        """
        self.k = k
        self.d = d
        self.standardize_features = standardize_features

    # ---------------------------------
    # Helper methods
    # ---------------------------------

    def _vote(self, neigh_labels: Tensor) -> int:
        neigh_labels = neigh_labels.data
        # find the most frequent class of the nearest neighbors
        # and resolve any ties if they occur by picking the closest class by distance
        labels, counts = np.unique(neigh_labels, return_counts=True)
        winners = labels[counts == counts.max()]
        if winners.size == 1:
            return winners[0]
        for lbl in neigh_labels:
            if lbl in winners:
                return lbl

    # ---------------------------------
    # Public API
    # ---------------------------------

    def fit(self, X: Tensor, Y: Tensor) -> KNN:
        """Stores the samples data along with their labels

        Args:
            X (Tensor): m x n matrix where m is the number of n-feature samples
            Y (Tensor): m x 1 (or m,) labels corresponding to samples from X
        """
        m, _ = X.shape
        assert 1 <= self.k <= m, "k must be between 1 and number of samples"
        assert Y.shape[0] == m, "X and Y must have the same number of samples"

        if self.standardize_features:
            # compute per-feature stats (reduce over samples) without tracking grads
            with no_grad():
                mu = mean(X.T)   # (n,)
                sd = std(X.T)    # (n,)
            self.means = mu.data
            self.stds  = sd.data
            eps = 1e-12
            self.X = (X.data - self.means) / (self.stds + eps)
        else:
            self.X = X.data

        self.Y = Y.data
        return self
        
    def predict(self, x_q: Tensor) -> int:
        """Predicts the class the x_q data point belongs to

        Args:
            x_q (Tensor): data point to be classified; dimension must equal
                the number of features of the training samples

        Returns:
            predicted class (int)
        """
        xq = x_q.data
        assert xq.shape[0] == self.X.shape[1], "Query must have same # of features"

        if self.standardize_features:
            eps = 1e-12
            xq = (xq - self.means) / (self.stds + eps)

        # distance wrapper (Tensor -> scalar), done under no_grad
        def _dist_row(row: np.ndarray) -> float:
            with no_grad():
                return float(self.d(Tensor(row), Tensor(xq)).data)

        distances = np.apply_along_axis(_dist_row, 1, self.X)
        neigh_labels = self.Y[np.argsort(distances)[:self.k]]
        return self._vote(Tensor(neigh_labels))


__all__ = [
    "KNN"
]

if __name__ == "__main__":
    pass
