# knn
import unittest as ut
import numpy as np

from pureml.machinery import Tensor
from pureml.models.classical.knn import KNN


def _to_label(pred):
    """Coerce model output to a Python int (handles Tensor scalar or int)."""
    if isinstance(pred, Tensor):
        arr = pred.data
        return int(np.asarray(arr).ravel()[0])
    return int(pred)


def manhattan(a: Tensor, b: Tensor) -> Tensor:
    """L1 distance in the required signature."""
    return Tensor(np.abs(a.data - b.data).sum(), requires_grad=False)


class TestKNN(ut.TestCase):
    def test_k1_nearest_neighbor_1d(self):
        # Points at 0 (label 0) and 5 (label 1)
        X = Tensor(np.array([[0.0], [5.0]], dtype=np.float32))
        y = Tensor(np.array([0, 1]))
        knn = KNN(k=1, standardize_features=False).fit(X, y)

        self.assertEqual(_to_label(knn.predict(Tensor(np.array([4.9], dtype=np.float32)))), 1)
        self.assertEqual(_to_label(knn.predict(Tensor(np.array([0.2], dtype=np.float32)))), 0)

    def test_k3_majority_vote_1d(self):
        # Neighbors near 1.5 are {1(0),2(1),0(0)} → majority label 0
        X = Tensor(np.array([[0.0], [1.0], [2.0], [10.0]], dtype=np.float64))
        y = Tensor(np.array([0, 0, 1, 1]))
        knn = KNN(k=3, standardize_features=False).fit(X, y)

        self.assertEqual(_to_label(knn.predict(Tensor(np.array([1.5], dtype=np.float64)))), 0)

    def test_custom_distance_changes_decision(self):
        # Query q = (-3,-3). Candidates:
        # A = (-3, 0), label 0 → dE=3.0, dM=3
        # B = (-1,-1), label 1 → dE≈2.828, dM=4
        # Euclidean picks B (label 1); Manhattan picks A (label 0).
        X = Tensor(np.array([[-3.0,  0.0],
                             [-1.0, -1.0]], dtype=np.float64))
        y = Tensor(np.array([0, 1]))
        q = Tensor(np.array([-3.0, -3.0], dtype=np.float64))

        pred_eu = _to_label(KNN(k=1, standardize_features=False).fit(X, y).predict(q))
        self.assertEqual(pred_eu, 1)

        pred_ma = _to_label(KNN(k=1, d=manhattan, standardize_features=False).fit(X, y).predict(q))
        self.assertEqual(pred_ma, 0)


if __name__ == "__main__":
    ut.main(verbosity=2)
