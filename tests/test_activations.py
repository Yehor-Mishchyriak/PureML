# activations
import unittest as ut
import numpy as np

import pureml.activations as act
from pureml.machinery import Tensor


def _rng(seed=0):
    return np.random.default_rng(seed)


class TestSigmoid(ut.TestCase):
    def test_forward_range_and_shape(self):
        X = Tensor(np.linspace(-6, 6, 25).reshape(5, 5), requires_grad=False)
        Y = act.sigmoid(X)
        self.assertEqual(Y.data.shape, X.data.shape)
        self.assertTrue(np.all(Y.data > 0) and np.all(Y.data < 1))

    def test_backward_matches_formula(self):
        X = Tensor(np.linspace(-3, 3, 31), requires_grad=True)
        Y = act.sigmoid(X)
        Y.backward()  # upstream ones
        s = Y.data
        grad_expected = s * (1.0 - s)
        np.testing.assert_allclose(X.grad, grad_expected, rtol=1e-6, atol=1e-8)


class TestReLU(ut.TestCase):
    def test_forward_nonnegativity(self):
        X = Tensor(np.array([[-2.0, -0.1, 0.0, 0.3, 5.0]]), requires_grad=False)
        Y = act.relu(X)
        np.testing.assert_array_equal(Y.data, np.array([[0.0, 0.0, 0.0, 0.3, 5.0]]))

    def test_backward_piecewise_linear(self):
        # Avoid x == 0 exactly to avoid ambiguity
        X = Tensor(np.array([-2.0, -1.0, -0.5, 0.1, 2.0, 7.0]), requires_grad=True)
        Y = act.relu(X)
        Y.backward()  # upstream ones
        grad_expected = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        np.testing.assert_allclose(X.grad, grad_expected, rtol=1e-6, atol=1e-8)


class TestTanh(ut.TestCase):
    def test_forward_range(self):
        X = Tensor(np.linspace(-4, 4, 41), requires_grad=False)
        Y = act.tanh(X)
        self.assertTrue(np.all(Y.data > -1) and np.all(Y.data < 1))

    def test_backward_matches_formula(self):
        X = Tensor(np.linspace(-3, 3, 31), requires_grad=True)
        Y = act.tanh(X)
        Y.backward()
        t = Y.data
        grad_expected = 1.0 - t * t
        np.testing.assert_allclose(X.grad, grad_expected, rtol=1e-6, atol=1e-8)


class TestSoftmax(ut.TestCase):
    def test_sums_to_one_axis_last(self):
        rng = _rng(1)
        X = Tensor(rng.standard_normal((8, 5)), requires_grad=False)
        S = act.softmax(X, axis=-1)
        np.testing.assert_allclose(S.data.sum(axis=-1), np.ones(8), rtol=1e-7, atol=1e-9)
        self.assertEqual(S.data.shape, X.data.shape)

    def test_shift_invariance(self):
        rng = _rng(2)
        X = rng.standard_normal((6, 7))
        c = rng.standard_normal((6, 1))  # broadcastable offset per row
        S1 = act.softmax(Tensor(X), axis=-1).data
        S2 = act.softmax(Tensor(X + c), axis=-1).data
        np.testing.assert_allclose(S1, S2, rtol=1e-7, atol=1e-9)

    def test_backward_jvp_axis_last(self):
        rng = _rng(3)
        X = Tensor(rng.standard_normal((4, 6)), requires_grad=True)
        S = act.softmax(X, axis=-1)
        U = rng.standard_normal((4, 6))  # upstream gradient
        S.backward(U)

        s = S.data
        dot = np.sum(U * s, axis=-1, keepdims=True)
        grad_expected = (U - dot) * s
        np.testing.assert_allclose(X.grad, grad_expected, rtol=1e-6, atol=1e-8)

    def test_backward_jvp_axis_middle_on_3d(self):
        rng = _rng(4)
        Xv = rng.standard_normal((2, 3, 4))
        X = Tensor(Xv, requires_grad=True)
        axis = 1
        S = act.softmax(X, axis=axis)
        U = rng.standard_normal((2, 3, 4))
        S.backward(U)

        s = S.data
        dot = np.sum(U * s, axis=axis, keepdims=True)
        grad_expected = (U - dot) * s
        np.testing.assert_allclose(X.grad, grad_expected, rtol=1e-6, atol=1e-8)

    def test_numerical_stability_large_values(self):
        rng = _rng(5)
        X = Tensor(1000.0 * rng.standard_normal((3, 7)))  # huge magnitude
        S = act.softmax(X, axis=-1)
        # Finite, sums to one
        self.assertTrue(np.all(np.isfinite(S.data)))
        np.testing.assert_allclose(S.data.sum(axis=-1), np.ones(3), rtol=1e-6, atol=1e-8)


class TestLogSoftmax(ut.TestCase):
    def test_exp_logsoftmax_equals_softmax(self):
        rng = _rng(6)
        X = Tensor(rng.standard_normal((5, 9)))
        L = act.log_softmax(X, axis=-1).data
        S = act.softmax(X, axis=-1).data
        np.testing.assert_allclose(np.exp(L), S, rtol=1e-7, atol=1e-9)

    def test_numerical_stability_large_values(self):
        rng = _rng(7)
        X = Tensor(1000.0 * rng.standard_normal((4, 6)))
        L = act.log_softmax(X, axis=-1).data
        self.assertTrue(np.all(np.isfinite(L)))  # should not overflow/underflow

    def test_backward_jvp(self):
        rng = _rng(8)
        axis = -1
        X = Tensor(rng.standard_normal((3, 5)), requires_grad=True)
        L = act.log_softmax(X, axis=axis)
        U = rng.standard_normal((3, 5))
        L.backward(U)

        s = act.softmax(Tensor(X.data), axis=axis).data  # reuse forward softmax for expected grad
        sumU = np.sum(U, axis=axis, keepdims=True)
        grad_expected = U - sumU * s
        np.testing.assert_allclose(X.grad, grad_expected, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    ut.main(verbosity=2)
