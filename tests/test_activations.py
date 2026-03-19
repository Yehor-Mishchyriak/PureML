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


class TestLeakyReLU(ut.TestCase):
    def test_forward_matches_piecewise_default_slope(self):
        Xv = np.array([[-3.0, -1.0, 0.0, 2.0, 4.0]], dtype=np.float64)
        X = Tensor(Xv, requires_grad=False)
        Y = act.leaky_relu(X)  # default slope=0.01

        expected = np.where(Xv > 0.0, Xv, 0.01 * Xv)
        self.assertEqual(Y.data.shape, Xv.shape)
        np.testing.assert_allclose(Y.data, expected, rtol=1e-7, atol=1e-9)

    def test_forward_matches_piecewise_custom_slope(self):
        rng = _rng(9)
        slope = 0.2
        Xv = rng.standard_normal((3, 4, 5))
        X = Tensor(Xv, requires_grad=False)
        Y = act.leaky_relu(X, negative_slope=slope)

        expected = np.where(Xv > 0.0, Xv, slope * Xv)
        np.testing.assert_allclose(Y.data, expected, rtol=1e-7, atol=1e-9)

    def test_backward_matches_formula_default_slope_with_custom_upstream(self):
        rng = _rng(10)
        Xv = np.array([-2.0, -0.5, 0.0, 0.3, 1.7], dtype=np.float64)
        X = Tensor(Xv, requires_grad=True)
        Y = act.leaky_relu(X)  # default slope=0.01
        U = rng.standard_normal(Xv.shape)
        Y.backward(U)

        local = np.where(Xv > 0.0, 1.0, 0.01)
        grad_expected = U * local
        np.testing.assert_allclose(X.grad, grad_expected, rtol=1e-6, atol=1e-8)

    def test_backward_matches_formula_custom_slope(self):
        rng = _rng(11)
        slope = 0.35
        Xv = rng.standard_normal((2, 3, 4))
        X = Tensor(Xv, requires_grad=True)
        Y = act.leaky_relu(X, negative_slope=slope)
        U = rng.standard_normal((2, 3, 4))
        Y.backward(U)

        local = np.where(Xv > 0.0, 1.0, slope)
        grad_expected = U * local
        np.testing.assert_allclose(X.grad, grad_expected, rtol=1e-6, atol=1e-8)

    def test_zero_input_uses_negative_slope_in_backward(self):
        slope = 0.17
        X = Tensor(np.array([0.0], dtype=np.float64), requires_grad=True)
        Y = act.leaky_relu(X, negative_slope=slope)
        Y.backward(np.array([1.0], dtype=np.float64))
        np.testing.assert_allclose(X.grad, np.array([slope]), rtol=1e-7, atol=1e-9)

    def test_backward_matches_finite_difference_gradient(self):
        rng = _rng(12)
        slope = 0.15
        Xv = rng.standard_normal((2, 3)).astype(np.float64)
        U = rng.standard_normal((2, 3)).astype(np.float64)

        X = Tensor(Xv.copy(), requires_grad=True)
        Y = act.leaky_relu(X, negative_slope=slope)
        Y.backward(U)
        grad_auto = X.grad.copy()

        eps = 1e-6
        grad_num = np.zeros_like(Xv)

        def scalar_obj(arr: np.ndarray) -> float:
            out = np.where(arr > 0.0, arr, slope * arr)
            return float(np.sum(out * U))

        for idx in np.ndindex(Xv.shape):
            x_plus = Xv.copy()
            x_minus = Xv.copy()
            x_plus[idx] += eps
            x_minus[idx] -= eps
            grad_num[idx] = (scalar_obj(x_plus) - scalar_obj(x_minus)) / (2.0 * eps)

        np.testing.assert_allclose(grad_auto, grad_num, rtol=1e-5, atol=1e-6)


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
