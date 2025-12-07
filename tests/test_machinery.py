# machinery
import unittest as ut
import numpy as np
from pureml.general_math import sum as t_sum
import pureml as pm
import pureml.machinery as mach

def _rng(seed=0):
    return np.random.default_rng(seed)

class TestTensorBasics(ut.TestCase):
    def test_init_and_shape(self):
        x = pm.Tensor(np.arange(6, dtype=np.float64).reshape(2, 3), requires_grad=True)
        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad)
        self.assertEqual(x.shape, (2, 3))
        np.testing.assert_array_equal(x.data, np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float64))

    def test_requires_grad_propagation_and_no_grad(self):
        a = pm.Tensor(np.ones((2, 2)), requires_grad=True)
        b = pm.Tensor(np.ones((2, 2)), requires_grad=False)
        y = a + b
        self.assertTrue(y.requires_grad)

        with pm.no_grad():
            z = a + b
            self.assertFalse(pm.is_grad_enabled())
            self.assertFalse(z.requires_grad)
        # back to normal
        self.assertTrue(pm.is_grad_enabled())

class TestElementwiseAutograd(ut.TestCase):
    def test_square_backward(self):
        rng = _rng(1)
        x = pm.Tensor(rng.standard_normal((4, 5)), requires_grad=True)
        y = x * x                # y has same shape; backward seeds ones_like
        y.backward()
        np.testing.assert_allclose(x.grad, 2.0 * x.data, rtol=1e-6, atol=1e-8)

    def test_custom_upstream_grad(self):
        rng = _rng(2)
        x = pm.Tensor(np.abs(rng.standard_normal((3, 4))) + 0.5, requires_grad=True)
        y = x ** pm.Tensor(np.ones_like(x.data))  # effectively identity but keeps autograd path
        upstream = np.full_like(y.data, 0.5)
        y.backward(upstream)
        # dy/dx = 1 → dL/dx = upstream
        np.testing.assert_allclose(x.grad, upstream, rtol=1e-6, atol=1e-8)

    def test_truediv_grad(self):
        rng = _rng(3)
        a = pm.Tensor(rng.uniform(0.5, 1.5, (3, 3)), requires_grad=True)
        b = pm.Tensor(rng.uniform(0.5, 1.5, (3, 3)), requires_grad=True)
        z = a / b
        z.backward()  # upstream = 1
        # dz/da = 1/b ; dz/db = -a/b^2
        np.testing.assert_allclose(a.grad, 1.0 / b.data, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(b.grad, -(a.data / (b.data ** 2)), rtol=1e-6, atol=1e-8)

    def test_fanout_grad_accumulates(self):
        x = pm.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y1 = x * pm.Tensor(np.array([2.0, 2.0, 2.0]))  # dy1/dx = 2
        y2 = x * pm.Tensor(np.array([3.0, 3.0, 3.0]))  # dy2/dx = 3
        z  = y1 + y2                                   # dz/dx = 5
        z.backward()                                   # upstream 1
        np.testing.assert_allclose(x.grad, np.array([5.0, 5.0, 5.0]), rtol=1e-6, atol=1e-8)

class TestBroadcasting(ut.TestCase):
    def test_add_broadcast_grad(self):
        x = pm.Tensor(np.ones((2, 3)), requires_grad=True)
        y = pm.Tensor(np.array([[10.0, 20.0, 30.0]]), requires_grad=True)  # (1,3)
        z = x + y                      # (2,3)
        z.backward()                   # upstream ones
        # dx = 1
        np.testing.assert_allclose(x.grad, np.ones_like(x.data), rtol=1e-6, atol=1e-8)
        # dy = sum over axis=0 (because y broadcast on axis 0): 2 rows → 2
        np.testing.assert_allclose(y.grad, np.array([[2.0, 2.0, 2.0]]), rtol=1e-6, atol=1e-8)

    def test_mul_broadcast_grad(self):
        x = pm.Tensor(np.array([[1.0], [2.0], [3.0]]), requires_grad=True)  # (3,1)
        y = pm.Tensor(np.array([[4.0, 5.0, 6.0, 7.0]]), requires_grad=True) # (1,4)
        z = x * y                         # (3,4)
        z.backward()                      # upstream ones
        # dx = sum over broadcasted axis=1: sum(y) = 4+5+6+7 = 22, repeated for 3 rows
        np.testing.assert_allclose(x.grad, np.array([[22.0], [22.0], [22.0]]), rtol=1e-6, atol=1e-8)
        # dy = sum over broadcasted axis=0: sum(x) = 1+2+3 = 6, repeated across 4 cols
        np.testing.assert_allclose(y.grad, np.array([[6.0, 6.0, 6.0, 6.0]]), rtol=1e-6, atol=1e-8)


class TestMatrixOps(ut.TestCase):
    def test_matmul_grad(self):
        rng = _rng(4)
        A = pm.Tensor(rng.standard_normal((2, 3)), requires_grad=True)
        B = pm.Tensor(rng.standard_normal((3, 4)), requires_grad=True)
        Y = A @ B                              # (2,4)
        Y.backward()                           # upstream ones
        up = np.ones_like(Y.data)
        dA_expected = up @ B.data.T            # (2,3)
        dB_expected = A.data.T @ up            # (3,4)
        np.testing.assert_allclose(A.grad, dA_expected, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(B.grad, dB_expected, rtol=1e-6, atol=1e-8)

    def test_batched_matmul_grad(self):
        rng = _rng(5)
        A = pm.Tensor(rng.standard_normal((5, 2, 3)), requires_grad=True)
        B = pm.Tensor(rng.standard_normal((5, 3, 4)), requires_grad=True)
        Y = A @ B                              # (5,2,4)
        Y.backward()
        up = np.ones_like(Y.data)
        dA_expected = up @ np.swapaxes(B.data, -1, -2)       # (5,2,3)
        dB_expected = np.swapaxes(A.data, -1, -2) @ up       # (5,3,4)
        np.testing.assert_allclose(A.grad, dA_expected, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(B.grad, dB_expected, rtol=1e-6, atol=1e-8)

    def test_transpose_grad(self):
        X = pm.Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        Y = X.T                                # (3,2)
        Y.backward()                           # upstream ones
        # dX = transpose(ones_like(Y))
        np.testing.assert_allclose(X.grad, np.ones_like(X.data), rtol=1e-6, atol=1e-8)

class TestReshapeFlatten(ut.TestCase):
    def test_reshape_backward(self):
        rng = _rng(6)
        X = pm.Tensor(rng.standard_normal((2, 3, 4)), requires_grad=True)
        Y = X.reshape(6, 4)
        Y.backward()                           # upstream ones → just reshaped back
        np.testing.assert_allclose(X.grad, np.ones_like(X.data), rtol=1e-6, atol=1e-8)

    def test_flatten_keep_batch(self):
        X = pm.Tensor(np.arange(2*3*4.0).reshape(2, 3, 4), requires_grad=True)
        Y = X.flatten(keep_batch=True)         # (2, 12)
        self.assertEqual(Y.data.shape, (2, 12))
        Y.backward()                           # upstream ones
        np.testing.assert_allclose(X.grad, np.ones_like(X.data), rtol=1e-6, atol=1e-8)

    def test_flatten_all(self):
        X = pm.Tensor(np.arange(24.0).reshape(2, 3, 4), requires_grad=True)
        Y = X.flatten(keep_batch=False)        # (24,)
        self.assertEqual(Y.data.shape, (24,))
        Y.backward()
        np.testing.assert_allclose(X.grad, np.ones_like(X.data), rtol=1e-6, atol=1e-8)

class TestElementaryFns(ut.TestCase):
    def test_ln_grad(self):
        x = pm.Tensor(np.array([0.5, 1.0, 2.0]), requires_grad=True)
        y = mach.ln(x)
        y.backward()                           # upstream ones
        np.testing.assert_allclose(x.grad, 1.0 / x.data, rtol=1e-6, atol=1e-8)

    def test_log2_grad(self):
        x = pm.Tensor(np.array([0.5, 1.0, 2.0]), requires_grad=True)
        y = mach.log2(x)
        y.backward()
        np.testing.assert_allclose(x.grad, 1.0 / (x.data * np.log(2.0)), rtol=1e-6, atol=1e-8)

    def test_sqrt_grad(self):
        x = pm.Tensor(np.array([0.5, 1.0, 2.0]), requires_grad=True)
        y = mach.sqrt(x)
        y.backward()
        np.testing.assert_allclose(x.grad, 0.5 / np.sqrt(x.data), rtol=1e-6, atol=1e-8)

class TestNonDiffAndGraphUtils(ut.TestCase):
    def test_argmax_is_nondiff(self):
        X = pm.Tensor(np.array([[1.0, 3.0, 2.0]]), requires_grad=True)
        idx = X.argmax(axis=-1)
        self.assertFalse(idx.requires_grad)
        np.testing.assert_array_equal(idx.data, np.array([1]))

    def test_backward_noop_when_requires_grad_false(self):
        X = pm.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=False)
        # Should not raise
        X.backward()
        self.assertIsNone(X.grad)

    def test_zero_grad_graph(self):
        x = pm.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = x * x
        y.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)
        y.zero_grad_graph()
        self.assertIsNone(x.grad)
        self.assertIsNone(y.grad)

    def test_detach_graph(self):
        x = pm.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = x + pm.Tensor(np.array([1.0, 1.0, 1.0]))
        y.detach_graph()
        self.assertIsNone(y._creator)
        # Backward from y should not flow into x
        y.backward()
        self.assertIsNone(x.grad)

    def test_fwd_context_freed_after_backward(self):
        x = pm.Tensor(np.array([0.5, 1.0, 2.0]), requires_grad=True)
        y = mach.log2(x)             # node caches context (including "out")
        node = y._creator
        self.assertIsNotNone(node)
        self.assertIsNotNone(node.fwd_ctx)     # lazily created dict
        y.backward()
        # After backward, context should be freed by engine
        self.assertIsNone(node._fwd_cache)

class TestTensorSlice(ut.TestCase):
    def test_basic_slice_forward_and_backward(self):
        X_np = np.arange(3*4.0).reshape(3, 4)
        X = pm.Tensor(X_np.copy(), requires_grad=True)

        Y = X[:, 1:3]                # shape (3, 2)
        self.assertEqual(Y.data.shape, (3, 2))
        np.testing.assert_allclose(Y.data, X_np[:, 1:3], rtol=0, atol=0)

        # upstream ones on the slice -> ones in the sliced region, zeros elsewhere
        Y.backward(np.ones_like(Y.data))

        expected = np.zeros_like(X_np)
        expected[:, 1:3] = 1.0
        np.testing.assert_allclose(X.grad, expected, rtol=1e-6, atol=1e-8)

    def test_single_element_index_backward(self):
        X_np = np.arange(2*3.0).reshape(2, 3)
        X = pm.Tensor(X_np, requires_grad=True)

        y = X[1, 2]                  # scalar
        self.assertEqual(y.data.shape, ())
        y.backward(np.array(2.0))    # custom upstream scalar

        expected = np.zeros_like(X_np)
        expected[1, 2] = 2.0
        np.testing.assert_allclose(X.grad, expected, rtol=1e-6, atol=1e-8)

    def test_bool_mask_forward_and_backward(self):
        X_np = np.array([[1., 2., 3.],
                         [4., 5., 6.]])
        mask = np.array([[ True, False,  True],
                         [False,  True, False]])
        X = pm.Tensor(X_np, requires_grad=True)

        y = X[mask]                  # 1D packed selection of 3 elements
        self.assertEqual(y.data.shape, (3,))
        np.testing.assert_allclose(y.data, X_np[mask], rtol=0, atol=0)

        y.backward(np.ones_like(y.data))

        expected = np.zeros_like(X_np)
        expected[mask] = 1.0
        np.testing.assert_allclose(X.grad, expected, rtol=1e-6, atol=1e-8)

    def test_ellipsis_and_none_dims(self):
        X_np = np.arange(2*3*4.0).reshape(2, 3, 4)
        X = pm.Tensor(X_np, requires_grad=True)

        # Insert leading axis and slice the last dim
        Y = X[None, ..., 1:]         # shape (1, 2, 3, 3)
        self.assertEqual(Y.data.shape, (1, 2, 3, 3))
        np.testing.assert_allclose(Y.data, X_np[None, :, :, 1:], rtol=0, atol=0)

        Y.backward(np.ones_like(Y.data))
        expected = np.zeros_like(X_np)
        expected[:, :, 1:] = 1.0
        np.testing.assert_allclose(X.grad, expected, rtol=1e-6, atol=1e-8)

    def test_advanced_integer_arrays_elementwise_pairs(self):
        # Index with two integer arrays of the same shape: selects element-wise pairs
        X_np = np.zeros((3, 4), dtype=np.float64)
        X = pm.Tensor(X_np, requires_grad=True)

        I = np.array([0, 1, 1])      # rows
        J = np.array([1, 2, 2])      # cols (note the repeated (1,2))
        y = X[I, J]                  # shape (3,)

        # Use distinct upstream so we can see accumulation: (0,1)->1.0 ; (1,2)->2.0 ; (1,2)->3.0
        upstream = np.array([1.0, 2.0, 3.0])
        y.backward(upstream)

        expected = np.zeros_like(X_np)
        expected[0, 1] += 1.0
        expected[1, 2] += 2.0
        expected[1, 2] += 3.0        # accumulation on repeated index!
        np.testing.assert_allclose(X.grad, expected, rtol=1e-6, atol=1e-8)

    def test_row_gather_with_repeats_accumulates(self):
        # Advanced indexing with a 1D integer array (row gather), repeated rows
        X_np = np.arange(3*5.0).reshape(3, 5)
        X = pm.Tensor(X_np, requires_grad=True)

        rows = np.array([1, 1, 2])   # repeat row 1
        Y = X[rows, :]               # shape (3, 5)

        # upstream ones -> row 1 should get +2, row 2 +1
        Y.backward(np.ones_like(Y.data))

        expected = np.zeros_like(X_np)
        expected[1, :] += 2.0
        expected[2, :] += 1.0

        np.testing.assert_allclose(X.grad, expected, rtol=1e-6, atol=1e-8)

    def test_requires_grad_propagation_through_slice(self):
        X = pm.Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        Y = X[:, :2]
        self.assertTrue(Y.requires_grad)

        Z = pm.Tensor(np.arange(6.0).reshape(2, 3), requires_grad=False)
        W = Z[:, :2]
        self.assertFalse(W.requires_grad)


class TestDetachAndExport(ut.TestCase):
    def test_detach_returns_leaf_shared_and_blocks_grad(self):
        x = pm.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = x * x                               # y gets a creator; x stays a leaf
        self.assertIsNotNone(y._creator)

        d = y.detach()                          # new leaf, shared storage
        self.assertFalse(d.requires_grad)
        self.assertIsNone(d._creator)
        self.assertTrue(np.shares_memory(d.data, y.data))

        # ops on detached tensor don't affect the old graph
        z = t_sum(d * d)                        # reduction op (detached path)
        self.assertFalse(z.requires_grad)
        z.backward()                            # no-op w.r.t. original graph
        self.assertIsNone(x.grad)
        self.assertIsNone(y.grad)

    def test_detach_inplace_stops_tracking(self):
        x = pm.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = x * x                               # build some op; y has creator, x is leaf
        self.assertIsNotNone(y._creator)
        self.assertIsNone(x._creator)

        x.detach_()                             # in-place: stop tracking on x
        self.assertFalse(x.requires_grad)
        self.assertIsNone(x._creator)
        self.assertIsNone(x.grad)

        # future ops with only this input should not require grad
        z = x + pm.Tensor(np.array([1.0, 1.0, 1.0]), requires_grad=False)
        self.assertFalse(z.requires_grad)

    def test_requires_grad_toggle_builds_graph(self):
        x = pm.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=False)
        x = x.requires_grad_(True)
        self.assertTrue(x.requires_grad)

        y = t_sum(x * x)                        # use autograd-aware reduction
        self.assertTrue(y.requires_grad)
        y.backward()

        np.testing.assert_allclose(x.grad, 2.0 * x.data, rtol=1e-6, atol=1e-8)

    def test_numpy_copy_is_isolated(self):
        x = pm.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        out = x.numpy()                         # default: copy=True, readonly=False
        self.assertFalse(np.shares_memory(out, x.data))
        out[0] = 999.0
        np.testing.assert_array_equal(x.data, np.array([1.0, 2.0, 3.0]))

    def test_numpy_view_readonly_shares_storage(self):
        x = pm.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        view = x.numpy(copy=False, readonly=True)
        self.assertTrue(np.shares_memory(view, x.data))
        self.assertFalse(view.flags.writeable)
        with self.assertRaises(ValueError):
            view[0] = 42.0                      # readonly → must raise


class TestGraphUtilitiesExtended(ut.TestCase):
    def test_detach_graph_clears_creators_requires_grad_and_frees_contexts(self):
        # Build a small chain with cached context in nodes
        x = pm.Tensor(np.array([0.5, 1.0, 2.0]), requires_grad=True)
        u = mach.sqrt(x)                          # caches "out" in context
        v = mach.log2(u)                          # also uses context
        y = u + v

        fn_u = u._creator
        fn_v = v._creator
        self.assertIsNotNone(fn_u)
        self.assertIsNotNone(fn_v)

        # contexts should exist after forward
        _ = y.data
        self.assertIsNotNone(fn_u.fwd_ctx)
        self.assertIsNotNone(fn_v.fwd_ctx)

        y.detach_graph()

        # creators cleared and requires_grad disabled upstream
        for t in (x, u, v, y):
            self.assertIsNone(t._creator)
            self.assertFalse(t.requires_grad)
            self.assertIsNone(t.grad)

        # cached forward contexts freed
        self.assertIsNone(fn_u._fwd_cache)
        self.assertIsNone(fn_v._fwd_cache)

        # Backward after detaching should be a no-op
        y.backward()
        self.assertIsNone(x.grad)

    def test_zero_grad_graph_on_fanout(self):
        x = pm.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        a = x * pm.Tensor(np.array([2.0, 2.0, 2.0]))
        b = x * pm.Tensor(np.array([3.0, 3.0, 3.0]))
        z = t_sum(a + b)
        z.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(z.grad)

        z.zero_grad_graph()
        self.assertIsNone(x.grad)
        self.assertIsNone(z.grad)


if __name__ == "__main__":
    ut.main(verbosity=2)
