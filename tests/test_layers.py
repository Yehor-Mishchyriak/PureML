# layers
import unittest as ut
import numpy as np

# Public API
from pureml.machinery import Tensor
from pureml.layers import Affine, Dropout, BatchNorm1d, Embedding
from pureml.general_math import mean

def _rng(seed=0):
    return np.random.default_rng(seed)

def _decode_method_buf(buf):
    # Helper to robustly decode method buffer across dtype variations
    if isinstance(buf, np.ndarray):
        v = buf.item()
    else:
        v = buf
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "ignore")
    return str(v)


# --------------------------- Affine ---------------------------
class TestAffine(ut.TestCase):
    def test_forward_shapes_batch_and_single(self):
        B, n, m = 7, 5, 3
        layer = Affine(n, m)
        Xb = Tensor(_rng(0).standard_normal((B, n)))
        Yb = layer(Xb)
        self.assertEqual(Yb.data.shape, (B, m))

        # Single example (1D); layer should accept and return 1D output
        x = Tensor(_rng(1).standard_normal(n))
        y = layer(x)
        self.assertEqual(y.data.shape, (m,))

    def test_backward_grads_match_formulas(self):
        B, n, m = 8, 4, 6
        rng = _rng(2)
        layer = Affine(n, m)
        X = Tensor(rng.standard_normal((B, n)), requires_grad=True)
        Y = layer(X)                     # (B,m)

        # Backward with upstream ones to keep formulas simple
        U = np.ones_like(Y.data)
        Y.backward(U)

        # Expected grads:
        W = layer.W.data                 # (n,m)
        dX_expected = U @ W.T            # (B,n)
        dW_expected = X.data.T @ U       # (n,m)
        db_expected = U.sum(axis=0)      # (m,)

        np.testing.assert_allclose(X.grad, dX_expected, rtol=1e-6, atol=1e-8)
        self.assertIsNotNone(layer.W.grad)
        self.assertIsNotNone(layer.b.grad)
        np.testing.assert_allclose(layer.W.grad, dW_expected, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(layer.b.grad, db_expected, rtol=1e-6, atol=1e-8)

    def test_bias_broadcasting(self):
        B, n, m = 5, 3, 4
        layer = Affine(n, m)
        X = Tensor(np.ones((B, n)), requires_grad=True)
        Y = layer(X)
        Y.backward(np.ones_like(Y.data))
        # db should be all B's (sum of ones over batch)
        np.testing.assert_allclose(layer.b.grad, np.full((m,), B, dtype=float), rtol=1e-6, atol=1e-8)

    def test_seeded_reproducibility_and_named_buffers(self):
        n, m = 6, 5
        a1 = Affine(n, m, seed=1337)
        a2 = Affine(n, m, seed=1337)
        np.testing.assert_allclose(a1.W.data, a2.W.data, rtol=0, atol=0)
        np.testing.assert_allclose(a1.b.data, a2.b.data, rtol=0, atol=0)

        bufs = a1.named_buffers()
        self.assertIn("seed", bufs)
        self.assertIn("method", bufs)
        self.assertEqual(int(bufs["seed"].item()), 1337)
        self.assertEqual(_decode_method_buf(bufs["method"]), a1.method)

    def test_apply_state_roundtrip_and_transpose(self):
        n, m = 4, 3
        a = Affine(n, m, seed=0)

        W_nm = _rng(1).standard_normal((n, m))
        b_m  = _rng(2).standard_normal((m,))
        a.apply_state(tunable=(W_nm, b_m))
        np.testing.assert_allclose(a.W.data, W_nm, rtol=0, atol=0)
        np.testing.assert_allclose(a.b.data, b_m, rtol=0, atol=0)

        # Provide transposed W (m, n) — layer must transpose to (n, m)
        W_mn = _rng(3).standard_normal((m, n))
        a.apply_state(tunable=(W_mn, b_m))
        np.testing.assert_allclose(a.W.data, W_mn.T, rtol=0, atol=0)

        # Shape validation
        with self.assertRaises(ValueError):
            a.apply_state(tunable=(np.zeros((n+1, m)), b_m))
        with self.assertRaises(ValueError):
            a.apply_state(tunable=(np.zeros((n, m)), np.zeros((m+1,))))
        with self.assertRaises(ValueError):
            a.apply_state(tunable=(np.zeros((n, m)),))  # must be 2 arrays

    def test_apply_state_buffers_update_meta(self):
        n, m = 3, 2
        a = Affine(n, m, seed=1)
        a.apply_state(buffers={"seed": np.asarray(999, dtype=np.int64),
                               "method": np.array(b"custom-init", dtype=np.bytes_)})
        self.assertEqual(a.seed, 999)
        self.assertEqual(a.method, "custom-init")
        bufs = a.named_buffers()
        self.assertEqual(int(bufs["seed"].item()), 999)
        self.assertEqual(_decode_method_buf(bufs["method"]), "custom-init")

    def test_input_dim_validation_errors(self):
        n, m = 5, 4
        a = Affine(n, m)
        with self.assertRaises(ValueError):
            _ = a(Tensor(np.zeros((2, n+1))))  # wrong last dim
        with self.assertRaises(ValueError):
            _ = a(Tensor(np.zeros(n+1)))       # wrong 1D length

    def test_biasless_forward_backward_and_params(self):
        B, n, m = 6, 4, 5
        rng = _rng(0)
        a = Affine(n, m, bias=False, seed=123)

        X = Tensor(rng.standard_normal((B, n)), requires_grad=True)
        Y = a(X)  # should compute X @ W (no + b)
        self.assertEqual(Y.data.shape, (B, m))

        # backprop a simple upstream to check formulas
        U = np.ones_like(Y.data)
        Y.backward(U)

        dX_expected = U @ a.W.data.T          # (B,n)
        dW_expected = X.data.T @ U            # (n,m)

        np.testing.assert_allclose(X.grad, dX_expected, rtol=1e-6, atol=1e-8)
        self.assertIsNotNone(a.W.grad)
        np.testing.assert_allclose(a.W.grad, dW_expected, rtol=1e-6, atol=1e-8)

        # b exists but is frozen: no grad should be accumulated
        self.assertFalse(a.b.requires_grad)
        self.assertIsNone(a.b.grad)

        # parameters tuple must only contain W when bias=False
        self.assertEqual(len(a.parameters), 1)
        self.assertIs(a.parameters[0], a.W)

        # buffer should advertise use_bias=0
        self.assertIn("use_bias", a.named_buffers())
        self.assertEqual(int(a.named_buffers()["use_bias"].item()), 0)

    def test_apply_state_use_bias_toggle_and_tunable_counts(self):
        n, m = 3, 2
        a = Affine(n, m, seed=0)

        # turn bias off via buffers -> b zeroed & frozen; params expect only W
        a.apply_state(buffers={"use_bias": np.asarray(0, dtype=np.int8)})
        self.assertFalse(a.use_bias)
        self.assertFalse(a.b.requires_grad)
        np.testing.assert_allclose(a.b.data, np.zeros((m,)), rtol=0, atol=0)
        self.assertEqual(len(a.parameters), 1)

        # with bias disabled, only one tunable (W) should be accepted
        W_new = _rng(1).standard_normal((n, m))
        a.apply_state(tunable=(W_new,))
        np.testing.assert_allclose(a.W.data, W_new, rtol=0, atol=0)

        # passing (W, b) when bias is disabled must raise
        with self.assertRaises(ValueError):
            a.apply_state(tunable=(W_new, np.zeros((m,))))

        # turn bias back on; same Tensor retained but now trainable
        a.apply_state(buffers={"use_bias": np.asarray(1, dtype=np.int8)})
        self.assertTrue(a.use_bias)
        self.assertTrue(a.b.requires_grad)
        self.assertEqual(len(a.parameters), 2)

        # now both W and b must be provided
        b_new = _rng(2).standard_normal((m,))
        a.apply_state(tunable=(W_new, b_new))
        np.testing.assert_allclose(a.W.data, W_new, rtol=0, atol=0)
        np.testing.assert_allclose(a.b.data, b_new, rtol=0, atol=0)

    def test_zero_bias_equivalence_with_biasless_layer(self):
        n, m = 5, 3
        rng = _rng(4)
        # same seed so W is the same; make the biased layer have b == 0
        a_bias = Affine(n, m, seed=7)
        a_bias.b.data[...] = 0.0
        a_nobias = Affine(n, m, bias=False, seed=7)

        X = Tensor(rng.standard_normal((10, n)))
        Y1 = a_bias(X)
        Y2 = a_nobias(X)
        np.testing.assert_allclose(Y1.data, Y2.data, rtol=0, atol=0)

# --------------------------- Dropout ---------------------------
class TestDropout(ut.TestCase):
    def test_eval_identity(self):
        X = Tensor(_rng(0).standard_normal((4, 6)))
        d = Dropout(p=0.75, seed=123).eval()  # eval => identity
        Y = d(X)
        np.testing.assert_allclose(Y.data, X.data, rtol=0, atol=0)

    def test_train_p0_identity(self):
        X = Tensor(_rng(1).standard_normal((5, 7)))
        d = Dropout(p=0.0, seed=42).train()   # no drop
        Y = d(X)
        np.testing.assert_allclose(Y.data, X.data, rtol=0, atol=0)

    def test_seeded_determinism(self):
        X = Tensor(np.ones((100, 50)))  # large enough to exercise mask
        d1 = Dropout(p=0.6, seed=2024).train()
        d2 = Dropout(p=0.6, seed=2024).train()
        Y1 = d1(X)
        Y2 = d2(X)
        np.testing.assert_allclose(Y1.data, Y2.data, rtol=0, atol=0)

    def test_apply_state_buffers_update(self):
        X = Tensor(_rng(0).standard_normal((64, 32)))

        # Reference instance
        ref = Dropout(p=0.3, seed=7).train()
        Y_ref = ref(X)

        # Another instance with different config, then restore via apply_state
        d = Dropout(p=0.8, seed=999, training=False)  # start different on purpose
        d.apply_state(buffers={
            "p": np.asarray(0.3, dtype=np.float64),
            "seed": np.asarray(7, dtype=np.int64),
            "training": np.asarray(1, dtype=np.int8),
        })
        Y_new = d(X)

        np.testing.assert_allclose(Y_ref.data, Y_new.data, rtol=0, atol=0)

        # Flip to eval via buffers => should become identity
        d.apply_state(buffers={"training": np.asarray(0, dtype=np.int8)})
        Y_eval = d(X)
        np.testing.assert_allclose(Y_eval.data, X.data, rtol=0, atol=0)


# --------------------------- BatchNorm1d ---------------------------
class TestBatchNorm1d(ut.TestCase):
    def test_running_stats_update_in_train(self):
        B, F = 16, 5
        rng = _rng(10)
        bn = BatchNorm1d(F, momentum=0.2).train()

        # Capture initial stats if present; otherwise create placeholders
        rm0 = getattr(bn, "running_mean", None)
        rv0 = getattr(bn, "running_variance", None)
        if rm0 is not None: rm0 = rm0.data.copy()
        if rv0 is not None: rv0 = rv0.data.copy()

        X = Tensor(rng.standard_normal((B, F)))
        _ = bn(X)   # one training forward should nudge running stats

        # Running stats should be finite and (likely) changed
        self.assertTrue(hasattr(bn, "running_mean"))
        self.assertTrue(hasattr(bn, "running_variance"))
        self.assertTrue(np.all(np.isfinite(bn.running_mean.data)))
        self.assertTrue(np.all(np.isfinite(bn.running_variance.data)))
        if rm0 is not None:
            self.assertFalse(np.allclose(bn.running_mean.data, rm0))
        if rv0 is not None:
            self.assertFalse(np.allclose(bn.running_variance.data, rv0))

    def test_eval_does_not_mutate_running_stats(self):
        B, F = 8, 3
        rng = _rng(11)
        bn = BatchNorm1d(F, momentum=0.1).train()
        _ = bn(Tensor(rng.standard_normal((B, F))))  # prime running stats

        rm_before = bn.running_mean.data.copy()
        rv_before = bn.running_variance.data.copy()

        bn.eval()
        _ = bn(Tensor(rng.standard_normal((B, F))))  # eval forward; should not change stats

        np.testing.assert_allclose(bn.running_mean.data, rm_before, rtol=0, atol=0)
        np.testing.assert_allclose(bn.running_variance.data, rv_before, rtol=0, atol=0)

    def test_backward_input_grad_shape(self):
        B, F = 10, 4
        rng = _rng(12)
        bn = BatchNorm1d(F, momentum=0.2).train()
        X = Tensor(rng.standard_normal((B, F)), requires_grad=True)
        Y = bn(X)
        L = mean((Y * Y))
        L.backward()
        # Must produce input gradients of same shape
        self.assertIsNotNone(X.grad)
        self.assertEqual(X.grad.shape, X.data.shape)

    def test_apply_state_restores_running_stats(self):
        B, F = 12, 4
        rng = _rng(13)

        # First BN to generate nontrivial running stats
        bn1 = BatchNorm1d(F, momentum=0.1).train()
        _ = bn1(Tensor(rng.standard_normal((B, F))))
        rm_saved = bn1.running_mean.data.copy()
        rv_saved = bn1.running_variance.data.copy()

        # Fresh BN with different stats
        bn2 = BatchNorm1d(F, momentum=0.1).train()
        self.assertFalse(np.allclose(bn2.running_mean.data, rm_saved))
        self.assertFalse(np.allclose(bn2.running_variance.data, rv_saved))

        # Restore via default Layer.apply_state (since BN exposes named_buffers Tensors)
        bn2.apply_state(buffers={"running_mean": rm_saved, "running_variance": rv_saved})
        np.testing.assert_allclose(bn2.running_mean.data, rm_saved, rtol=0, atol=0)
        np.testing.assert_allclose(bn2.running_variance.data, rv_saved, rtol=0, atol=0)


# --------------------------- Embedding ---------------------------
class TestEmbedding(ut.TestCase):
    def test_forward_shapes_and_values(self):
        V, D = 6, 4
        # deterministic weights to compare against numpy gather
        W_arr = np.arange(V * D, dtype=np.float64).reshape(V, D)
        W = Tensor(W_arr, requires_grad=True)
        emb = Embedding(V, D, W=W)

        idx_np = np.array([[1, 3, 2],
                           [0, 4, 5]], dtype=np.int64)
        idx = Tensor(idx_np, requires_grad=False)

        Y = emb(idx)
        self.assertEqual(Y.data.shape, (2, 3, D))
        np.testing.assert_allclose(Y.data, W_arr[idx_np], rtol=0, atol=0)

    def test_backward_accumulates_repeats_and_respects_padding(self):
        V, D = 7, 3
        pad_idx = 0
        W = Tensor(np.zeros((V, D), dtype=np.float64), requires_grad=True)
        emb = Embedding(V, D, pad_idx=pad_idx, W=W)

        idx_np = np.array([[1, 1, 3, 1, 0],
                           [2, 0, 2, 2, 0]], dtype=np.int64)
        idx = Tensor(idx_np, requires_grad=False)

        Y = emb(idx)
        Y.backward(np.ones_like(Y.data))

        counts = np.bincount(idx_np.reshape(-1), minlength=V)
        expected = np.repeat(counts[:, None], D, axis=1).astype(np.float64)
        expected[pad_idx, :] = 0.0

        np.testing.assert_allclose(emb.W.grad, expected, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(emb.W.data[pad_idx], np.zeros(D), rtol=0, atol=0)

    def test_backward_matches_manual_scatter_add_with_random_upstream(self):
        V, D = 5, 2
        W = Tensor(np.zeros((V, D), dtype=np.float64), requires_grad=True)
        emb = Embedding(V, D, W=W)

        idx_np = np.array([[4, 1, 1],
                           [3, 4, 0]], dtype=np.int64)
        idx = Tensor(idx_np, requires_grad=False)
        Y = emb(idx)

        rng = _rng(123)
        upstream = rng.standard_normal(Y.data.shape)
        Y.backward(upstream)

        # manual scatter-add
        I = idx_np.reshape(-1)
        G = upstream.reshape(-1, D)
        dW_manual = np.zeros((V, D), dtype=np.float64)
        for i, g in zip(I, G):
            dW_manual[i] += g

        np.testing.assert_allclose(emb.W.grad, dW_manual, rtol=1e-6, atol=1e-8)

    def test_out_of_range_indices_raise(self):
        V, D = 4, 3
        emb = Embedding(V, D)
        idx = Tensor(np.array([[0, 1, 4]], dtype=np.int64), requires_grad=False)  # 4 is OOR
        with self.assertRaises(IndexError):
            _ = emb(idx)

    def test_float_indices_are_cast_to_int(self):
        V, D = 6, 3
        W_arr = np.arange(V * D, dtype=np.float64).reshape(V, D)
        W = Tensor(W_arr, requires_grad=True)
        emb = Embedding(V, D, W=W)

        idx_int = np.array([[1, 2],
                            [3, 0]], dtype=np.int64)
        idx_float = idx_int.astype(np.float64)

        Y1 = emb(Tensor(idx_int, requires_grad=False))
        Y2 = emb(Tensor(idx_float, requires_grad=False))
        np.testing.assert_allclose(Y1.data, Y2.data, rtol=0, atol=0)

    def test_buffers_roundtrip_padding_seed_method(self):
        V, D = 5, 4
        emb = Embedding(V, D, pad_idx=2, seed=42)
        bufs = emb.named_buffers()
        self.assertIn("padding_idx", bufs)
        self.assertIn("seed", bufs)
        self.assertIn("method", bufs)
        self.assertEqual(int(bufs["padding_idx"].item()), 2)
        self.assertEqual(int(bufs["seed"].item()), 42)
        self.assertEqual(_decode_method_buf(bufs["method"]), emb.method)

        # restore to a different padding index and seed/method
        emb.apply_state(buffers={
            "padding_idx": np.asarray(3, dtype=np.int64),
            "seed": np.asarray(777, dtype=np.int64),
            "method": np.array(b"alt-init", dtype=np.bytes_),
        })
        self.assertEqual(emb.padding_idx, 3)
        self.assertEqual(emb.seed, 777)
        self.assertEqual(emb.method, "alt-init")

    def test_preinitialized_W_shape_validation(self):
        V, D = 4, 3
        badW = Tensor(np.zeros((D, V), dtype=np.float64), requires_grad=True)
        with self.assertRaises(ValueError):
            _ = Embedding(V, D, W=badW)

    def test_seeding_repro_initialization_and_zero_padding_row(self):
        V, D = 8, 6
        pad = 5
        e1 = Embedding(V, D, pad_idx=pad, seed=99)
        e2 = Embedding(V, D, pad_idx=pad, seed=99)
        np.testing.assert_allclose(e1.W.data, e2.W.data, rtol=0, atol=0)
        # padding row must be all zeros
        np.testing.assert_allclose(e1.W.data[pad], np.zeros(D), rtol=0, atol=0)
        np.testing.assert_allclose(e2.W.data[pad], np.zeros(D), rtol=0, atol=0)

    def test_apply_state_weight_and_buffer_updates(self):
        V, D = 7, 5
        emb = Embedding(V, D, pad_idx=1, seed=0)
        W_new = _rng(0).standard_normal((V, D))
        emb.apply_state(tunable=(W_new,))
        np.testing.assert_allclose(emb.W.data, W_new, rtol=0, atol=0)

        # Method/seed/padding via buffers
        emb.apply_state(buffers={
            "method": np.array(b"custom-emb-init", dtype=np.bytes_),
            "seed": np.asarray(1234, dtype=np.int64),
            "padding_idx": np.asarray(3, dtype=np.int64)
        })
        self.assertEqual(emb.method, "custom-emb-init")
        self.assertEqual(emb.seed, 1234)
        self.assertEqual(emb.padding_idx, 3)

        # Tunable validation
        with self.assertRaises(ValueError):
            emb.apply_state(tunable=(np.zeros((D, V)),))  # wrong shape
        with self.assertRaises(ValueError):
            emb.apply_state(tunable=(np.zeros((V, D)), np.zeros((V, D))))  # too many arrays


if __name__ == "__main__":
    ut.main(verbosity=2)
