# layers
import unittest as ut
import numpy as np

# Public API
from pureml.machinery import Tensor
from pureml.layers import (
    Layer, Affine, Dropout, Dropout2d, BatchNorm1d, LayerNorm1d, Embedding,
    Conv1D, Conv2D, MaxPool1D, MeanPool1D, MaxPool2D, MeanPool2D, BatchNorm2d,
    unfold1d, unfold2d, output_len_1d, output_shape_2d
)
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

def _manual_unfold1d(
        X: np.ndarray,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        pad_with: float = 0.0) -> np.ndarray:
    B, C, L = X.shape
    kL = int(kernel_size)
    sL = int(stride)
    pL = int(padding)
    dL = int(dilation)

    X_pad = np.pad(X, ((0, 0), (0, 0), (pL, pL)), mode="constant", constant_values=pad_with)
    L_out = int(np.floor((L + 2*pL - dL*(kL - 1) - 1) / sL) + 1)
    cols = np.zeros((B, C * kL, L_out), dtype=X.dtype)

    for b in range(B):
        for c in range(C):
            for t in range(kL):
                r = c * kL + t
                for p in range(L_out):
                    l_idx = p * sL + t * dL
                    cols[b, r, p] = X_pad[b, c, l_idx]
    return cols

def _manual_unfold2d(
        X: np.ndarray,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        pad_with: float = 0.0) -> np.ndarray:
    B, C, H, W = X.shape
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    X_pad = np.pad(X, ((0, 0), (0, 0), (pH, pH), (pW, pW)), mode="constant", constant_values=pad_with)
    H_out = int(np.floor((H + 2*pH - dH*(kH - 1) - 1) / sH) + 1)
    W_out = int(np.floor((W + 2*pW - dW*(kW - 1) - 1) / sW) + 1)
    cols = np.zeros((B, C * kH * kW, H_out * W_out), dtype=X.dtype)

    for b in range(B):
        for c in range(C):
            for kh in range(kH):
                for kw in range(kW):
                    r = c * (kH * kW) + kh * kW + kw
                    for oh in range(H_out):
                        for ow in range(W_out):
                            p = oh * W_out + ow
                            h_idx = oh * sH + kh * dH
                            w_idx = ow * sW + kw * dW
                            cols[b, r, p] = X_pad[b, c, h_idx, w_idx]
    return cols

def _manual_fold1d_grad(
        upstream_grad: np.ndarray,
        in_shape: tuple[int, int, int],
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int) -> np.ndarray:
    B, C, L = in_shape
    kL = int(kernel_size)
    sL = int(stride)
    pL = int(padding)
    dL = int(dilation)

    L_out = int(np.floor((L + 2*pL - dL*(kL - 1) - 1) / sL) + 1)
    gpad = np.zeros((B, C, L + 2*pL), dtype=upstream_grad.dtype)
    for b in range(B):
        for c in range(C):
            for t in range(kL):
                r = c * kL + t
                for p in range(L_out):
                    l_idx = p * sL + t * dL
                    gpad[b, c, l_idx] += upstream_grad[b, r, p]
    return gpad[:, :, pL:pL+L]

def _manual_fold2d_grad(
        upstream_grad: np.ndarray,
        in_shape: tuple[int, int, int, int],
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int]) -> np.ndarray:
    B, C, H, W = in_shape
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    H_out = int(np.floor((H + 2*pH - dH*(kH - 1) - 1) / sH) + 1)
    W_out = int(np.floor((W + 2*pW - dW*(kW - 1) - 1) / sW) + 1)
    gpad = np.zeros((B, C, H + 2*pH, W + 2*pW), dtype=upstream_grad.dtype)
    for b in range(B):
        for c in range(C):
            for kh in range(kH):
                for kw in range(kW):
                    r = c * (kH * kW) + kh * kW + kw
                    for oh in range(H_out):
                        for ow in range(W_out):
                            p = oh * W_out + ow
                            h_idx = oh * sH + kh * dH
                            w_idx = ow * sW + kw * dW
                            gpad[b, c, h_idx, w_idx] += upstream_grad[b, r, p]
    return gpad[:, :, pH:pH+H, pW:pW+W]


def _manual_conv1d(
        X: np.ndarray,
        W: np.ndarray,
        b: np.ndarray | None,
        *,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        pad_with: float = 0.0) -> np.ndarray:
    cols = _manual_unfold1d(
        X,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        pad_with=pad_with
    )  # (B, C*kL, L_out)
    Z = np.matmul(W, cols)  # (B, O, L_out)
    if b is not None:
        Z = Z + b[None, :, None]
    return Z


def _manual_conv2d(
        X: np.ndarray,
        W: np.ndarray,
        b: np.ndarray | None,
        *,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        pad_with: float = 0.0) -> np.ndarray:
    B, _, H, W_in = X.shape
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    H_out = int(np.floor((H + 2*pH - dH*(kH - 1) - 1) / sH) + 1)
    W_out = int(np.floor((W_in + 2*pW - dW*(kW - 1) - 1) / sW) + 1)

    cols = _manual_unfold2d(
        X,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        pad_with=pad_with
    )  # (B, C*kH*kW, H_out*W_out)
    Z = np.matmul(W, cols)  # (B, O, H_out*W_out)
    if b is not None:
        Z = Z + b[None, :, None]
    return Z.reshape(B, W.shape[0], H_out, W_out)


def _manual_pool1d(
        X: np.ndarray,
        *,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        pad_with: float = 0.0,
        mode: str) -> np.ndarray:
    cols = _manual_unfold1d(
        X,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        pad_with=pad_with
    )  # (B, C*kL, L_out)
    B, C, _ = X.shape
    K = int(kernel_size)
    U = cols.reshape(B, C, K, -1)
    if mode == "mean":
        return U.mean(axis=2)
    if mode == "max":
        return U.max(axis=2)
    raise ValueError(f"Unknown mode: {mode}")


def _manual_pool2d(
        X: np.ndarray,
        *,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        pad_with: float = 0.0,
        mode: str) -> np.ndarray:
    cols = _manual_unfold2d(
        X,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        pad_with=pad_with
    )  # (B, C*kH*kW, P)
    B, C, _, _ = X.shape
    kH, kW = kernel_size
    K = kH * kW
    U = cols.reshape(B, C, K, -1)
    if mode == "mean":
        return U.mean(axis=2)
    if mode == "max":
        return U.max(axis=2)
    raise ValueError(f"Unknown mode: {mode}")


def _manual_pool1d_input_grad(
        X: np.ndarray,
        upstream: np.ndarray,
        *,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        pad_with: float = 0.0,
        mode: str) -> np.ndarray:
    cols = _manual_unfold1d(
        X,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        pad_with=pad_with
    )  # (B, C*kL, P)
    B, C, _ = X.shape
    K = int(kernel_size)
    U = cols.reshape(B, C, K, -1)
    if mode == "mean":
        gU = np.broadcast_to(upstream[:, :, None, :] / K, U.shape).copy()
    elif mode == "max":
        idx = np.argmax(U, axis=2)  # (B, C, P)
        gU = np.zeros_like(U)
        np.put_along_axis(gU, idx[:, :, None, :], upstream[:, :, None, :], axis=2)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    gcols = gU.reshape(B, C * K, -1)
    return _manual_fold1d_grad(gcols, X.shape, kernel_size, stride, padding, dilation)


def _manual_pool2d_input_grad(
        X: np.ndarray,
        upstream: np.ndarray,
        *,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        pad_with: float = 0.0,
        mode: str) -> np.ndarray:
    cols = _manual_unfold2d(
        X,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        pad_with=pad_with
    )  # (B, C*kH*kW, P)
    B, C, _, _ = X.shape
    kH, kW = kernel_size
    K = kH * kW
    U = cols.reshape(B, C, K, -1)
    if mode == "mean":
        gU = np.broadcast_to(upstream[:, :, None, :] / K, U.shape).copy()
    elif mode == "max":
        idx = np.argmax(U, axis=2)  # (B, C, P)
        gU = np.zeros_like(U)
        np.put_along_axis(gU, idx[:, :, None, :], upstream[:, :, None, :], axis=2)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    gcols = gU.reshape(B, C * K, -1)
    return _manual_fold2d_grad(gcols, X.shape, kernel_size, stride, padding, dilation)

class _ToyLayer(Layer):
    def __init__(self):
        super().__init__()
        self.w = Tensor(np.ones((2,), dtype=np.float64), requires_grad=True)

    @property
    def parameters(self) -> tuple[Tensor, ...]:
        return (self.w,)

    def __call__(self, X: Tensor) -> Tensor:
        return X


class TestLayerBase(ut.TestCase):
    def test_apply_state_rejects_partial_tunable_payload(self):
        toy = _ToyLayer()
        with self.assertRaises(ValueError):
            toy.apply_state(tunable=(np.zeros((2,)), np.zeros((2,))))

    def test_apply_state_skips_none_entries(self):
        toy = _ToyLayer()
        before = toy.w.data.copy()
        toy.apply_state(tunable=(None,))
        np.testing.assert_allclose(toy.w.data, before, rtol=0, atol=0)


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

    def test_dropout_grad_returns_none_for_mask_and_scale(self):
        rng = _rng(99)
        X = rng.standard_normal((4, 3))
        mask = (rng.random((4, 3)) < 0.7).astype(np.float64)
        scale = np.asarray(1.0 / 0.7, dtype=np.float64)
        upstream = rng.standard_normal((4, 3))

        gX, gmask, gscale = Dropout._dropout_grad(upstream, X, mask, scale)
        np.testing.assert_allclose(gX, upstream * (mask * scale), rtol=1e-6, atol=1e-8)
        self.assertIsNone(gmask)
        self.assertIsNone(gscale)


class TestDropout2d(ut.TestCase):
    def test_eval_identity(self):
        X = Tensor(_rng(0).standard_normal((4, 6, 8, 8)))
        d = Dropout2d(p=0.75, seed=123).eval()  # eval => identity
        Y = d(X)
        np.testing.assert_allclose(Y.data, X.data, rtol=0, atol=0)

    def test_train_p0_identity(self):
        X = Tensor(_rng(1).standard_normal((5, 7, 4, 4)))
        d = Dropout2d(p=0.0, seed=42).train()  # no drop
        Y = d(X)
        np.testing.assert_allclose(Y.data, X.data, rtol=0, atol=0)

    def test_seeded_determinism(self):
        X = Tensor(np.ones((16, 8, 5, 5)))  # large enough to exercise mask
        d1 = Dropout2d(p=0.6, seed=2024).train()
        d2 = Dropout2d(p=0.6, seed=2024).train()
        Y1 = d1(X)
        Y2 = d2(X)
        np.testing.assert_allclose(Y1.data, Y2.data, rtol=0, atol=0)

    def test_channelwise_mask(self):
        X = Tensor(np.ones((3, 4, 6, 6), dtype=np.float64))
        d = Dropout2d(p=0.5, seed=11).train()
        Y = d(X).data
        allowed = {0.0, 2.0}
        for b in range(Y.shape[0]):
            for c in range(Y.shape[1]):
                vals = np.unique(Y[b, c])
                self.assertEqual(len(vals), 1)  # whole map is dropped/kept together
                self.assertIn(float(vals[0]), allowed)

    def test_invalid_rank_raises(self):
        d = Dropout2d(p=0.5, seed=1).train()
        with self.assertRaises(ValueError):
            _ = d(Tensor(np.zeros((3, 5))))

    def test_dropout_grad_returns_none_for_mask_and_scale(self):
        rng = _rng(99)
        X = rng.standard_normal((2, 3, 4, 4))
        mask = (rng.random((2, 3, 1, 1)) < 0.7).astype(np.float64)
        scale = np.asarray(1.0 / 0.7, dtype=np.float64)
        upstream = rng.standard_normal((2, 3, 4, 4))

        gX, gmask, gscale = Dropout2d._dropout_grad(upstream, X, mask, scale)
        np.testing.assert_allclose(gX, upstream * (mask * scale), rtol=1e-6, atol=1e-8)
        self.assertIsNone(gmask)
        self.assertIsNone(gscale)


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

        # Restore running-stat buffers through BN's state loader
        bn2.apply_state(buffers={"running_mean": rm_saved, "running_variance": rv_saved})
        np.testing.assert_allclose(bn2.running_mean.data, rm_saved, rtol=0, atol=0)
        np.testing.assert_allclose(bn2.running_variance.data, rv_saved, rtol=0, atol=0)

    def test_named_buffers_include_bn_metadata(self):
        F = 5
        bn = BatchNorm1d(F, eps=1e-4, momentum=0.25).eval()
        bufs = bn.named_buffers()
        self.assertIn("running_mean", bufs)
        self.assertIn("running_variance", bufs)
        self.assertIn("eps", bufs)
        self.assertIn("momentum", bufs)
        self.assertIn("training", bufs)
        self.assertIn("num_features", bufs)
        self.assertIsInstance(bufs["eps"], Tensor)
        self.assertAlmostEqual(float(bufs["eps"].data), 1e-4, places=12)
        self.assertAlmostEqual(float(np.asarray(bufs["momentum"]).item()), 0.25, places=12)
        self.assertEqual(int(np.asarray(bufs["training"]).item()), 0)
        self.assertEqual(int(np.asarray(bufs["num_features"]).item()), F)

    def test_apply_state_restores_metadata_and_validates_num_features(self):
        F = 4
        bn = BatchNorm1d(F, eps=1e-5, momentum=0.1).train()
        bn.apply_state(buffers={
            "eps": np.asarray(1e-3, dtype=np.float64),
            "momentum": np.asarray(0.33, dtype=np.float64),
            "training": np.asarray(0, dtype=np.int8),
            "num_features": np.asarray(F, dtype=np.int64),
        })
        self.assertAlmostEqual(float(bn.eps.data), 1e-3, places=12)
        self.assertAlmostEqual(float(bn.momentum), 0.33, places=12)
        self.assertFalse(bn.training)

        with self.assertRaises(ValueError):
            bn.apply_state(buffers={"num_features": np.asarray(F + 1, dtype=np.int64)})


class TestBatchNorm2d(ut.TestCase):
    def test_forward_matches_manual_formula_in_train(self):
        B, C, H, W = 3, 4, 5, 6
        rng = _rng(514)
        X_np = rng.standard_normal((B, C, H, W))
        X = Tensor(X_np, requires_grad=False)
        bn = BatchNorm2d(C, eps=1e-5, momentum=0.2).train()

        Y = bn(X).data
        X_flat = np.transpose(X_np, (0, 2, 3, 1)).reshape(B * H * W, C)
        mu = X_flat.mean(axis=0)
        var = ((X_flat - mu) ** 2).mean(axis=0)
        Y_flat = (X_flat - mu) / np.sqrt(var + float(bn.eps.data))
        Y_flat = Y_flat * bn.gamma.data + bn.beta.data
        expected = np.transpose(Y_flat.reshape(B, H, W, C), (0, 3, 1, 2))

        self.assertEqual(Y.shape, (B, C, H, W))
        np.testing.assert_allclose(Y, expected, rtol=1e-6, atol=1e-8)

    def test_running_stats_update_and_eval_freeze(self):
        B, C, H, W = 2, 3, 4, 4
        rng = _rng(515)
        bn = BatchNorm2d(C, momentum=0.1).train()

        _ = bn(Tensor(rng.standard_normal((B, C, H, W))))
        rm_before = bn.running_mean.data.copy()
        rv_before = bn.running_variance.data.copy()
        self.assertTrue(np.all(np.isfinite(rm_before)))
        self.assertTrue(np.all(np.isfinite(rv_before)))

        bn.eval()
        _ = bn(Tensor(rng.standard_normal((B, C, H, W))))
        np.testing.assert_allclose(bn.running_mean.data, rm_before, rtol=0, atol=0)
        np.testing.assert_allclose(bn.running_variance.data, rv_before, rtol=0, atol=0)

    def test_backward_shape_and_apply_state(self):
        B, C, H, W = 2, 5, 3, 3
        rng = _rng(516)
        bn = BatchNorm2d(C).train()
        X = Tensor(rng.standard_normal((B, C, H, W)), requires_grad=True)
        Y = bn(X)
        L = mean(Y * Y)
        L.backward()

        self.assertIsNotNone(X.grad)
        self.assertEqual(X.grad.shape, X.shape)
        self.assertIsNotNone(bn.gamma.grad)
        self.assertIsNotNone(bn.beta.grad)

        gamma_new = rng.standard_normal((C,))
        beta_new = rng.standard_normal((C,))
        bn.apply_state(
            tunable=(gamma_new, beta_new),
            buffers={
                "momentum": np.asarray(0.33, dtype=np.float64),
                "training": np.asarray(0, dtype=np.int8),
                "num_features": np.asarray(C, dtype=np.int64),
            },
        )
        np.testing.assert_allclose(bn.gamma.data, gamma_new, rtol=0, atol=0)
        np.testing.assert_allclose(bn.beta.data, beta_new, rtol=0, atol=0)
        self.assertAlmostEqual(float(bn.momentum), 0.33, places=12)
        self.assertFalse(bn.training)

        with self.assertRaises(ValueError):
            bn.apply_state(buffers={"num_features": np.asarray(C + 1, dtype=np.int64)})

    def test_input_validation_and_buffer_contract(self):
        bn = BatchNorm2d(3)
        self.assertEqual(len(bn.parameters), 2)
        bufs = bn.named_buffers()
        self.assertIn("running_mean", bufs)
        self.assertIn("running_variance", bufs)
        self.assertIn("eps", bufs)
        self.assertIn("momentum", bufs)
        self.assertIn("training", bufs)
        self.assertIn("num_features", bufs)

        with self.assertRaises(ValueError):
            _ = bn(Tensor(np.zeros((2, 3, 4))))      # ndim mismatch
        with self.assertRaises(ValueError):
            _ = bn(Tensor(np.zeros((2, 2, 4, 4))))   # channel mismatch


# --------------------------- LayerNorm1d ---------------------------
class TestLayerNorm1d(ut.TestCase):
    def test_forward_2d_matches_manual_formula(self):
        B, F = 7, 5
        rng = _rng(21)
        X_np = rng.standard_normal((B, F))
        X = Tensor(X_np, requires_grad=False)

        ln = LayerNorm1d(F, eps=1e-5, bias=True)
        Y = ln(X)

        mu = X_np.mean(axis=-1, keepdims=True)
        var = ((X_np - mu) ** 2).mean(axis=-1, keepdims=True)
        expected = (X_np - mu) / np.sqrt(var + ln.eps.data)
        expected = expected * ln.gamma.data + ln.beta.data

        self.assertEqual(Y.data.shape, X_np.shape)
        np.testing.assert_allclose(Y.data, expected, rtol=1e-6, atol=1e-8)

    def test_forward_3d_normalizes_last_axis_per_sample(self):
        B, T, F = 4, 3, 6
        rng = _rng(22)
        X_np = rng.standard_normal((B, T, F))
        X = Tensor(X_np, requires_grad=False)
        ln = LayerNorm1d(F, bias=False)

        Y = ln(X).data
        self.assertEqual(Y.shape, (B, T, F))

        # bias=False and gamma initialized to ones -> pure normalized output
        mu = Y.mean(axis=-1)
        var = Y.var(axis=-1)
        np.testing.assert_allclose(mu, np.zeros((B, T)), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(var, np.ones((B, T)), rtol=5e-5, atol=5e-5)

    def test_affine_parameters_are_applied(self):
        B, F = 5, 4
        rng = _rng(23)
        X_np = rng.standard_normal((B, F))
        X = Tensor(X_np, requires_grad=False)

        gamma = Tensor(np.array([2.0, 0.5, -1.0, 3.0]), requires_grad=True)
        beta  = Tensor(np.array([0.1, -0.2, 0.3, 1.1]), requires_grad=True)
        ln = LayerNorm1d(F, gamma=gamma, beta=beta, bias=True)
        Y = ln(X).data

        mu = X_np.mean(axis=-1, keepdims=True)
        var = ((X_np - mu) ** 2).mean(axis=-1, keepdims=True)
        xhat = (X_np - mu) / np.sqrt(var + ln.eps.data)
        expected = xhat * gamma.data + beta.data
        np.testing.assert_allclose(Y, expected, rtol=1e-6, atol=1e-8)

    def test_bias_false_parameter_contract_and_forward(self):
        B, F = 6, 3
        rng = _rng(24)
        X_np = rng.standard_normal((B, F))
        X = Tensor(X_np, requires_grad=False)

        ln = LayerNorm1d(F, bias=False)
        self.assertFalse(ln.beta.requires_grad)
        self.assertEqual(len(ln.parameters), 1)
        self.assertIs(ln.parameters[0], ln.gamma)

        Y = ln(X).data
        mu = X_np.mean(axis=-1, keepdims=True)
        var = ((X_np - mu) ** 2).mean(axis=-1, keepdims=True)
        expected = (X_np - mu) / np.sqrt(var + ln.eps.data) * ln.gamma.data
        np.testing.assert_allclose(Y, expected, rtol=1e-6, atol=1e-8)

    def test_backward_shapes_and_bias_grad_behavior(self):
        B, F = 8, 5
        rng = _rng(25)
        X = Tensor(rng.standard_normal((B, F)), requires_grad=True)

        # bias=True -> gamma and beta both receive gradients
        ln_b = LayerNorm1d(F, bias=True)
        Yb = ln_b(X)
        Lb = mean(Yb * Yb)
        Lb.backward()
        self.assertIsNotNone(X.grad)
        self.assertEqual(X.grad.shape, X.shape)
        self.assertIsNotNone(ln_b.gamma.grad)
        self.assertIsNotNone(ln_b.beta.grad)

        # bias=False -> beta should stay grad-free
        X2 = Tensor(rng.standard_normal((B, F)), requires_grad=True)
        ln_nb = LayerNorm1d(F, bias=False)
        Yn = ln_nb(X2)
        Ln = mean(Yn * Yn)
        Ln.backward()
        self.assertIsNotNone(ln_nb.gamma.grad)
        self.assertIsNone(ln_nb.beta.grad)

    def test_input_validation(self):
        F = 4
        ln = LayerNorm1d(F)

        with self.assertRaises(TypeError):
            _ = ln(np.zeros((2, F)))  # non-Tensor

        with self.assertRaises(ValueError):
            _ = ln(Tensor(np.array(1.0)))  # ndim == 0

        with self.assertRaises(ValueError):
            _ = ln(Tensor(np.zeros((3, F + 1))))  # wrong last dim

    def test_ctor_validation(self):
        with self.assertRaises(ValueError):
            _ = LayerNorm1d(0)
        with self.assertRaises(ValueError):
            _ = LayerNorm1d(4, eps=0.0)
        with self.assertRaises(ValueError):
            _ = LayerNorm1d(4, eps=-1e-5)

        with self.assertRaises(ValueError):
            _ = LayerNorm1d(4, gamma=Tensor(np.zeros((5,))))
        with self.assertRaises(ValueError):
            _ = LayerNorm1d(4, beta=Tensor(np.zeros((5,))))
        with self.assertRaises(ValueError):
            _ = LayerNorm1d(4, bias=False, beta=Tensor(np.zeros((4,))))

    def test_mode_toggle_does_not_change_outputs(self):
        B, F = 6, 4
        rng = _rng(26)
        X = Tensor(rng.standard_normal((B, F)), requires_grad=False)
        ln = LayerNorm1d(F).train()
        Y_train = ln(X).data.copy()
        ln.eval()
        Y_eval = ln(X).data.copy()
        np.testing.assert_allclose(Y_train, Y_eval, rtol=1e-6, atol=1e-8)

    def test_named_buffers_include_eps_and_use_bias(self):
        ln = LayerNorm1d(3, bias=False, eps=1e-4)
        bufs = ln.named_buffers()
        self.assertIn("eps", bufs)
        self.assertIn("use_bias", bufs)
        self.assertIsInstance(bufs["eps"], Tensor)
        self.assertFalse(bufs["use_bias"])
        self.assertAlmostEqual(float(bufs["eps"].data), 1e-4, places=12)

    def test_apply_state_restores_buffers_and_respects_bias_toggle(self):
        F = 4
        ln = LayerNorm1d(F, bias=True, eps=1e-5)

        # turn bias off and restore eps through buffers
        ln.apply_state(buffers={
            "use_bias": np.asarray(0, dtype=np.int8),
            "eps": np.asarray(1e-3, dtype=np.float64),
        })
        self.assertFalse(ln.use_bias)
        self.assertFalse(ln.beta.requires_grad)
        self.assertEqual(len(ln.parameters), 1)
        self.assertAlmostEqual(float(ln.eps.data), 1e-3, places=12)

        # bias=False -> only gamma may be supplied
        gamma_new = _rng(120).standard_normal((F,))
        ln.apply_state(tunable=(gamma_new,))
        np.testing.assert_allclose(ln.gamma.data, gamma_new, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            ln.apply_state(tunable=(gamma_new, np.zeros((F,))))

        # turn bias on and load both gamma/beta
        beta_new = _rng(121).standard_normal((F,))
        ln.apply_state(buffers={"use_bias": np.asarray(1, dtype=np.int8)})
        self.assertTrue(ln.use_bias)
        self.assertTrue(ln.beta.requires_grad)
        self.assertEqual(len(ln.parameters), 2)
        ln.apply_state(tunable=(gamma_new, beta_new))
        np.testing.assert_allclose(ln.beta.data, beta_new, rtol=0, atol=0)


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


class TestConv1D(ut.TestCase):
    def test_forward_matches_manual_reference_with_bias(self):
        B, C, L = 2, 3, 9
        O = 4
        kL, sL, pL, dL = 3, 2, 1, 1
        rng = _rng(300)

        conv = Conv1D(C, O, kL, stride=sL, padding=pL, dilation=dL, use_bias=True, seed=7)
        X_np = rng.standard_normal((B, C, L))
        X = Tensor(X_np, requires_grad=False)

        Y = conv(X)
        Y_ref = _manual_conv1d(
            X_np, conv.W.data, conv.b.data,
            kernel_size=kL, stride=sL, padding=pL, dilation=dL, pad_with=conv.pad_with
        )
        self.assertEqual(Y.data.shape, Y_ref.shape)
        np.testing.assert_allclose(Y.data, Y_ref, rtol=1e-6, atol=1e-8)

    def test_forward_matches_manual_reference_without_bias(self):
        B, C, L = 2, 2, 8
        O = 3
        kL = 3
        rng = _rng(301)

        conv = Conv1D(C, O, kL, use_bias=False, seed=11)
        X_np = rng.standard_normal((B, C, L))
        X = Tensor(X_np, requires_grad=False)
        Y = conv(X)

        Y_ref = _manual_conv1d(
            X_np, conv.W.data, None,
            kernel_size=kL, stride=1, padding=0, dilation=1, pad_with=conv.pad_with
        )
        self.assertEqual(len(conv.parameters), 1)
        self.assertFalse(conv.b.requires_grad)
        np.testing.assert_allclose(Y.data, Y_ref, rtol=1e-6, atol=1e-8)

    def test_input_validation_errors(self):
        conv = Conv1D(2, 3, 3)
        with self.assertRaises(ValueError):
            _ = conv(Tensor(np.zeros((2, 2, 5, 1))))  # wrong ndim
        with self.assertRaises(ValueError):
            _ = conv(Tensor(np.zeros((2, 1, 5))))     # wrong channels

    def test_seeded_reproducibility_and_named_buffers(self):
        c1 = Conv1D(3, 4, 5, stride=2, padding=1, dilation=2, seed=2026)
        c2 = Conv1D(3, 4, 5, stride=2, padding=1, dilation=2, seed=2026)
        np.testing.assert_allclose(c1.W.data, c2.W.data, rtol=0, atol=0)
        np.testing.assert_allclose(c1.b.data, c2.b.data, rtol=0, atol=0)

        bufs = c1.named_buffers()
        self.assertEqual(int(bufs["seed"].item()), 2026)
        self.assertEqual(_decode_method_buf(bufs["method"]), c1.method)
        self.assertEqual(_decode_method_buf(bufs["nonlinearity"]), c1.nonlinearity)
        self.assertEqual(int(bufs["kernel_size"].item()), 5)
        self.assertEqual(int(bufs["stride"].item()), 2)
        self.assertEqual(int(bufs["padding"].item()), 1)
        self.assertEqual(int(bufs["dilation"].item()), 2)

    def test_init_accepts_preinitialized_W_and_b_and_transposed_W(self):
        C, O, kL = 2, 3, 4
        fan_in = C * kL
        W_ok = Tensor(_rng(302).standard_normal((O, fan_in)), requires_grad=True)
        b_ok = Tensor(_rng(303).standard_normal((O,)), requires_grad=True)
        c = Conv1D(C, O, kL, W=W_ok, b=b_ok, use_bias=True)
        np.testing.assert_allclose(c.W.data, W_ok.data, rtol=0, atol=0)
        np.testing.assert_allclose(c.b.data, b_ok.data, rtol=0, atol=0)

        W_t = Tensor(_rng(304).standard_normal((fan_in, O)), requires_grad=True)
        c2 = Conv1D(C, O, kL, W=W_t, use_bias=False)
        np.testing.assert_allclose(c2.W.data, W_t.data.T, rtol=0, atol=0)

        with self.assertRaises(ValueError):
            _ = Conv1D(C, O, kL, W=Tensor(np.zeros((O + 1, fan_in))))
        with self.assertRaises(ValueError):
            _ = Conv1D(C, O, kL, b=Tensor(np.zeros((O + 1,))), use_bias=True)
        with self.assertRaises(ValueError):
            _ = Conv1D(C, O, kL, b=Tensor(np.zeros((O,))), use_bias=False)

    def test_apply_state_roundtrip_and_bias_toggle(self):
        C, O, kL = 2, 3, 3
        c = Conv1D(C, O, kL, seed=0)

        W_new = _rng(305).standard_normal(c.W.data.shape)
        b_new = _rng(306).standard_normal(c.b.data.shape)
        c.apply_state(tunable=(W_new, b_new))
        np.testing.assert_allclose(c.W.data, W_new, rtol=0, atol=0)
        np.testing.assert_allclose(c.b.data, b_new, rtol=0, atol=0)

        c.apply_state(buffers={"use_bias": np.asarray(0, dtype=np.int8)})
        self.assertFalse(c.use_bias)
        self.assertEqual(len(c.parameters), 1)
        self.assertFalse(c.b.requires_grad)
        np.testing.assert_allclose(c.b.data, np.zeros_like(c.b.data), rtol=0, atol=0)

        c.apply_state(tunable=(W_new,))
        np.testing.assert_allclose(c.W.data, W_new, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            c.apply_state(tunable=(W_new, b_new))

        c.apply_state(buffers={"use_bias": np.asarray(1, dtype=np.int8)})
        self.assertTrue(c.use_bias)
        self.assertTrue(c.b.requires_grad)
        c.apply_state(tunable=(W_new, b_new))

    def test_backward_grads_match_manual_for_W_and_b(self):
        B, C, L = 2, 2, 7
        O = 3
        kL, sL, pL, dL = 3, 2, 1, 1
        rng = _rng(307)
        conv = Conv1D(C, O, kL, stride=sL, padding=pL, dilation=dL, use_bias=True, seed=5)
        X_np = rng.standard_normal((B, C, L))
        X = Tensor(X_np, requires_grad=True)
        Y = conv(X)  # (B, O, L_out)

        U = rng.standard_normal(Y.data.shape)
        Y.backward(U)

        cols = _manual_unfold1d(X_np, kL, sL, pL, dL, pad_with=conv.pad_with)
        dW_expected = np.einsum("bop,bcp->oc", U, cols)
        db_expected = U.sum(axis=(0, 2))
        np.testing.assert_allclose(conv.W.grad, dW_expected, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(conv.b.grad, db_expected, rtol=1e-6, atol=1e-8)

    def test_backward_input_grad_matches_finite_difference(self):
        B, C, L = 1, 1, 5
        O = 2
        kL = 3
        rng = _rng(308)
        conv = Conv1D(C, O, kL, seed=13)
        X0 = rng.standard_normal((B, C, L)).astype(np.float64)
        U = rng.standard_normal(conv(Tensor(X0)).data.shape).astype(np.float64)

        X = Tensor(X0.copy(), requires_grad=True)
        Y = conv(X)
        Y.backward(U)
        g_auto = X.grad.copy()

        eps = 1e-6
        g_num = np.zeros_like(X0)

        def obj(x_arr: np.ndarray) -> float:
            y = conv(Tensor(x_arr, requires_grad=False)).data
            return float(np.sum(y * U))

        for idx in np.ndindex(X0.shape):
            x_p = X0.copy()
            x_m = X0.copy()
            x_p[idx] += eps
            x_m[idx] -= eps
            g_num[idx] = (obj(x_p) - obj(x_m)) / (2.0 * eps)

        np.testing.assert_allclose(g_auto, g_num, rtol=1e-5, atol=1e-6)


class TestConv2D(ut.TestCase):
    def test_forward_matches_manual_reference_with_bias(self):
        B, C, H, W = 2, 2, 6, 7
        O = 3
        k = (3, 2)
        s = (2, 1)
        p = (1, 1)
        d = (1, 1)
        rng = _rng(400)

        conv = Conv2D(C, O, k, stride=s, padding=p, dilation=d, use_bias=True, seed=17)
        X_np = rng.standard_normal((B, C, H, W))
        X = Tensor(X_np, requires_grad=False)
        Y = conv(X)

        Y_ref = _manual_conv2d(
            X_np, conv.W.data, conv.b.data,
            kernel_size=k, stride=s, padding=p, dilation=d, pad_with=conv.pad_with
        )
        self.assertEqual(Y.data.shape, Y_ref.shape)
        np.testing.assert_allclose(Y.data, Y_ref, rtol=1e-6, atol=1e-8)

    def test_forward_matches_manual_reference_without_bias(self):
        B, C, H, W = 2, 1, 5, 5
        O = 4
        k = (2, 2)
        rng = _rng(401)
        conv = Conv2D(C, O, k, use_bias=False, seed=19)

        X_np = rng.standard_normal((B, C, H, W))
        Y = conv(Tensor(X_np, requires_grad=False))
        Y_ref = _manual_conv2d(
            X_np, conv.W.data, None,
            kernel_size=k, stride=(1, 1), padding=(0, 0), dilation=(1, 1), pad_with=conv.pad_with
        )
        self.assertEqual(len(conv.parameters), 1)
        self.assertFalse(conv.b.requires_grad)
        np.testing.assert_allclose(Y.data, Y_ref, rtol=1e-6, atol=1e-8)

    def test_input_validation_errors(self):
        conv = Conv2D(2, 3, (3, 3))
        with self.assertRaises(ValueError):
            _ = conv(Tensor(np.zeros((2, 2, 5))))     # wrong ndim
        with self.assertRaises(ValueError):
            _ = conv(Tensor(np.zeros((2, 1, 5, 5))))  # wrong channels

    def test_seeded_reproducibility_and_named_buffers(self):
        c1 = Conv2D(3, 4, (3, 5), stride=(2, 1), padding=(1, 2), dilation=(2, 1), seed=2027)
        c2 = Conv2D(3, 4, (3, 5), stride=(2, 1), padding=(1, 2), dilation=(2, 1), seed=2027)
        np.testing.assert_allclose(c1.W.data, c2.W.data, rtol=0, atol=0)
        np.testing.assert_allclose(c1.b.data, c2.b.data, rtol=0, atol=0)

        bufs = c1.named_buffers()
        self.assertEqual(int(bufs["seed"].item()), 2027)
        self.assertEqual(_decode_method_buf(bufs["method"]), c1.method)
        self.assertEqual(_decode_method_buf(bufs["nonlinearity"]), c1.nonlinearity)
        np.testing.assert_array_equal(bufs["kernel_size"], np.asarray((3, 5), dtype=np.int64))
        np.testing.assert_array_equal(bufs["stride"], np.asarray((2, 1), dtype=np.int64))
        np.testing.assert_array_equal(bufs["padding"], np.asarray((1, 2), dtype=np.int64))
        np.testing.assert_array_equal(bufs["dilation"], np.asarray((2, 1), dtype=np.int64))

    def test_init_accepts_preinitialized_W_and_b_and_transposed_W(self):
        C, O = 2, 3
        k = (2, 2)
        fan_in = C * k[0] * k[1]
        W_ok = Tensor(_rng(402).standard_normal((O, fan_in)), requires_grad=True)
        b_ok = Tensor(_rng(403).standard_normal((O,)), requires_grad=True)
        c = Conv2D(C, O, k, W=W_ok, b=b_ok, use_bias=True)
        np.testing.assert_allclose(c.W.data, W_ok.data, rtol=0, atol=0)
        np.testing.assert_allclose(c.b.data, b_ok.data, rtol=0, atol=0)

        W_t = Tensor(_rng(404).standard_normal((fan_in, O)), requires_grad=True)
        c2 = Conv2D(C, O, k, W=W_t, use_bias=False)
        np.testing.assert_allclose(c2.W.data, W_t.data.T, rtol=0, atol=0)

        with self.assertRaises(ValueError):
            _ = Conv2D(C, O, k, W=Tensor(np.zeros((O + 1, fan_in))))
        with self.assertRaises(ValueError):
            _ = Conv2D(C, O, k, b=Tensor(np.zeros((O + 1,))), use_bias=True)
        with self.assertRaises(ValueError):
            _ = Conv2D(C, O, k, b=Tensor(np.zeros((O,))), use_bias=False)

    def test_apply_state_roundtrip_and_bias_toggle(self):
        C, O = 2, 3
        k = (3, 2)
        c = Conv2D(C, O, k, seed=0)

        W_new = _rng(405).standard_normal(c.W.data.shape)
        b_new = _rng(406).standard_normal(c.b.data.shape)
        c.apply_state(tunable=(W_new, b_new))
        np.testing.assert_allclose(c.W.data, W_new, rtol=0, atol=0)
        np.testing.assert_allclose(c.b.data, b_new, rtol=0, atol=0)

        c.apply_state(buffers={"use_bias": np.asarray(0, dtype=np.int8)})
        self.assertFalse(c.use_bias)
        self.assertEqual(len(c.parameters), 1)
        self.assertFalse(c.b.requires_grad)
        np.testing.assert_allclose(c.b.data, np.zeros_like(c.b.data), rtol=0, atol=0)

        c.apply_state(tunable=(W_new,))
        with self.assertRaises(ValueError):
            c.apply_state(tunable=(W_new, b_new))

        c.apply_state(buffers={"use_bias": np.asarray(1, dtype=np.int8)})
        self.assertTrue(c.use_bias)
        self.assertTrue(c.b.requires_grad)
        c.apply_state(tunable=(W_new, b_new))

    def test_backward_grads_match_manual_for_W_and_b(self):
        B, C, H, W = 2, 2, 6, 5
        O = 3
        k = (3, 2)
        s = (2, 1)
        p = (1, 1)
        d = (1, 1)
        rng = _rng(407)
        conv = Conv2D(C, O, k, stride=s, padding=p, dilation=d, use_bias=True, seed=3)
        X_np = rng.standard_normal((B, C, H, W))
        X = Tensor(X_np, requires_grad=True)
        Y = conv(X)  # (B, O, H_out, W_out)

        U = rng.standard_normal(Y.data.shape)
        Y.backward(U)

        cols = _manual_unfold2d(X_np, k, s, p, d, pad_with=conv.pad_with)  # (B, Ck, P)
        U_flat = U.reshape(B, O, -1)  # (B, O, P)
        dW_expected = np.einsum("bop,bcp->oc", U_flat, cols)
        db_expected = U_flat.sum(axis=(0, 2))
        np.testing.assert_allclose(conv.W.grad, dW_expected, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(conv.b.grad, db_expected, rtol=1e-6, atol=1e-8)

    def test_backward_input_grad_matches_finite_difference(self):
        B, C, H, W = 1, 1, 4, 4
        O = 2
        k = (2, 2)
        rng = _rng(408)
        conv = Conv2D(C, O, k, seed=23)
        X0 = rng.standard_normal((B, C, H, W)).astype(np.float64)
        U = rng.standard_normal(conv(Tensor(X0)).data.shape).astype(np.float64)

        X = Tensor(X0.copy(), requires_grad=True)
        Y = conv(X)
        Y.backward(U)
        g_auto = X.grad.copy()

        eps = 1e-6
        g_num = np.zeros_like(X0)

        def obj(x_arr: np.ndarray) -> float:
            y = conv(Tensor(x_arr, requires_grad=False)).data
            return float(np.sum(y * U))

        for idx in np.ndindex(X0.shape):
            x_p = X0.copy()
            x_m = X0.copy()
            x_p[idx] += eps
            x_m[idx] -= eps
            g_num[idx] = (obj(x_p) - obj(x_m)) / (2.0 * eps)

        np.testing.assert_allclose(g_auto, g_num, rtol=1e-5, atol=1e-6)


class TestMeanPool1D(ut.TestCase):
    def test_forward_matches_manual_reference(self):
        B, C, L = 2, 3, 9
        kL, sL, pL, dL = 3, 2, 1, 1
        rng = _rng(500)
        X_np = rng.standard_normal((B, C, L))
        layer = MeanPool1D(kL, stride=sL, padding=pL, dilation=dL, pad_with=-0.25)

        Y = layer(Tensor(X_np, requires_grad=False))
        Y_ref = _manual_pool1d(
            X_np, kernel_size=kL, stride=sL, padding=pL, dilation=dL, pad_with=-0.25, mode="mean"
        )
        self.assertEqual(Y.data.shape, Y_ref.shape)
        np.testing.assert_allclose(Y.data, Y_ref, rtol=1e-6, atol=1e-8)

    def test_backward_input_grad_matches_manual_reference(self):
        B, C, L = 2, 2, 8
        kL, sL, pL, dL = 3, 1, 1, 2
        rng = _rng(501)
        X_np = rng.standard_normal((B, C, L))
        X = Tensor(X_np.copy(), requires_grad=True)
        layer = MeanPool1D(kL, stride=sL, padding=pL, dilation=dL, pad_with=0.5)
        Y = layer(X)
        U = rng.standard_normal(Y.data.shape)
        Y.backward(U)

        g_ref = _manual_pool1d_input_grad(
            X_np, U, kernel_size=kL, stride=sL, padding=pL, dilation=dL, pad_with=0.5, mode="mean"
        )
        np.testing.assert_allclose(X.grad, g_ref, rtol=1e-6, atol=1e-8)

    def test_parameter_contract_and_validation_errors(self):
        layer = MeanPool1D(3)
        self.assertEqual(layer.parameters, ())
        with self.assertRaises(ValueError):
            _ = layer(Tensor(np.zeros((2, 3, 4, 5))))  # ndim mismatch
        with self.assertRaises(ValueError):
            _ = MeanPool1D(5)(Tensor(np.zeros((1, 1, 2))))  # invalid output size


class TestMaxPool1D(ut.TestCase):
    def test_forward_matches_manual_reference(self):
        B, C, L = 2, 2, 10
        kL, sL, pL, dL = 4, 2, 1, 1
        rng = _rng(510)
        X_np = rng.standard_normal((B, C, L))
        # tiny monotonic bias to avoid ties in windows
        X_np += np.linspace(0.0, 1e-7, L)[None, None, :]
        layer = MaxPool1D(kL, stride=sL, padding=pL, dilation=dL, pad_with=-3.0)

        Y = layer(Tensor(X_np, requires_grad=False))
        Y_ref = _manual_pool1d(
            X_np, kernel_size=kL, stride=sL, padding=pL, dilation=dL, pad_with=-3.0, mode="max"
        )
        self.assertEqual(Y.data.shape, Y_ref.shape)
        np.testing.assert_allclose(Y.data, Y_ref, rtol=1e-6, atol=1e-8)

    def test_backward_input_grad_matches_manual_reference(self):
        B, C, L = 2, 2, 9
        kL, sL, pL, dL = 3, 2, 1, 1
        rng = _rng(511)
        X_np = rng.standard_normal((B, C, L))
        X_np += np.linspace(0.0, 1e-7, L)[None, None, :]
        X = Tensor(X_np.copy(), requires_grad=True)
        layer = MaxPool1D(kL, stride=sL, padding=pL, dilation=dL, pad_with=-2.0)
        Y = layer(X)
        U = rng.standard_normal(Y.data.shape)
        Y.backward(U)

        g_ref = _manual_pool1d_input_grad(
            X_np, U, kernel_size=kL, stride=sL, padding=pL, dilation=dL, pad_with=-2.0, mode="max"
        )
        np.testing.assert_allclose(X.grad, g_ref, rtol=1e-6, atol=1e-8)

    def test_parameter_contract_and_validation_errors(self):
        layer = MaxPool1D(3)
        self.assertEqual(layer.parameters, ())
        with self.assertRaises(ValueError):
            _ = layer(Tensor(np.zeros((2, 3, 4, 5))))  # ndim mismatch
        with self.assertRaises(ValueError):
            _ = MaxPool1D(5)(Tensor(np.zeros((1, 1, 2))))  # invalid output size


class TestMeanPool2D(ut.TestCase):
    def test_forward_matches_manual_reference(self):
        B, C, H, W = 2, 2, 6, 7
        k = (2, 3)
        s = (2, 1)
        p = (1, 0)
        d = (1, 2)
        rng = _rng(520)
        X_np = rng.standard_normal((B, C, H, W))
        layer = MeanPool2D(k, stride=s, padding=p, dilation=d, pad_with=0.25)

        Y = layer(Tensor(X_np, requires_grad=False))
        Y_flat = _manual_pool2d(
            X_np, kernel_size=k, stride=s, padding=p, dilation=d, pad_with=0.25, mode="mean"
        )
        H_out = int(np.floor((H + 2*p[0] - d[0]*(k[0] - 1) - 1) / s[0]) + 1)
        W_out = int(np.floor((W + 2*p[1] - d[1]*(k[1] - 1) - 1) / s[1]) + 1)
        Y_ref = Y_flat.reshape(B, C, H_out, W_out)
        self.assertEqual(Y.data.shape, Y_ref.shape)
        np.testing.assert_allclose(Y.data, Y_ref, rtol=1e-6, atol=1e-8)

    def test_backward_input_grad_matches_manual_reference(self):
        B, C, H, W = 1, 2, 5, 6
        k = (3, 2)
        s = (1, 2)
        p = (1, 1)
        d = (1, 1)
        rng = _rng(521)
        X_np = rng.standard_normal((B, C, H, W))
        X = Tensor(X_np.copy(), requires_grad=True)
        layer = MeanPool2D(k, stride=s, padding=p, dilation=d, pad_with=-1.0)
        Y = layer(X)
        U = rng.standard_normal(Y.data.shape)
        Y.backward(U)

        g_ref = _manual_pool2d_input_grad(
            X_np,
            U.reshape(B, C, -1),
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            pad_with=-1.0,
            mode="mean"
        )
        np.testing.assert_allclose(X.grad, g_ref, rtol=1e-6, atol=1e-8)

    def test_parameter_contract_and_validation_errors(self):
        layer = MeanPool2D((2, 2))
        self.assertEqual(layer.parameters, ())
        with self.assertRaises(ValueError):
            _ = layer(Tensor(np.zeros((2, 3, 4))))  # ndim mismatch
        with self.assertRaises(ValueError):
            _ = MeanPool2D((5, 5))(Tensor(np.zeros((1, 1, 2, 2))))  # invalid output size
        with self.assertRaises(TypeError):
            _ = MeanPool2D((2,))  # bad tuple arity


class TestMaxPool2D(ut.TestCase):
    def test_forward_matches_manual_reference(self):
        B, C, H, W = 2, 2, 6, 6
        k = (2, 2)
        s = (2, 1)
        p = (1, 1)
        d = (1, 1)
        rng = _rng(530)
        X_np = rng.standard_normal((B, C, H, W))
        X_np += np.linspace(0.0, 1e-7, H * W).reshape(1, 1, H, W)
        layer = MaxPool2D(k, stride=s, padding=p, dilation=d, pad_with=-2.0)

        Y = layer(Tensor(X_np, requires_grad=False))
        Y_flat = _manual_pool2d(
            X_np, kernel_size=k, stride=s, padding=p, dilation=d, pad_with=-2.0, mode="max"
        )
        H_out = int(np.floor((H + 2*p[0] - d[0]*(k[0] - 1) - 1) / s[0]) + 1)
        W_out = int(np.floor((W + 2*p[1] - d[1]*(k[1] - 1) - 1) / s[1]) + 1)
        Y_ref = Y_flat.reshape(B, C, H_out, W_out)
        self.assertEqual(Y.data.shape, Y_ref.shape)
        np.testing.assert_allclose(Y.data, Y_ref, rtol=1e-6, atol=1e-8)

    def test_backward_input_grad_matches_manual_reference(self):
        B, C, H, W = 1, 2, 5, 5
        k = (3, 2)
        s = (1, 1)
        p = (1, 0)
        d = (1, 1)
        rng = _rng(531)
        X_np = rng.standard_normal((B, C, H, W))
        X_np += np.linspace(0.0, 1e-7, H * W).reshape(1, 1, H, W)
        X = Tensor(X_np.copy(), requires_grad=True)
        layer = MaxPool2D(k, stride=s, padding=p, dilation=d, pad_with=-5.0)
        Y = layer(X)
        U = rng.standard_normal(Y.data.shape)
        Y.backward(U)

        g_ref = _manual_pool2d_input_grad(
            X_np,
            U.reshape(B, C, -1),
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            pad_with=-5.0,
            mode="max"
        )
        np.testing.assert_allclose(X.grad, g_ref, rtol=1e-6, atol=1e-8)

    def test_parameter_contract_and_validation_errors(self):
        layer = MaxPool2D((2, 2))
        self.assertEqual(layer.parameters, ())
        with self.assertRaises(ValueError):
            _ = layer(Tensor(np.zeros((2, 3, 4))))  # ndim mismatch
        with self.assertRaises(ValueError):
            _ = MaxPool2D((5, 5))(Tensor(np.zeros((1, 1, 2, 2))))  # invalid output size
        with self.assertRaises(TypeError):
            _ = MaxPool2D((2,))  # bad tuple arity


class TestUnfold(ut.TestCase):
    def test_unfold1d_forward_matches_manual_reference(self):
        rng = _rng(101)
        X_np = rng.standard_normal((2, 3, 9))
        kL, sL, pL, dL = 3, 2, 1, 1

        Y = unfold1d(Tensor(X_np), kL, sL, pL, dL, pad_with=0.0)
        expected = _manual_unfold1d(X_np, kL, sL, pL, dL, pad_with=0.0)
        np.testing.assert_allclose(Y.data, expected, rtol=1e-6, atol=1e-8)
        self.assertEqual(Y.data.shape, expected.shape)

    def test_unfold1d_forward_respects_padding_and_dilation(self):
        X_np = np.array([[[1., 2., 3., 4., 5.]]], dtype=np.float64)  # (1,1,5)
        Y = unfold1d(Tensor(X_np), kernel_size=3, stride=1, padding=2, dilation=2, pad_with=-1.0)
        expected = _manual_unfold1d(X_np, kernel_size=3, stride=1, padding=2, dilation=2, pad_with=-1.0)
        np.testing.assert_allclose(Y.data, expected, rtol=0, atol=0)

    def test_unfold1d_backward_matches_manual_scatter_add(self):
        rng = _rng(102)
        X = Tensor(rng.standard_normal((2, 2, 8)), requires_grad=True)
        kL, sL, pL, dL = 4, 2, 1, 1
        Y = unfold1d(X, kL, sL, pL, dL, pad_with=0.0)
        upstream = rng.standard_normal(Y.data.shape)
        Y.backward(upstream)

        expected_dX = _manual_fold1d_grad(upstream, X.data.shape, kL, sL, pL, dL)
        np.testing.assert_allclose(X.grad, expected_dX, rtol=1e-6, atol=1e-8)

    def test_unfold1d_backward_overlap_accumulation_pattern(self):
        X = Tensor(np.zeros((1, 1, 5), dtype=np.float64), requires_grad=True)
        Y = unfold1d(X, kernel_size=3, stride=1, padding=0, dilation=1, pad_with=0.0)
        Y.backward(np.ones_like(Y.data))
        expected = np.array([[[1., 2., 3., 2., 1.]]], dtype=np.float64)
        np.testing.assert_allclose(X.grad, expected, rtol=0, atol=0)

    def test_unfold2d_forward_matches_manual_reference(self):
        rng = _rng(103)
        X_np = rng.standard_normal((2, 2, 5, 6))
        k = (2, 3)
        s = (1, 2)
        p = (1, 0)
        d = (1, 1)
        Y = unfold2d(Tensor(X_np), k, s, p, d, pad_with=0.0)
        expected = _manual_unfold2d(X_np, k, s, p, d, pad_with=0.0)
        np.testing.assert_allclose(Y.data, expected, rtol=1e-6, atol=1e-8)
        self.assertEqual(Y.data.shape, expected.shape)

    def test_unfold2d_forward_respects_padding_and_dilation(self):
        X_np = np.arange(1, 10, dtype=np.float64).reshape(1, 1, 3, 3)
        Y = unfold2d(Tensor(X_np), kernel_size=(2, 2), stride=(1, 1), padding=(2, 1), dilation=(2, 1), pad_with=-5.0)
        expected = _manual_unfold2d(X_np, kernel_size=(2, 2), stride=(1, 1), padding=(2, 1), dilation=(2, 1), pad_with=-5.0)
        np.testing.assert_allclose(Y.data, expected, rtol=0, atol=0)

    def test_unfold2d_backward_matches_manual_scatter_add(self):
        rng = _rng(104)
        X = Tensor(rng.standard_normal((2, 3, 6, 5)), requires_grad=True)
        k = (3, 2)
        s = (2, 1)
        p = (1, 1)
        d = (1, 1)
        Y = unfold2d(X, k, s, p, d, pad_with=0.0)
        upstream = rng.standard_normal(Y.data.shape)
        Y.backward(upstream)

        expected_dX = _manual_fold2d_grad(upstream, X.data.shape, k, s, p, d)
        np.testing.assert_allclose(X.grad, expected_dX, rtol=1e-6, atol=1e-8)

    def test_unfold2d_backward_overlap_accumulation_pattern(self):
        X = Tensor(np.zeros((1, 1, 4, 4), dtype=np.float64), requires_grad=True)
        Y = unfold2d(X, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0), dilation=(1, 1), pad_with=0.0)
        Y.backward(np.ones_like(Y.data))
        expected = np.array(
            [[[[1., 2., 2., 1.],
               [2., 4., 4., 2.],
               [2., 4., 4., 2.],
               [1., 2., 2., 1.]]]],
            dtype=np.float64
        )
        np.testing.assert_allclose(X.grad, expected, rtol=0, atol=0)

    def test_unfold_validation_errors(self):
        with self.assertRaises(ValueError):
            _ = unfold1d(Tensor(np.zeros((2, 3, 4, 5))), 3, 1, 0, 1)  # ndim mismatch
        with self.assertRaises(ValueError):
            _ = unfold1d(Tensor(np.zeros((1, 1, 5))), 0, 1, 0, 1)      # bad kernel
        with self.assertRaises(ValueError):
            _ = unfold1d(Tensor(np.zeros((1, 1, 5))), 3, 1, -1, 1)     # bad padding
        with self.assertRaises(ValueError):
            _ = unfold1d(Tensor(np.zeros((1, 1, 2))), 5, 1, 0, 1)      # invalid output size

        with self.assertRaises(ValueError):
            _ = unfold2d(Tensor(np.zeros((1, 1, 5))), (3, 3), (1, 1), (0, 0), (1, 1))  # ndim mismatch
        with self.assertRaises(TypeError):
            _ = unfold2d(Tensor(np.zeros((1, 1, 5, 5))), 3, (1, 1), (0, 0), (1, 1))     # tuple check
        with self.assertRaises(ValueError):
            _ = unfold2d(Tensor(np.zeros((1, 1, 4, 4))), (0, 3), (1, 1), (0, 0), (1, 1)) # bad kernel
        with self.assertRaises(ValueError):
            _ = unfold2d(Tensor(np.zeros((1, 1, 2, 2))), (5, 5), (1, 1), (0, 0), (1, 1)) # invalid output size

class TestOutputShapeHelpers(ut.TestCase):
    def test_output_len_1d_matches_common_cases(self):
        self.assertEqual(output_len_1d(28, 3, stride=1, padding=1, dilation=1), 28)
        self.assertEqual(output_len_1d(28, 2, stride=2, padding=0, dilation=1), 14)
        self.assertEqual(output_len_1d(10, 3, stride=1, padding=2, dilation=2), 10)

    def test_output_len_1d_validation(self):
        with self.assertRaises(ValueError):
            _ = output_len_1d(0, 3)
        with self.assertRaises(ValueError):
            _ = output_len_1d(8, 0)
        with self.assertRaises(ValueError):
            _ = output_len_1d(8, 3, stride=0)
        with self.assertRaises(ValueError):
            _ = output_len_1d(8, 3, padding=-1)
        with self.assertRaises(ValueError):
            _ = output_len_1d(2, 5, stride=1, padding=0, dilation=1)

    def test_output_shape_2d_matches_common_cases(self):
        self.assertEqual(output_shape_2d(28, 28, 3, stride=1, padding=1, dilation=1), (28, 28))
        self.assertEqual(output_shape_2d(28, 28, 2, stride=2, padding=0, dilation=1), (14, 14))
        self.assertEqual(output_shape_2d(7, 7, (3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)), (5, 5))

    def test_output_shape_2d_validation(self):
        with self.assertRaises(TypeError):
            _ = output_shape_2d(10, 10, (3,))
        with self.assertRaises(TypeError):
            _ = output_shape_2d(10, 10, 3, stride=(1, 2, 3))
        with self.assertRaises(ValueError):
            _ = output_shape_2d(0, 10, 3)
        with self.assertRaises(ValueError):
            _ = output_shape_2d(10, 10, (0, 3))
        with self.assertRaises(ValueError):
            _ = output_shape_2d(10, 10, 3, padding=(-1, 0))
        with self.assertRaises(ValueError):
            _ = output_shape_2d(2, 2, (5, 5), stride=1, padding=0, dilation=1)


if __name__ == "__main__":
    ut.main(verbosity=2)
