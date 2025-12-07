# storage
import unittest as ut
import tempfile
from pathlib import Path
import numpy as np

from pureml.machinery import Tensor
from pureml.layers import Affine
from pureml.base import NN
import pureml.base as pm_base
from pureml.layers import BatchNorm1d
from pureml.optimizers import Adam, SGD, StepLR


# ----------------------------- Helpers -----------------------------
def _params_snapshot(model: NN) -> list[np.ndarray]:
    """Collect a deterministic copy of all tunable param arrays on the model (in declared order)."""
    arrs = []
    # Prefer a public attribute, but fall back to scanning attributes
    for name, val in model.__dict__.items():
        # Common pattern: layers as attributes exposing .W and .b as Tensors
        for an in ("W", "weight"):
            if hasattr(val, an) and isinstance(getattr(val, an), Tensor):
                arrs.append(getattr(val, an).data.copy())
        for bn in ("b", "bias"):
            if hasattr(val, bn) and isinstance(getattr(val, bn), Tensor):
                arrs.append(getattr(val, bn).data.copy())
    return arrs

def _buffers_snapshot(model: NN) -> dict[str, np.ndarray]:
    """Collect common buffer arrays (e.g., BN running stats) by name."""
    out = {}
    for name, val in model.__dict__.items():
        if BatchNorm1d is not None and isinstance(val, BatchNorm1d.__class__):
            pass  # won't hit since BN is a class, keep generic checks below
        # Generic: look for typical BN buffer names on any attribute
        if hasattr(val, "running_mean") and isinstance(val.running_mean, Tensor):
            out[f"{name}.running_mean"] = val.running_mean.data.copy()
        if hasattr(val, "running_variance") and isinstance(val.running_variance, Tensor):
            out[f"{name}.running_variance"] = val.running_variance.data.copy()
    return out


# ----------------------------- Tiny Models -----------------------------
class TinyMLP(NN):
    """Minimal MLP: Affine → (optionally BN) → Affine. Activation is irrelevant for persistence."""
    def __init__(self, use_bn: bool = False, seed: int = 0):
        self.L1 = Affine(8, 4)
        self.BN = BatchNorm1d(4) if (use_bn and BatchNorm1d is not None) else None
        self.L2 = Affine(4, 3)
        self.train()

    def predict(self, x: Tensor) -> Tensor:
        x = x.reshape(-1, x.shape[-1]) if x.data.ndim > 2 else x
        y = self.L1(x)
        if self.BN is not None:
            y = self.BN(y)
        y = self.L2(y)
        return y


# ----------------------------- Tests: Params-only -----------------------------
class TestModelParamsPersistence(ut.TestCase):
    def test_save_params_and_load_into_fresh_model(self):
        self.assertTrue(hasattr(pm_base, "save_mdl_params"), "save_mdl_params missing in pureml.base")
        # Some repos also export 'load_state' (file → model)
        if not hasattr(pm_base, "load_state"):
            self.skipTest("pureml.base.load_state not available; cannot test loading from file.")

        mdl_src = TinyMLP(use_bn=False)
        params_src = _params_snapshot(mdl_src)

        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "params_only"
            pm_base.save_mdl_params(mdl_src, raw)  # should create .pureml.zip
            pth = raw.with_suffix(".pureml.zip")
            self.assertTrue(pth.exists(), "Expected .pureml.zip file was not created.")

            # Fresh model initialized differently
            mdl_dst = TinyMLP(use_bn=False)
            # Ensure at least one param differs initially
            params_dst0 = _params_snapshot(mdl_dst)
            self.assertFalse(all(np.allclose(a, b) for a, b in zip(params_src, params_dst0)),
                             "Fresh model accidentally matches source before load.")

            # Load file into destination model (params-only path)
            pm_base.load_state(mdl_dst, pth)
            params_dst = _params_snapshot(mdl_dst)
            for a, b in zip(params_src, params_dst):
                np.testing.assert_allclose(b, a, rtol=0, atol=0, err_msg="Params mismatch after load_state().")

    def test_save_params_adds_suffix(self):
        mdl = TinyMLP()
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "my_params"
            pm_base.save_mdl_params(mdl, raw)
            self.assertTrue(raw.with_suffix(".pureml.zip").exists())


# ----------------------------- Tests: Full state (params + buffers) -----------------------------
class TestModelFullStatePersistence(ut.TestCase):
    def test_full_state_round_trip_including_bn_buffers(self):
        if not hasattr(pm_base, "save_full_state") or not hasattr(pm_base, "load_state"):
            self.skipTest("pureml.base.save_full_state/load_state not available.")

        rng = np.random.default_rng(0)
        X = Tensor(rng.standard_normal((16, 8)), requires_grad=False)

        mdl_src = TinyMLP(use_bn=True)
        # Prime BN running stats with one train forward
        _ = mdl_src.predict(X)
        params_src = _params_snapshot(mdl_src)
        bufs_src = _buffers_snapshot(mdl_src)

        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "full"
            pm_base.save_full_state(mdl_src, raw)
            pth = raw.with_suffix(".pureml.zip")
            self.assertTrue(pth.exists(), "Expected .pureml.zip file was not created.")

            # Fresh model, then load full state
            mdl_dst = TinyMLP(use_bn=True)
            pm_base.load_state(mdl_dst, pth)

            params_dst = _params_snapshot(mdl_dst)
            for a, b in zip(params_src, params_dst):
                np.testing.assert_allclose(b, a, rtol=0, atol=0)

            bufs_dst = _buffers_snapshot(mdl_dst)
            # All BN buffers that existed in src should match in dst
            self.assertTrue(len(bufs_src) > 0, "No BN buffers found on source model; test requires BN.")
            self.assertEqual(bufs_src.keys(), bufs_dst.keys())
            for k in bufs_src:
                np.testing.assert_allclose(bufs_dst[k], bufs_src[k], rtol=0, atol=0)

    def test_loading_wrong_architecture_raises(self):
        """Saving from a model with (8→4→3) then loading into (9→4→3) should fail (shape guard)."""
        if not hasattr(pm_base, "save_full_state") or not hasattr(pm_base, "load_state"):
            self.skipTest("pureml.base.save_full_state/load_state not available.")

        mdl_src = TinyMLP(use_bn=False)
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "full_arch"
            pm_base.save_full_state(mdl_src, raw)
            pth = raw.with_suffix(".pureml.zip")
            self.assertTrue(pth.exists())

            # Incompatible destination model (different input dim)
            class BadTinyMLP(NN):
                def __init__(self):
                    self.L1 = Affine(9, 4)  # 9 != 8
                    self.L2 = Affine(4, 3)
                    self.train()
                def predict(self, x: Tensor) -> Tensor:
                    return self.L2(self.L1(x))

            mdl_bad = BadTinyMLP()
            with self.assertRaises((RuntimeError, ValueError)):
                pm_base.load_state(mdl_bad, pth)


# ----------------------------- Tests: Optimizer & Scheduler -----------------------------
class TestOptimAndSchedPersistence(ut.TestCase):
    def test_optimizer_round_trip_produces_same_next_step(self):
        # Single parameter tensor; keep it simple
        pA = Tensor(np.array([0.3, -0.7, 1.2], dtype=np.float64), requires_grad=True)
        pB = Tensor(np.array([0.3, -0.7, 1.2], dtype=np.float64), requires_grad=True)
        optA = Adam([pA], lr=0.01, weight_decay=0.1, decoupled_wd=True)

        # Run a few steps to populate internal state
        for _ in range(3):
            pA.grad = np.ones_like(pA.data)
            optA.step()

        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "adam"
            optA.save_state(raw)
            pth = raw.with_suffix(".pureml.zip")
            self.assertTrue(pth.exists())

            optB = Adam([pB], lr=0.01, weight_decay=0.1, decoupled_wd=True)
            optB.load_state(pth)

            # Next identical step should produce identical parameters
            pA.grad = np.ones_like(pA.data)
            pB.grad = np.ones_like(pB.data)
            optA.step(); optB.step()
            np.testing.assert_allclose(pA.data, pB.data, rtol=1e-12, atol=1e-12)

    def test_scheduler_round_trip_matches_next_lr(self):
        p1 = Tensor(np.array([1.0]), requires_grad=False)
        p2 = Tensor(np.array([1.0]), requires_grad=False)
        opt1 = SGD([p1], lr=1.0)
        opt2 = SGD([p2], lr=1.0)

        sch1 = StepLR(opt1, step_size=3, gamma=0.1)
        # advance a few steps
        for _ in range(4):
            sch1.step()

        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "sched"
            sch1.save_state(raw)
            pth = raw.with_suffix(".pureml.zip")
            self.assertTrue(pth.exists())

            sch2 = StepLR(opt2, step_size=3, gamma=0.1)
            sch2.load_state(pth)
            # Next step must match and must update attached optimizers equally
            lr1 = sch1.step()
            lr2 = sch2.step()
            self.assertAlmostEqual(lr1, lr2, places=12)
            self.assertAlmostEqual(opt1.lr, opt2.lr, places=12)


if __name__ == "__main__":
    ut.main(verbosity=2)
