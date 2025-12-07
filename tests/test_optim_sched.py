# optim / schedulers
import unittest as ut
import tempfile
from pathlib import Path
import numpy as np

from pureml.machinery import Tensor
from pureml.layers import Affine
from pureml.optimizers import SGD, AdaGrad, RMSProp, Adam, StepLR, ExponentialLR, CosineAnnealingLR
from pureml.general_math import mean


def _rng(seed=0):
    return np.random.default_rng(seed)


class TinyModel:
    def __init__(self, n=4, m=2, seed=0):
        # Seed the layer init so tests are fully deterministic
        self.layer = Affine(n, m, seed=seed)
        self.rng = _rng(seed)

    @property
    def params(self):
        return [self.layer.W, self.layer.b]

    def loss(self, B=16):
        # Stochastic-by-default helper (kept for other tests)
        X = Tensor(self.rng.standard_normal((B, self.layer.W.data.shape[0])), requires_grad=False)
        Y = self.layer(X)
        L = mean(Y * Y)
        return L


def _loss_on(mdl: TinyModel, X: Tensor) -> Tensor:
    """Deterministic full-batch loss used in optimizer reduction tests."""
    Y = mdl.layer(X)
    return mean(Y * Y)


class TestOptimizersBasic(ut.TestCase):
    def test_sgd_step_and_zero_grad(self):
        mdl = TinyModel(n=4, m=2, seed=1)
        L = mdl.loss(B=32)
        L.backward()
        W0 = mdl.layer.W.data.copy()
        b0 = mdl.layer.b.data.copy()

        opt = SGD(mdl.params, lr=0.1, beta=0.0, weight_decay=0.0)
        opt.step()
        opt.zero_grad()

        self.assertFalse(np.allclose(mdl.layer.W.data, W0))
        self.assertFalse(np.allclose(mdl.layer.b.data, b0))
        self.assertIsNone(mdl.layer.W.grad)
        self.assertIsNone(mdl.layer.b.grad)

    def test_other_opts_reduce_loss_over_few_steps(self):
        # Fixed full-batch input for determinism (no SGD noise)
        for Opt, lr in [(AdaGrad, 0.3), (RMSProp, 0.01), (Adam, 0.01)]:
            with self.subTest(optimizer=Opt.__name__):
                seed = 2
                mdl = TinyModel(n=6, m=3, seed=seed)

                # Fixed dataset for the whole test
                rg = _rng(12345)
                X_fixed = Tensor(
                    rg.standard_normal((64, mdl.layer.W.data.shape[0])),
                    requires_grad=False
                )

                opt = Opt(mdl.params, lr=lr)

                # Record initial loss
                L0 = float(_loss_on(mdl, X_fixed).data)

                # Run several deterministic full-batch steps
                losses = [L0]
                for _ in range(40):
                    # Clear grads explicitly to avoid accumulation
                    mdl.layer.W.grad = None
                    mdl.layer.b.grad = None

                    L = _loss_on(mdl, X_fixed)
                    L.backward()
                    opt.step()

                    losses.append(float(_loss_on(mdl, X_fixed).data))

                Lmin = min(losses[1:])  # best after the first step
                self.assertLess(Lmin, L0, f"{Opt.__name__} failed to reduce loss: min={Lmin} !< L0={L0}")

    def test_weight_decay_coupled_vs_decoupled_with_momentum(self):
        # One-parameter test to isolate effect
        w_init = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        # model params as a single Tensor (no layer needed)
        p_dec = Tensor(w_init.copy(), requires_grad=True)
        p_cpl = Tensor(w_init.copy(), requires_grad=True)
        # set gradients to a fixed vector (no autograd graph necessary after this)
        g = np.array([0.5, -0.25, 0.1], dtype=np.float64)
        p_dec.grad = g.copy()
        p_cpl.grad = g.copy()

        wd = 0.4; lr = 0.1; beta = 0.9

        opt_dec = SGD([p_dec], lr=lr, beta=beta, weight_decay=wd, decoupled_wd=True)
        opt_cpl = SGD([p_cpl], lr=lr, beta=beta, weight_decay=wd, decoupled_wd=False)
        opt_dec.step()
        opt_cpl.step()

        # With momentum, decoupled and coupled produce different updates on first step
        self.assertFalse(np.allclose(p_dec.data, p_cpl.data))

    def test_adam_state_save_load_roundtrip(self):
        # Prepare a "model" with a single parameter to simplify equality checks
        w0 = np.array([0.3, -0.7, 1.2], dtype=np.float64)
        pA = Tensor(w0.copy(), requires_grad=True)
        pB = Tensor(w0.copy(), requires_grad=True)

        optA = Adam([pA], lr=0.01, weight_decay=0.1, decoupled_wd=True)
        # run a few steps to populate v/r and t
        for _ in range(3):
            pA.grad = np.ones_like(pA.data)
            optA.step()

        # Save state
        with tempfile.TemporaryDirectory() as d:
            pth = Path(d) / "adam_state"
            optA.save_state(pth)
            # Construct identical optimizer for pB and load state
            optB = Adam([pB], lr=0.01, weight_decay=0.1, decoupled_wd=True)
            optB.load_state(pth.with_suffix(".pureml.zip"))

            # Next step on both with identical gradients → identical params
            pA.grad = np.ones_like(pA.data)
            pB.grad = np.ones_like(pB.data)
            optA.step()
            optB.step()

            np.testing.assert_allclose(pA.data, pB.data, rtol=1e-12, atol=1e-12)


class TestLRSchedulers(ut.TestCase):
    def test_step_lr_progression(self):
        # base lr 1.0
        p = Tensor(np.array([1.0]), requires_grad=False)
        opt = SGD([p], lr=1.0)
        sch = StepLR(opt, step_size=2, gamma=0.5)
        lrs = [sch.step() for _ in range(5)]
        self.assertEqual(lrs, [1.0, 1.0, 0.5, 0.5, 0.25])
        # optim.lr should track
        self.assertAlmostEqual(opt.lr, lrs[-1], places=12)

    def test_exponential_lr_progression(self):
        p = Tensor(np.array([1.0]), requires_grad=False)
        opt = SGD([p], lr=2.0)
        sch = ExponentialLR(opt, gamma=0.9)
        lrs = [sch.step() for _ in range(4)]
        # steps: 0,1,2,3
        exp = [2.0, 2.0*0.9, 2.0*(0.9**2), 2.0*(0.9**3)]
        for a,b in zip(lrs, exp):
            self.assertAlmostEqual(a, b, places=12)
        self.assertAlmostEqual(opt.lr, lrs[-1], places=12)

    def test_cosine_anneal_edges_and_clamp(self):
        p = Tensor(np.array([1.0]), requires_grad=False)
        opt = SGD([p], lr=1.0)
        sch = CosineAnnealingLR(opt, T_max=10, eta_min=0.2)
        # step to 0
        lr0 = sch.step()
        self.assertAlmostEqual(lr0, 1.0, places=12)
        # halfway (step=5)
        lr5 = sch.step(5-0)  # already at 0, move to 5
        # expected cosine: 0.2 + 0.5*(1-0.2)*(1+cos(pi*0.5)) = 0.2 + 0.5*0.8*(1+0)=0.2+0.4=0.6
        self.assertAlmostEqual(lr5, 0.6, places=12)
        # beyond T_max clamps to eta_min
        lr_end = sch.step(20)  # jump past end
        self.assertAlmostEqual(lr_end, 0.2, places=12)
        self.assertAlmostEqual(opt.lr, 0.2, places=12)

    def test_scheduler_state_save_load(self):
        p1 = Tensor(np.array([1.0]), requires_grad=False)
        p2 = Tensor(np.array([1.0]), requires_grad=False)
        opt1 = SGD([p1], lr=1.0)
        opt2 = SGD([p2], lr=1.0)

        sch1 = StepLR(opt1, step_size=3, gamma=0.1)
        # advance a few steps
        for _ in range(4):
            sch1.step()
        # save
        with tempfile.TemporaryDirectory() as d:
            pth = Path(d) / "sched_state"
            sch1.save_state(pth)

            # load into a fresh scheduler bound to a fresh optimizer
            sch2 = StepLR(opt2, step_size=3, gamma=0.1)
            sch2.load_state(pth.with_suffix(".pureml.zip"))
            # states should match
            self.assertEqual(sch2.last_step, sch1.last_step)
            self.assertAlmostEqual(sch2.base_lr, sch1.base_lr, places=12)
            self.assertAlmostEqual(sch2.lr, sch1.lr, places=12)
            # and next step should produce identical lr and set on their optimizers
            lr1 = sch1.step()
            lr2 = sch2.step()
            self.assertAlmostEqual(lr1, lr2, places=12)
            self.assertAlmostEqual(opt1.lr, opt2.lr, places=12)

    def test_steplr_input_validation(self):
        p = Tensor(np.array([1.0]), requires_grad=False)
        opt = SGD([p], lr=1.0)
        with self.assertRaises(ValueError):
            StepLR(opt, step_size=0, gamma=0.5)


if __name__ == "__main__":
    ut.main(verbosity=2)
