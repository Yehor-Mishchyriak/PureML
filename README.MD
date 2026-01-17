# PureML — a tiny, transparent deep-learning framework in NumPy

[![PyPI](https://img.shields.io/pypi/v/ym-pure-ml)](https://pypi.org/project/ym-pure-ml/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/Yehor-Mishchyriak/PureML/blob/main/LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![Docs](https://img.shields.io/badge/docs-available-blue)](https://ymishchyriak.com/docs/PUREML-DOCS)
[![status](https://joss.theoj.org/papers/3aa26bc026244dcf3a477bd74ce4c0ff/status.svg)](https://joss.theoj.org/papers/3aa26bc026244dcf3a477bd74ce4c0ff)

![LOGO](https://raw.githubusercontent.com/Yehor-Mishchyriak/PureML/main/assets/PureML_Logo_cropped.png)

PureML is a learning-friendly deep-learning framework built entirely on top of **NumPy**. It aims to be **small, readable, and hackable** while still being practical for real experiments and teaching.

- **No hidden magic** — a Tensor class + autodiff engine with dynamic computation graph and efficient VJPs for backward passes
- **Batteries included** — core layers (Affine, Dropout, BatchNorm1d), common losses, common optimizers, and a `DataLoader`
- **Self-contained dataset demo** — a ready-to-use MNIST reader and an end-to-end “MNIST Beater” model
- **Portable persistence** — zarr-backed `ArrayStorage` with zip compression for saving/loading model state

> If you like **scikit-learn’s** simplicity and wish **deep learning** felt the same way for small/medium projects, PureML is for you.

---

## Install

PureML targets Python **3.11+** and NumPy **2.x**.

```bash
pip install ym-pure-ml
```

The only runtime deps are: `numpy`, `zarr`

---

# UPDATES: [(skip update messages)](#quickstart-fit-mnist-in-a-few-lines)

## v1.2.8 — What’s new
- **Modernization of packaging and CI/CD, documentation and attribution fixes, and the addition of community guidelines.**

## v1.2.7 — What’s new
- **BUG FIX: DataLoader's seed**
  - DataLoader now respects the provided `seed` (instead of always generating a new one) so shuffling is reproducible across runs when a seed is set.

## v1.2.6 — What’s new
- Small update: added logo to the docs and the PyPi page, as well as the README

## v1.2.5 — What’s new
- **`Affine` and `Embedding` layers now ensure grads are ALWAYS tracked for supplied W and b tensors**
- Dedicated docs site is live: https://ymishchyriak.com/docs/PUREML-DOCS (mirrors this repo and stays current).

## v1.2.4 — What’s new
- **Affine: optional bias**
  - Affine(fan_in, fan_out, bias=False, ...) is now supported.
  - When bias=False, the layer keeps a zero bias tensor with requires_grad=False and excludes it from .parameters.
  - **Checkpointing**: use_bias is persisted in named_buffers() and honored by apply_state().
    - Turning bias off during apply_state() zeroes the stored bias and freezes it.
    - Turning bias on reuses the same tensor and re-enables requires_grad.
  - Loader accepts weight arrays shaped either (fan_in, fan_out) or (fan_out, fan_in) and auto-transposes to internal (n, m).

## v1.2.3 — What’s new

  - Added utility functions `rng_from_seed(seed: int | None = None)` and `get_random_seed()` for reproducible RNG creation.  
    - `rng_from_seed` returns both the RNG and the resolved seed; if no seed is provided, `get_random_seed()` generates one using cryptographically secure OS randomness.  
  - All core layers (`Affine`, `Embedding`, `Dropout`, etc.) now support explicit seeds for deterministic initialization.  
  - Seeds and initialization methods are now persisted in checkpoints and automatically restored via each layer’s `apply_state()` method.  
  - Fixed the "Fit MNIST in a few lines" README example — it now correctly calls the `.numpy()` method (parentheses were missing).

## v1.2.2 — What’s new

- **Autograd: correct `detach` semantics (+ in-place variant)**
  - `Tensor.detach()` now returns a **new leaf tensor** that **shares storage**, has **no creator**, and **`requires_grad=False`**.
  - New **in-place** `Tensor.detach_()` for stopping tracking on the current object.
  - New `Tensor.requires_grad_(bool)` toggler (in-place), PyTorch-style.
  - Migration note: if you relied on the old in-place behavior of `detach()`, switch to `detach_()` or reassign: `x = x.detach()`.

- **Safe array export API**
  - New `Tensor.numpy(copy=True, readonly=False)` helper:
    - `copy=True` returns a defensive copy (default).
    - `readonly=True` marks the returned array non-writable (works with views or copies).
  - Rationale: keep `.data` as the **mutable** param buffer for optimizers, while providing a safe way for read-only exports.
  - Bottom-line: DO NOT ACCESS `.data` attribute directly, unless you REALLY need it! Instead, call .numpy() API. In future updates, .data may be hidden completely to avoid users accidentally mutating tensors.

- **Graph utilities: iterative and memory-safe**
  - `_collect_graph()` rewritten as an **iterative** ancestor walk (no recursion limits).
  - `zero_grad_graph()` and `detach_graph()` now use a **single traversal** and
    - free each node’s cached forward context via `fn._free_fwd_ctx()` before unlinking,
    - set `t._creator=None`, `t.grad=None`,
    - and (for `detach_graph`) **`t.requires_grad=False`** to prevent future history building.
  - Net effect: **lower peak memory** and safer teardown of large graphs.

- **Docs/logging polish**
  - Clearer docstrings for graph collection (it collects **upstream/ancestor** nodes).
  - More informative debug logs for backward/graph utilities.

## v1.2.1 — What’s new

- **BUG FIX: NN base-class API**
  - Now, self(x, y, ...) does not error within a class inheriting from NN. Previously, `__call__` function expected a single tensor, but now the signature is (*args, **kwargs), so you can define the .predict method with any signature and still use self(...) interface.
- **training_utils: TensorDataset now ALWAYS returns Tensor instances**
  - Previously, if you initialized a TensorDataset from numpy arrays, it would return numpy array instances via `__getitem__`; Now, we enforce Tensor output, which protects us from downstream errors. In case you want to access the numpy data, just call .numpy() method on your Tensor.

## v1.2.0 — What’s new

- **Autodiff-aware slicing (NumPy semantics) for `Tensor`**
  - Supports ints, slices, ellipsis (`...`), `None` (newaxis), boolean masks, and advanced integer arrays.
  - Backward pass **scatter-adds** into a zeros-like array of the input’s shape (handles repeated/overlapping indices correctly).

- **`Embedding` layer**
  - A learned lookup table for integer indices: input `(...,)` of ints → output `(..., D)` embeddings.
  - API: `Embedding(V, D, pad_idx=None, W=None)`.
  - If `pad_idx` is set, that row is **initialized to zeros** and **receives no gradient** (useful for `<PAD>` tokens).
  - Correctly accumulates gradients for repeated indices.

- **BUG FIX: `TensorValuedFunction` context merging**
  - User-supplied forward contexts are now **merged into** the node’s internal context and **persist through backward**.
  - Previously, the node could overwrite the provided context in some advanced cases, leading to missing cached values (e.g., `padding_idx`, flattened indices) during gradient computation.

---

## Quickstart: Fit MNIST in a few lines

```python
from pureml.models.neural_networks import MNIST_BEATER
from pureml.datasets import MnistDataset

# 1) Load data (train uses one-hot labels; test gives class indices)
with MnistDataset("train") as train, MnistDataset("test") as test:
    # 2) Build the tiny network: Affine(784→256) → ReLU → Affine(256→10)
    model = MNIST_BEATER().train()

    # 3) Fit on the training set
    model.fit(train, batch_size=128, num_epochs=5)

    # 4) Switch to eval: model.predict returns class indices
    model.eval()
    # Example: run on one batch from the test set
    X_test, y_test = test[:128]
    preds = model(X_test)
    print(preds.numpy()[:10])  # class ids
```

What you get out of the box:

- A tiny network that learns MNIST
- Clean logging of epoch loss
- An inference mode (`.eval()`) that returns class indices directly

---

## Core concepts

### 1) Tensors & Autodiff
PureML wraps NumPy arrays in `Tensor` objects that record operations and expose `.backward()` for gradient calculation. The Tensor supports:
- Elementwise + matmul ops (`+ - * / **`, `@`, `.T`)
- Reshaping helpers like `.reshape(...)` and `.flatten(...)`
- Non-grad ops like `.argmax(...)`
- A `no_grad` context manager for inference/metrics

> The goal is **clarity**: gradients are implemented as explicit vector-Jacobian products (VJPs) you can read in one file.

### 2) Layers
- **Affine (Linear)** — `Y = X @ W + b` (with sensible init)
- **Dropout**
- **BatchNorm1d** — with running mean/variance buffers and momentum

Layers expose:
- `.parameters` (trainables)
- `.named_buffers()` (non-trainable state)
- `.train()` / `.eval()` modes

### 3) Losses
- `MSE`
- `BCE` (probabilities) and `Sigmoid+BCE` (logits)
- `CCE` (categorical cross-entropy; supports `from_logits=True`)

### 4) Optimizers & Schedulers

PureML ships with four optimizers and three lightweight LR schedulers. All optimizers share the same interface:

- Construct with a flat list of model params (`model.parameters`) and a base learning rate.
- Call `optim.zero_grad()` → backprop → `optim.step()` each iteration.
- Optional **weight decay** is supported in both classic (coupled L2) and AdamW-style **decoupled** forms via `decoupled_wd` (defaults to `True`).
- All have robust checkpointing: `save_state("path")` writes a single `.pureml.zip`; `load_state("path")` restores hyperparameters, per-parameter slots (e.g., momentums), and even current parameter values for deterministic resume.

**Available optimizers**

- **SGD** — stochastic gradient descent with optional momentum.  
  - Args: `lr`, `beta=0.0` (momentum), `weight_decay=0.0`, `decoupled_wd=True`  
  - Update (with momentum):  
    `v ← β·v + (1−β)·g`, then (AdamW-style if decoupled) `w ← w − lr·(wd·w) − lr·v`

- **AdaGrad** — per-parameter adaptive rates via accumulated squared grads.  
  - Args: `lr`, `weight_decay=0.0`, `delta=1e-7`, `decoupled_wd=True`  
  - Accumulator: `r ← r + g⊙g`; update: `w ← w − lr·g / (sqrt(r)+δ)`

- **RMSProp** — EMA of squared grads.  
  - Args: `lr`, `weight_decay=0.0`, `beta=0.9`, `delta=1e-6`, `decoupled_wd=True`  
  - Accumulator: `r ← EMA_β(g⊙g)`; update: `w ← w − lr·g / (sqrt(r)+δ)`

- **Adam / AdamW** — first & second moments with bias correction.  
  - Args: `lr`, `weight_decay=0.0`, `beta1=0.9`, `beta2=0.999`, `delta=1e-8`, `decoupled_wd=True`  
  - Moments: `v ← EMA_{β1}(g)`, `r ← EMA_{β2}(g⊙g)`  
    Bias-correct: `v̂ = v/(1−β1^t)`, `r̂ = r/(1−β2^t)`  
    Update (AdamW if decoupled): `w ← w − lr·(wd·w) − lr· v̂/(sqrt(r̂)+δ)`

> **Coupled vs decoupled weight decay:**  
> Set `decoupled_wd=False` to apply classic L2 regularization **through the gradient** (`g ← g + wd·w`).  
> Leave it as `True` (default) for AdamW-style **parameter decay** (`w ← w − lr·wd·w`) applied separately from the gradient step.

**LR schedulers**

Schedulers wrap an optimizer and update `optim.lr` when you call `sched.step()`:

- `StepLR(optim, step_size, gamma=0.1)` → piecewise constant: multiply by `gamma` every `step_size` steps.
- `ExponentialLR(optim, gamma)` → smooth exponential decay each step.
- `CosineAnnealingLR(optim, T_max, eta_min=0.0)` → half-cosine from `base_lr` to `eta_min` over `T_max` steps.

All schedulers expose `save_state(...)` / `load_state(...)` and `step(n=1) -> new_lr`.

**Usage**

```python
from pureml.optimizers import Adam, StepLR   # also: SGD, AdaGrad, RMSProp; ExponentialLR, CosineAnnealingLR
from pureml.losses import CCE
from pureml.training_utils import DataLoader
from pureml.models.neural_networks import MNIST_BEATER
from pureml.datasets.MNIST import MnistDataset

model = MNIST_BEATER().train()

optim = Adam(model.parameters, lr=1e-3, weight_decay=1e-2)   # AdamW by default (decoupled_wd=True)
sched = StepLR(optim, step_size=1000, gamma=0.5)             # optional

for epoch in range(5):
    for X, Y in DataLoader(MnistDataset('train'), batch_size=128, shuffle=True):
        optim.zero_grad()
        logits = model(X)
        loss = CCE(Y, logits, from_logits=True)
        loss.backward()
        optim.step()
        sched.step()  # call per-batch or per-epoch as you prefer
```

### 5) Data utilities
- Minimal `Dataset` protocol (`__len__`, `__getitem__`)
- `DataLoader` with batching, shuffling, slice fast-paths, and an optional seeded RNG
- Helpers like `one_hot(...)` and `multi_hot(...)`

---

## Saving & Loading

PureML provides two levels of persistence:

- **Parameters only** — compact save/load of learnable weights
- **Full state** — parameters **+** buffers **+** top-level literals (versioned), using zarr with Blosc(zstd) compression inside a `.zip`

```python
# Save only trainable parameters
model.save("mnist_params")

# Save full state (params + buffers + literals) to .pureml.zip
model.save_state("mnist_full_state")

# Load later
model = MNIST_BEATER().eval().load_state("mnist_full_state.pureml.zip")
```

---

## MNIST dataset included

The repo ships a compressed zarr archive of MNIST (uint8, 28×28). The `MnistDataset`:
- Normalizes images to `[0,1]` float64
- Uses one-hot labels for training mode
- Supports slicing and context-manager cleanup

---

## Why PureML?

- **Read the source, learn the math.** Every gradient is explicit and local.
- **Great for teaching & research notes.** Small enough to copy into slides or notebooks.
- **Fast enough for classic datasets.** Vectorized NumPy code + light I/O.

If you need GPUs, distributed training, or huge model zoos, you should use PyTorch/JAX. PureML is intentionally light.

---

## Continuous Development (the following will be added soon)

- Convolutional layers and pooling
- Recurrent Layers
- Extra evaluation metrics (Precision, Recall, F1-Score)
- Training visualisation utilities

---

## Contributing

Issues, enhancement suggestions, and discussions are welcome!
If you want to contribute new code, please see the [guidelines](https://github.com/Yehor-Mishchyriak/PureML/blob/main/CONTRIBUTING.md).

---

## Code of Conduct
Please be respectful and constructive in all interactions. By participating in this project you agree to uphold a positive, harassment-free environment for everyone. If you experience or witness unacceptable behavior, please open a confidential issue or contact the [maintainer](https://github.com/Yehor-Mishchyriak) directly.

---

## License

Apache-2.0 — see `LICENSE` in this repo.
