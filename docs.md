# PureML Documentation

![LOGO](https://raw.githubusercontent.com/Yehor-Mishchyriak/PureML/main/assets/PureML_Logo_cropped.png)

Tiny but powerful 100% NumPy-based deep learning framework with explicit autodiff and lightweight utilities.

[![PyPI](https://img.shields.io/pypi/v/ym-pure-ml)](https://pypi.org/project/ym-pure-ml/) [![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/Yehor-Mishchyriak/PureML/blob/main/LICENSE) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) [![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/Yehor-Mishchyriak/PureML/) [![status](https://joss.theoj.org/papers/3aa26bc026244dcf3a477bd74ce4c0ff/status.svg)](https://joss.theoj.org/papers/3aa26bc026244dcf3a477bd74ce4c0ff)

---

## Installation

PureML targets Python 3.11+ with NumPy 2.x and zarr. PyPI: https://pypi.org/project/ym-pure-ml/

```bash
pip install ym-pure-ml
```

---

## Quickstart (MNIST)

```python
from pureml import Tensor
from pureml.activations import relu
from pureml.layers import Affine
from pureml.base import NN
from pureml.datasets import MnistDataset
from pureml.optimizers import Adam
from pureml.losses import CCE
from pureml.training_utils import DataLoader
from pureml.evaluation import accuracy
import time

class MNIST_BEATER(NN):

    def __init__(self) -> None:
        self.L1 = Affine(28*28, 256)
        self.L2 = Affine(256, 10)

    def predict(self, x: Tensor) -> Tensor:
        x = x.flatten(sample_ndim=2) # passing 2 because imgs in MNIST are 2D
        x = relu(self.L1(x))
        x = self.L2(x)
        if self.training:
            return x
        return x.argmax(axis=x.ndim-1) # argmax over the feature dim

with MnistDataset("train") as train, MnistDataset("test") as test:
    model = MNIST_BEATER().train()
    opt = Adam(model.parameters, lr=1e-3, weight_decay=1e-2)
    start_time = time.perf_counter()
    for _ in range(5):
        for X, Y in DataLoader(train, batch_size=128, shuffle=True):
            opt.zero_grad()
            logits = model(X)
            loss = CCE(Y, logits, from_logits=True)
            loss.backward()
            opt.step()
    end_time = time.perf_counter()
    model.eval()
    acc = accuracy(model, test, batch_size=1024)
print("Time taken: ", end_time - start_time, " sec.")
print(f"Test accuracy: {acc * 100}")
```

`MnistDataset("train")` yields `(Tensor image, one_hot Tensor label)`, normalized to `[0, 1]`. In eval mode `MNIST_BEATER.predict` returns class indices; in train mode it returns logits for loss computation.

---

## Tensors and Autograd

### Creating tensors

```python
from pureml import Tensor, is_grad_enabled, no_grad
from pureml.general_math import sum as tensor_sum

x = Tensor([[1, 2], [3, 4]], requires_grad=True)
y = Tensor(5)                      # requires_grad=False by default
```

- `requires_grad=True` tracks operations for backprop; integer/bool inputs are coerced to float if grads are tracked.
- `.dtype`, `.shape`, `.ndim` mirror the underlying NumPy array.
- `.numpy(copy=True, readonly=False)` returns a safe NumPy view/copy; prefer this over `.data` for read access.
- `.detach()` returns a new leaf tensor sharing storage with `requires_grad=False`; `.detach_()` toggles in-place. `.requires_grad_(bool)` mirrors PyTorch.

### Gradient flow

```python
out = tensor_sum(x * Tensor(2))
out.backward()        # seeds ones_like when grad is omitted
print(x.grad)         # accumulated gradient
x.zero_grad()         # sets grad to None
```

- Backprop walks creator nodes in reverse topological order. Broadcasting in the forward pass is undone automatically so input gradients match input shapes.
- `zero_grad_graph()` and `detach_graph()` clear grads/creators for the whole upstream graph.
- `no_grad()` context disables graph building; `is_grad_enabled()` reads the current flag.

### Built-in ops

- Elementwise arithmetic: `+`, `-`, `*`, `/`, `**`, unary `-`
- Comparisons: `.eq`, `.ne`, `.lt`, `.le`, `.gt`, `.ge` (return no-grad tensors)
- Reductions: `.all`, `.any`; `.argmax(axis, keepdims=False)` (non-differentiable)
- Linear algebra: `.T`, `@` (batched matmul supported)
- Reshaping: `.reshape(*shape)`, `.flatten(keep_batch=True, sample_ndim=None)`
- Indexing: `x[...]` uses NumPy semantics; backward scatter-adds into a zeros-like array of the input shape (supports advanced/repeated indices).
- Math helpers: `pureml.machinery.sqrt`, `ln`, `log2`

### Defining custom ops

`TensorValuedFunction(forward_fn, grad_fn)` builds a differentiable node. Both functions may accept a keyword-only `context` dict to reuse cached intermediates. If `grad_fn` is omitted, calling it raises `GradientNotDefined`.

---

## Activations (`pureml.activations`)

- `sigmoid(x)`
- `relu(x)`
- `tanh(x)`
- `softmax(x, axis=-1)` - stable, axis-aware
- `log_softmax(x, axis=-1)` - stable, axis-aware

Each returns a Tensor and has a Jacobian-free backward pass.

---

## Losses (`pureml.losses`)

All losses return scalar tensors (mean over all elements/samples).

- `MSE(Y, Y_hat)`: mean squared error.
- `BCE(Y, Y_hat, from_logits=False, label_smoothing=0.0)`: binary cross-entropy on probabilities or logits (`from_logits=True`). Label smoothing moves targets toward 0.5.
- `CCE(Y, Y_hat, from_logits=False, label_smoothing=0.0)`: categorical cross-entropy over the last axis. Accepts one-hot/soft labels. `from_logits=True` uses a stable log-softmax. Label smoothing mixes targets toward the uniform distribution over classes.

---

## Layers (`pureml.layers`)

Common interface: `.parameters` (trainables), `.named_buffers()` (non-trainable state), `.train()/.eval()` toggle `training` and call `on_mode_change`.

- **Affine(fan_in, fan_out, method="xavier-glorot-normal", W=None, b=None, bias=True, seed=None)**  
  Linear map `Y = X @ W + b`. Stores `W` as shape `(fan_in, fan_out)` (auto-transposes if provided as `(fan_out, fan_in)`). If `bias=False`, keeps a zero bias tensor with `requires_grad=False` and excludes it from `.parameters`. Checkpoint buffers include `method`, `seed`, and `use_bias`. Gradients are always tracked for supplied `W`/`b`.

- **Dropout(p=0.5, seed=None, training=True)**  
  Inverted dropout for 1D/2D inputs. Training mode zeros elements with prob `p` and scales survivors by `1/(1-p)`. Eval mode is identity. Buffers store `p`, `seed`, `training`.

- **BatchNorm1d(num_features, eps=1e-5, momentum=0.1, gamma=None, beta=None, running_variance=None, running_mean=None, training=True)**  
  Normalizes `(B, F)` inputs per feature. Training: uses batch mean/var and updates running stats via EMA (`running = (1-momentum)*running + momentum*batch`). Eval: uses running stats. Trainables: `gamma`, `beta`; buffers: `running_mean`, `running_variance`; `eps` is a fixed Tensor.

- **Embedding(V, D, pad_idx=None, method="xavier-glorot-normal", W=None, training=True, seed=None)**  
  Lookup table returning `(..., D)` embeddings for integer indices. If `pad_idx` is set, that row is zero-initialized and its gradient is zeroed. Gradients accumulate with `np.add.at` so repeated indices are handled. Buffers persist `padding_idx`, `seed`, `method`.

- **Initializer**: `xavier_glorot_normal(fan_in, fan_out, rng=None)` -> `(W, b)` tensors with `requires_grad=True`.

---

## General Math (`pureml.general_math`)

- `euclidean_distance(x, y)` -> scalar L2 distance
- `mean(X, axis=None)`
- `deviation(X, axis=-1)` -> `X - mean(X, axis)`
- `variance(X, axis=-1)`
- `std(X, axis=-1)` -> sqrt(variance)
- `sum(X, axis=-1 or None)` (default axis=-1)
- `ewma(running, current, beta)` -> exponential moving average (used by optimizers/BN)

All return Tensors with gradient support and honor the specified axis (negative axes allowed). Defaults: `mean` reduces over all elements when `axis=None`; `variance/std/sum` default to `axis=-1`.

---

## Data utilities (`pureml.training_utils`)

### Dataset abstractions

- **Dataset**: implement `__len__` and `__getitem__(int|slice)`.
- **TensorDataset(*arrays_or_tensors)**: wraps aligned arrays/Tensors; `__getitem__` always returns a tuple of Tensors with `requires_grad=False` (tuple length matches the number of fields). Length is inferred from the first dimension.

### DataLoader

```python
from pureml.training_utils import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)
for batch in loader:
    ...
```

- If the dataset supports slicing and `shuffle=False`, batches are contiguous slices (fast path). Otherwise indices are batched (supports shuffling).
- `__len__` returns the number of batches (drops the last incomplete batch when `drop_last=True`).
- `combine_samples_fn` controls how a list of samples is collated; defaults to `combine_samples`.
- Shuffling uses an internal `random.Random` seeded via `pureml.util.get_random_seed()` (the `seed` argument is currently not applied).

### Collation helpers

- `combine_samples(samples)`: stacks tuples of fields column-wise; falls back to stacking scalars/arrays/Tensors along a new batch axis.
- `_stack` (internal): stacks homogeneous items (Tensor -> Tensor, ndarray -> ndarray, scalar -> ndarray).

### Encoding helpers

- `one_hot(dims, label)`: scalar -> `(dims,)`; array/Tensor shape `S` -> `S + (dims,)`. Returns a Tensor if the input was a Tensor.
- `multi_hot(dims, labels)`: accepts 1D or 2D numeric arrays/Tensors or ragged list-of-lists; returns `(dims,)` or `(B, dims)` multi-hot vectors (Tensor if input was a Tensor).

---

## Models and state management (`pureml.base`)

- **BaseModel**: protocol with `.fit` and `.predict`. Provides:
  - `.state()` -> `(literals, layers)` where literals are JSON-safe non-layer fields; layers map names to parameter arrays and buffer arrays.
  - `save_state(pth, compression_level=3)` / `load_state(pth, strict=True, load_literals=True)` are implemented on `NN`.

- **NN(BaseModel)**: convenience for neural networks.
  - `.train()/ .eval()` toggle `_training` and propagate to child `Layer` attributes (including containers one level deep).
  - `.parameters` collects all layer parameters in attribute order.
  - `.save(pth)` saves only trainable parameters (backward-compatible).
  - `.save_state(pth)` saves parameters, buffers, and literals; stored with attrs `meta.kind="NNState"` and `model_class`.
  - `.load_state(pth, strict=True, load_literals=True)` loads a full state (shape-checked per parameter/buffer when `strict`).
  - `.load_params(param_pth)` loads parameters only (`<layer>.param.<i>` blocks).
  - Module-level helpers: `save_mdl_params(obj_or_dict, pth)`, `save_full_state(mdl, pth)`, `load_state(mdl, pth, ...)`, `get_mdl_params(mdl)`, `get_mdl_named_buffers(mdl)`.

Mode propagation matters for layers like Dropout/BatchNorm and for `MNIST_BEATER.predict`, which changes its return type depending on mode.

---

## Evaluation (`pureml.evaluation`)

- `accuracy(model, test_set, batch_size=32)`: top-1 accuracy. Handles logits/probabilities `(B, C)` by argmax over the last axis, or class indices `(B,)` directly. Targets can be one-hot or indices. Temporarily switches the model to eval mode (if available) and restores the previous mode.

---

## Optimizers and Schedulers (`pureml.optimizers`)

Usage pattern:

```python
opt = SGD(model.parameters, lr=0.1, beta=0.9, weight_decay=1e-4, decoupled_wd=True)
...
loss.backward()
opt.step()
opt.zero_grad()
```

Common features:
- Weight decay supports **coupled** L2 (`decoupled_wd=False` -> `g += wd * w`) and **decoupled** AdamW-style decay (`decoupled_wd=True` -> `w -= lr * wd * w`).
- Slot states are lazily initialized per parameter (shape-checked) and persisted via `save_state(path)` / `load_state(path, strict=True)`. Saves a single `.pureml.zip` with meta `{"class", "n_params", "hypers"}` plus per-parameter slots and parameter snapshots.
- `zero_grad()` sets every parameter’s `.grad` to `None`.

Optimizers:
- **SGD(model_params, lr, beta=0.0, weight_decay=0.0, decoupled_wd=True)** - optional momentum (`beta`) with exponential moving average `v`.
- **AdaGrad(model_params, lr, weight_decay=0.0, delta=1e-7, decoupled_wd=True)** - accumulates squared grads `r` and scales by `1/sqrt(r+delta)`.
- **RMSProp(model_params, lr, weight_decay=0.0, beta=0.9, delta=1e-6, decoupled_wd=True)** - EMA of squared grads `r`.
- **Adam(model_params, lr, weight_decay=0.0, beta1=0.9, beta2=0.999, delta=1e-8, decoupled_wd=True)** - first/second moments with bias correction; global step counter `_t` increments each `step()`.

Learning-rate schedulers (operate in-place on attached optimizer’s `lr`):
- **StepLR(optim, step_size, gamma=0.1, last_step=-1)**: piecewise-constant decay every `step_size` steps.
- **ExponentialLR(optim, gamma, last_step=-1)**: smooth per-step decay.
- **CosineAnnealingLR(optim, T_max, eta_min=0.0, last_step=-1)**: half-cosine from `base_lr` to `eta_min` over `T_max` steps. `step(n)` advances by `n` steps (default 1) and returns the new lr.

---

## Persistence backend (`pureml.util.ArrayStorage`)

`ArrayStorage` wraps a Zarr v3 root group for storing multiple appendable arrays with metadata. Backends:
- Local directory store `<name>.zarr` (read/write)
- Read-only ZipStore `<name>.zip`

Key methods:
- `write(arrays, to_block_named, arrays_per_chunk=None)`: append arrays (same shape/dtype) along axis 0; sets chunk length per block (defaults to 10 if unset).
- `read(from_block_named, ids=None)`: read all, a row, a slice, or an index tuple (copy).
- `block_iter(from_block_named, step=1)`: chunked iteration along axis 0.
- `delete_block(name)`: remove a block and its metadata (write mode only).
- `add_attr(key, val) / get_attr(key)`: JSON-safe root attrs (NumPy scalars/arrays coerced to python lists/scalars).
- `compress(into=None, compression_level)`: clone into a read-only ZipStore using Blosc(zstd).
- `compress_and_cleanup(output_pth, compression_level)`: contextmanager to write to a temp store then compress and remove the temp directory.

Other utilities in `pureml.util`:
- `batches_of(iterable, batch_size=-1, shuffle=False, out_as=list, ranges=False, inclusive_end=False, rng=None)`: flexible batching over sliceable or generic iterables; supports range mode (contiguous indices) when `shuffle=False`.
- `compose_steps((fn, kwargs_or_None), ...)`: compose a pipeline of unary functions with keyword args.
- `is_json_literal(x)`: checks if a value is JSON-safe for serialization.
- `get_random_seed()`, `rng_from_seed(seed=None)`: OS-random seed and seeded `np.random.default_rng`.

---

## Logging (`pureml.logging_util`)

`configure_logging(logs_dir, file_level=logging.DEBUG, console_level=logging.WARNING)` sets up the root logger once with:
- Timed rotating file handler (midnight rollover, 7 backups) writing to `logs_dir/pureml_<timestamp>.log`
- Console handler with the same format

Returns immediately if root already has handlers.

---

## Built-in datasets and models (`pureml.datasets`, `pureml.models`)

### MnistDataset

```python
from pureml.datasets import MnistDataset
train = MnistDataset("train")   # (Tensor image, one_hot label)
test  = MnistDataset("test")    # (Tensor image, class index)
```

- Backed by packaged resource `mnist-28x28_uint8.zarr.zip` (opened via `ArrayStorage` in read-only mode).
- Images are float32 in `[0, 1]` (divided by 255).
- Training labels are one-hot encoded to length 10; test labels are class indices.
- Implements context manager to close the underlying store.

### MNIST_BEATER (neural network)

- Architecture: `Affine(784 -> 256)` -> `ReLU` -> `Affine(256 -> 10)`.
- `predict(x)`: flattens the last two dims (e.g., 28x28 image) via `flatten(sample_ndim=2)`, applies the layers, and returns logits in training mode or class indices (`argmax` over the feature dim) in eval mode.
- `fit(dataset, batch_size, num_epochs)`: requires training mode; SGD(lr=0.01) + cross-entropy from logits. Uses `DataLoader(..., shuffle=True)` and logs epoch loss.

### KNN (classical)

- `KNN(k, d=euclidean_distance, standardize_features=True)`:
  - `fit(X, Y)`: stores samples/labels; optionally z-scores features per dimension (mean/std computed under `no_grad`). Enforces `1 <= k <= #samples` and matching sample counts.
  - `predict(x_q)`: standardizes the query if enabled, computes distances row-wise via `d`, selects `k` nearest labels, and breaks ties by nearest distance order.
