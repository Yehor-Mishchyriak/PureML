# PureML Documentation

![LOGO](https://raw.githubusercontent.com/Yehor-Mishchyriak/PureML/main/assets/PureML_Logo_cropped.png)

Tiny but powerful 100% NumPy-based deep learning framework with explicit autodiff and lightweight utilities.

[![PyPI](https://img.shields.io/pypi/v/ym-pure-ml)](https://pypi.org/project/ym-pure-ml/) ![PyPI downloads](https://img.shields.io/pypi/dm/ym-pure-ml) [![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/Yehor-Mishchyriak/PureML/blob/main/LICENSE) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) [![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/Yehor-Mishchyriak/PureML/) [![status](https://joss.theoj.org/papers/3aa26bc026244dcf3a477bd74ce4c0ff/status.svg)](https://joss.theoj.org/papers/3aa26bc026244dcf3a477bd74ce4c0ff) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18277559-blue)](https://doi.org/10.5281/zenodo.18277559)

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

### CNN Example (MNIST)

```python
from pureml.base import NN
from pureml.layers import Conv2D, MaxPool2D, Affine, BatchNorm2d, Dropout, Dropout2d, output_shape_2d
from pureml.activations import relu
from pureml import Tensor, no_grad
from pureml.training_utils import DataLoader
from pureml.datasets.MNIST import MnistDataset
from pureml.losses import CCE
from pureml.optimizers import SGD
from pureml.evaluation import accuracy


class ConvNet(NN):

    def __init__(self, in_features: int, img_res: tuple[int, int]):
        # Conv+Pool+BN Layer 1:
        H, W = img_res
        conv_kwargs = dict(kernel_size=2, stride=1, padding=0, dilation=1)
        self.conv1 = Conv2D(in_channels=in_features, out_channels=32, **conv_kwargs)
        H, W = output_shape_2d(H_in=H, W_in=W, **conv_kwargs)
        conv_kwargs["stride"] = 2
        conv_kwargs["kernel_size"] = 3
        self.pool1 = MaxPool2D(**conv_kwargs)
        H, W = output_shape_2d(H_in=H, W_in=W, **conv_kwargs)
        self.bn1 = BatchNorm2d(32)
        # Conv+Pool+BN Layer 2:
        conv_kwargs = dict(kernel_size=2, stride=1, padding=0, dilation=1)
        self.conv2 = Conv2D(in_channels=32, out_channels=64, **conv_kwargs)
        H, W = output_shape_2d(H_in=H, W_in=W, **conv_kwargs)
        conv_kwargs["stride"] = 2
        conv_kwargs["kernel_size"] = 3
        self.pool2 = MaxPool2D(**conv_kwargs)
        H, W = output_shape_2d(H_in=H, W_in=W, **conv_kwargs)
        self.bn2 = BatchNorm2d(64)
        # Conv+BN Layer 3:
        conv_kwargs = dict(kernel_size=2, stride=1, padding=0, dilation=1)
        self.conv3 = Conv2D(in_channels=64, out_channels=128, **conv_kwargs)
        H, W = output_shape_2d(H_in=H, W_in=W, **conv_kwargs)
        self.bn3 = BatchNorm2d(128)
        self.dropout2d = Dropout2d(p=0.1)
        # FC Layer 4:
        self.fc1 = Affine(128*H*W, 320, bias=False)
        # Dropout layer (fully connected block):
        self.dropout = Dropout(p=0.25)
        # FC Layer 5:
        self.fc2 = Affine(320, 10, bias=False)
    
    def predict(self, X: Tensor):
        X = X.unsqueeze(1)
        X = self.conv1(X)
        X = relu(X)
        X = self.pool1(X)
        X = self.bn1(X)
        # ---------------
        X = self.conv2(X)
        X = relu(X)
        X = self.pool2(X)
        X = self.bn2(X)
        # ---------------
        X = self.conv3(X)
        X = relu(X)
        X = self.bn3(X)
        X = self.dropout2d(X)
        # ---------------
        X = X.flatten()
        X = self.fc1(X)
        X = self.dropout(X)
        X = self.fc2(X)
        # ---------------
        if self.training:
            return X
        return X.argmax(axis=-1)
    

if __name__ == "__main__":
    model = ConvNet(1, (28, 28))
    optim = SGD(model.parameters, lr=0.01, beta=0.9, weight_decay=0.0001)
    with MnistDataset("train") as train:
        for epoch in range(1, 4):
            model.train()
            for X, Y in DataLoader(train, batch_size=64, shuffle=True):
                logits = model(X)
                loss = CCE(Y, logits, from_logits=True)
                optim.zero_grad()
                loss.backward()
                optim.step()
            model.eval()
            with no_grad():
                with MnistDataset("test") as test:
                    print(f"Accuracy after epoch {epoch} is {accuracy(model, test, batch_size=64)}")
    model.save_state("convnet_mnist_demo")
    print("Saved checkpoint to convnet_mnist_demo.pureml.zip")
```

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
- Tensor constructor args: `data` (array-like or Tensor), `requires_grad=False`, `dtype=None` (casts input), `copy=False`, `ensure_writable=True` (copies if read-only), `coerce_float_if_grad=True` (ints/bools -> float when tracking grads).

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
- Linear algebra: `.T`, `@` (batched matmul supported), `.dot(other)`
  - `Tensor.dot` supports only `1D·1D -> scalar` and `2D·2D -> matrix`.
  - Mixed-rank or rank>2 inputs are rejected with explicit shape/rank errors.
- Axis permutation: `.general_transpose(order)` for arbitrary N-D transposes with autodiff support.
- Reshaping: `.reshape(*shape)`, `.flatten(keep_batch=True, sample_ndim=None)`, `.squeeze(dim=None)`, `.unsqueeze(dim)`
- Indexing: `x[...]` uses NumPy semantics; backward scatter-adds into a zeros-like array of the input shape (supports advanced/repeated indices).
- Math helpers: `pureml.machinery.sqrt`, `ln`, `log2`

### Squeeze and Unsqueeze (`Tensor`)

Use these when you need to explicitly control singleton dimensions (`size == 1`) without breaking autograd.

- `x.squeeze(dim=None)`
  - Removes singleton dimensions.
  - `dim=None`: remove all singleton axes.
  - `dim=int | tuple[int, ...]`: remove only the specified singleton axis/axes.
  - Raises if any specified axis is not singleton.

- `x.unsqueeze(dim)`
  - Inserts a singleton dimension at axis `dim`.
  - Valid range for `dim` when `x.ndim = n`: `[-n-1, n]`.
  - Useful for making shapes broadcast-compatible or adding channel/batch-style axes.

When to use:
- Remove trailing singleton dims from model outputs before loss/metrics.
- Add a channel axis for convolution-style inputs.
- Align shapes intentionally for broadcasting (instead of relying on accidental broadcasting).

Examples:

```python
import numpy as np
from pureml import Tensor

# Remove singleton dims
logits = Tensor(np.random.randn(32, 10, 1), requires_grad=True)
logits2d = logits.squeeze(-1)      # (32, 10)

# Add channel dim
images = Tensor(np.random.randn(32, 28, 28), requires_grad=True)
images_nchw = images.unsqueeze(1)  # (32, 1, 28, 28)

# Gradients flow through both ops
out = (images_nchw * images_nchw).reshape(32, -1)
out.backward()
```

### Defining custom ops

`TensorValuedFunction(forward_fn, grad_fn)` builds a differentiable node. Both functions may accept a keyword-only `context` dict to reuse cached intermediates. If `grad_fn` is omitted, calling it raises `GradientNotDefined`.

---

## Activations (`pureml.activations`)

- `sigmoid(x: Tensor)`  
- `relu(x: Tensor)`  
- `leaky_relu(x: Tensor, negative_slope=0.01)`  
- `tanh(x: Tensor)`  
- `softmax(x: Tensor, axis=-1)` - stable, axis-aware  
- `log_softmax(x: Tensor, axis=-1)` - stable, axis-aware  
All accept any shape; `axis` is the class dimension for softmax/log_softmax. Return Tensors and provide Jacobian-free backward passes.

---

## Losses (`pureml.losses`)

All losses return scalar tensors (mean over all elements/samples).

- `MSE(Y, Y_hat)`  
  - `Y`, `Y_hat`: broadcastable tensors. Returns mean squared error over all elements.
- `BCE(Y, Y_hat, from_logits=False, label_smoothing=0.0)`  
  - `Y`: targets in {0,1} or probabilities; `Y_hat`: probabilities or logits (set `from_logits=True`).  
  - `label_smoothing` in [0,1): mixes targets toward 0.5. Returns mean over all elements.
- `CCE(Y, Y_hat, from_logits=False, label_smoothing=0.0)`  
  - `Y`: one-hot or soft labels; `Y_hat`: probabilities or logits (`from_logits=True`).  
  - Operates over the last axis; label smoothing mixes targets toward uniform over classes. Returns mean over batch/classes.

---

## Layers (`pureml.layers`)

Common interface: `.parameters` (trainables), `.named_buffers()` (non-trainable state), `.train()/.eval()` toggle `training` and call `on_mode_change`.

- **Affine(fan_in, fan_out, method="xavier-glorot-normal", W=None, b=None, bias=True, seed=None)**  
  - Linear map `Y = X @ W + b`.  
  - `method`: `"xavier-glorot-normal"` or `"kaiming-normal"`.
  - `W`: optional Tensor shaped `(fan_in, fan_out)` or `(fan_out, fan_in)` (auto-transposed).  
  - `b`: optional Tensor `(fan_out,)`; ignored when `bias=False`.  
  - Seeds init via `seed`; buffers persist `method`, `seed`, `use_bias`.  
  - Gradients are always tracked for supplied `W`/`b`.
  - Practical tip: use `"kaiming-normal"` for ReLU/LeakyReLU-heavy MLPs; use Xavier as a general default.

- **Conv1D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_bias=True, pad_with=0.0, W=None, b=None, method="kaiming-normal", nonlinearity="relu", training=True, seed=None)**  
  - Expects input shape `(B, C, L)` and returns `(B, out_channels, L_out)`.  
  - Output length: `L_out = floor((L + 2*pL - dL*(kL - 1) - 1) / sL) + 1`.  
  - `kernel_size`, `stride`, `padding`, `dilation`: int values controlling receptive field and output resolution.  
  - `W`: optional Tensor shaped `(out_channels, in_channels*kL)` or `(in_channels*kL, out_channels)` (auto-transposed).  
  - `b`: optional Tensor of shape `(out_channels,)`; if `use_bias=False`, bias is disabled and excluded from `.parameters`.  
  - Checkpointing support includes both trainables and layer config via `.named_buffers()` / `.apply_state()`.
  - Use when data has one spatial/temporal axis (signals, token-level feature maps, sensor streams).

- **Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_bias=True, pad_with=0.0, W=None, b=None, method="kaiming-normal", nonlinearity="relu", training=True, seed=None)**  
  - Expects input shape `(B, C, H, W)` and returns `(B, out_channels, H_out, W_out)`.  
  - Output size:
    - `H_out = floor((H + 2*pH - dH*(kH - 1) - 1) / sH) + 1`
    - `W_out = floor((W + 2*pW - dW*(kW - 1) - 1) / sW) + 1`  
  - `kernel_size`, `stride`, `padding`, `dilation`: int or tuple values controlling receptive field and output resolution.  
  - `W`: optional Tensor shaped `(out_channels, in_channels*kH*kW)` or `(in_channels*kH*kW, out_channels)` (auto-transposed).  
  - `b`: optional Tensor of shape `(out_channels,)`; if `use_bias=False`, bias is disabled and excluded from `.parameters`.  
  - Checkpointing support includes both trainables and layer config via `.named_buffers()` / `.apply_state()`.
  - Use when data has two spatial axes (images, spectrogram-like feature maps).

- **MaxPool1D(kernel_size, stride=1, padding=0, dilation=1, pad_with=0.0, training=True)**  
  - 1D max pooling.  
  - Expects input shape `(B, C, L)` and returns `(B, C, L_out)`.  
  - Output length: `L_out = floor((L + 2*pL - dL*(kL - 1) - 1) / sL) + 1`.  
  - Use to downsample 1D feature maps while keeping the strongest local response.

- **MeanPool1D(kernel_size, stride=1, padding=0, dilation=1, pad_with=0.0, training=True)**  
  - 1D average pooling.  
  - Expects input shape `(B, C, L)` and returns `(B, C, L_out)` using the same output formula as MaxPool1D.  
  - Use for smoother downsampling in 1D pipelines.

- **MaxPool2D(kernel_size, stride=1, padding=0, dilation=1, pad_with=0.0, training=True)**  
  - 2D max pooling.  
  - Expects input shape `(B, C, H, W)` and returns `(B, C, H_out, W_out)`.  
  - Output size:
    - `H_out = floor((H + 2*pH - dH*(kH - 1) - 1) / sH) + 1`
    - `W_out = floor((W + 2*pW - dW*(kW - 1) - 1) / sW) + 1`  
  - Use to reduce spatial resolution and add local translation tolerance in CNN blocks.

- **MeanPool2D(kernel_size, stride=1, padding=0, dilation=1, pad_with=0.0, training=True)**  
  - 2D average pooling.  
  - Expects input shape `(B, C, H, W)` and returns `(B, C, H_out, W_out)` with the same output formulas as MaxPool2D.  
  - Use when you want downsampling with less emphasis on isolated peaks.

- **Dropout(p=0.5, seed=None, training=True)**  
  - Inverted dropout for 1D/2D inputs.  
  - `p`: drop probability in [0,1]; `seed`: reproducible masks; `training`: initial mode.  
  - Training zeros elements with prob `p` and scales by `1/(1-p)`; eval is identity. Buffers store `p`, `seed`, `training`.

- **Dropout2d(p=0.5, seed=None, training=True)**  
  - Spatial (channel-wise) inverted dropout for 4D inputs `(B, C, H, W)`.  
  - Drops whole feature maps per sample (one Bernoulli decision per `(B, C)` channel map), then scales survivors by `1/(1-p)`.  
  - Best used in convolutional blocks before flattening; eval mode is identity. Buffers store `p`, `seed`, `training`.

- **BatchNorm1d(num_features, eps=1e-5, momentum=0.1, gamma=None, beta=None, running_variance=None, running_mean=None, training=True)**  
  - Normalizes `(B, F)` inputs per feature.  
  - `eps`: added inside sqrt for stability (stored as Tensor).  
  - `momentum`: EMA coefficient for running stats (`running = (1-m)*running + m*batch`).  
  - Optional trainables `gamma`, `beta` (shape `(F,)`); optional buffers to resume `running_mean/variance`.  
  - Training uses batch stats and updates running; eval uses running only.

- **BatchNorm2d(num_features, eps=1e-5, momentum=0.1, gamma=None, beta=None, running_variance=None, running_mean=None, training=True)**  
  - Channel-wise BatchNorm for NCHW tensors `(B, C, H, W)`.  
  - `num_features` is the channel count `C`.  
  - Uses the same behavior and checkpoint semantics as BatchNorm1d (train uses current-batch stats, eval uses running stats).  
  - Recommended normalization layer in convolutional blocks.

- **LayerNorm1d(num_features, gamma=None, beta=None, eps=1e-5, bias=True, training=True)**  
  - Normalizes across the **last axis** (feature axis), so input must end with `num_features` (e.g. `(B, F)` or `(B, T, F)`).  
  - Per-sample normalization: computes mean/variance on the last axis only.  
  - Affine parameters: learnable `gamma` and optional `beta` (both shape `(F,)`).  
  - If `bias=False`, `beta` is disabled (not trainable and excluded from `.parameters`).  
  - Buffers persist `eps` and `use_bias`; state loading restores both and validates parameter payloads.

- **Embedding(V, D, pad_idx=None, method="xavier-glorot-normal", W=None, training=True, seed=None)**  
  - `V`: vocab size; `D`: embedding dim.  
  - `pad_idx`: optional int; that row is zeroed and receives no grad.  
  - `W`: optional Tensor `(V, D)` init; else Xavier/Glorot with `seed`.  
  - Gradients accumulate correctly for repeated indices. Buffers persist `padding_idx`, `seed`, `method`.

- **Initializers**:
  - `xavier_glorot_normal(fan_in, fan_out, rng=None)` -> `(W, b)` tensors with `requires_grad=True`.
  - `kaiming_normal(fan_in, fan_out, nonlinearity="relu", param=None, rng=None)` -> `(W, b)` tensors with `requires_grad=True` (fan-in mode).
  - `calculate_gain(nonlinearity, param=None)` -> recommended gain scalar for supported nonlinearities (useful when choosing initialization for your activation stack).

### Unfold helpers

- `unfold1d(X, kernel_size, stride, padding, dilation, pad_with=0.0)`  
  - Input shape: `(B, C, L)`  
  - Output shape: `(B, C*kL, L_out)` where `L_out = floor((L + 2*pL - dL*(kL - 1) - 1) / sL) + 1`  
  - Backward pass uses scatter-add into padded input space, then crops back to `(B, C, L)`.

- `unfold2d(X, kernel_size, stride, padding, dilation, pad_with=0.0)`  
  - Input shape: `(B, C, H, W)`  
  - Output shape: `(B, C*kH*kW, H_out*W_out)` where  
    - `H_out = floor((H + 2*pH - dH*(kH - 1) - 1) / sH) + 1`  
    - `W_out = floor((W + 2*pW - dW*(kW - 1) - 1) / sW) + 1`  
  - Backward pass uses scatter-add into padded input space, then crops back to `(B, C, H, W)`.

### Output-shape helpers

- `output_len_1d(L_in, kernel_size, stride=1, padding=0, dilation=1)`  
  - Returns `L_out` for Conv1D/MaxPool1D/MeanPool1D-compatible settings.
  - Uses the same formula as the 1D layers:
    - `L_out = floor((L_in + 2*pL - dL*(kL - 1) - 1) / sL) + 1`
  - Raises a clear error when arguments are invalid or produce non-positive output length.

- `output_shape_2d(H_in, W_in, kernel_size, stride=1, padding=0, dilation=1)`  
  - Returns `(H_out, W_out)` for Conv2D/MaxPool2D/MeanPool2D-compatible settings.
  - Accepts `int` or `(h, w)` tuples for `kernel_size`, `stride`, `padding`, and `dilation`.
  - Uses the same formulas as the 2D layers:
    - `H_out = floor((H_in + 2*pH - dH*(kH - 1) - 1) / sH) + 1`
    - `W_out = floor((W_in + 2*pW - dW*(kW - 1) - 1) / sW) + 1`
  - Raises a clear error when arguments are invalid or produce non-positive output size.

---

## General Math (`pureml.general_math`)

- `euclidean_distance(x, y)` -> scalar L2 distance. Args: Tensors `x`, `y`; caches diff and norm.
- `mean(X, axis=None)` -> mean over axis (None means all elements). Returns broadcastable grad (1/N).
- `deviation(X, axis=-1)` -> `X - mean(X, axis)`. Caches mean for backward.
- `variance(X, axis=-1)` -> mean of squared deviations along axis. Grad scales by `2/N * dev`.
- `std(X, axis=-1)` -> sqrt(var + 1e-12) along axis. Grad uses cached std/dev.
- `sum(X, axis=-1 or None)` -> sum over axis; grad broadcasts upstream.
- `ewma(running, current, beta)` -> exponential moving average `beta*running + (1-beta)*current`.

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

- Args:
  - `dataset`: implements `__len__` and `__getitem__` (int or slice).
  - `batch_size`: items per batch (must be > 0).
  - `shuffle`: if True, shuffles indices each epoch using an internal `random.Random`.
  - `drop_last`: drop final incomplete batch when True.
  - `combine_samples_fn`: how to collate a list of samples; defaults to `combine_samples`.
  - `seed`: optional int; if set, shuffling is reproducible. If omitted, a secure seed from `util.get_random_seed()` is used.
- If the dataset supports slicing and `shuffle=False`, batches are contiguous slices (fast path). Otherwise indices are batched (supports shuffling).
- `__len__` returns the number of batches (drops the last incomplete batch when `drop_last=True`).

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
  - `.load_params(param_pth)` loads parameters-only checkpoints from `<layer>.<i>` blocks.
  - Note: full-state checkpoints use `<layer>.param.<i>` plus `<layer>.buf.<name>` blocks.
  - Module-level helpers: `save_mdl_params(obj_or_dict, pth)`, `save_full_state(mdl, pth)`, `load_state(mdl, pth, ...)`, `get_mdl_params(mdl)`, `get_mdl_named_buffers(mdl)`.

Mode propagation matters for layers like Dropout/BatchNorm and for `MNIST_BEATER.predict`, which changes its return type depending on mode.

---

## Evaluation (`pureml.evaluation`)

- `accuracy(model, test_set, batch_size=32)`: top-1 accuracy.  
  - `model`: PureML model implementing `__call__/predict` and optional `train()/eval()`.  
  - `test_set`: Dataset yielding `(X, Y)`; Y can be one-hot or class indices.  
  - `batch_size`: DataLoader batch size.  
  Handles logits/probabilities `(B, C)` by argmax over last axis, or class indices `(B,)` directly. Temporarily switches the model to eval mode and restores the prior mode.

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
- **SGD(model_params, lr, beta=0.0, weight_decay=0.0, decoupled_wd=True)**  
  - `lr`: learning rate; `beta`: momentum (0 disables momentum).  
  - `weight_decay`: L2; `decoupled_wd=True` applies AdamW-style decay, else coupled.  
  - Slots: momentum buffer `_v`.
- **AdaGrad(model_params, lr, weight_decay=0.0, delta=1e-7, decoupled_wd=True)**  
  - `delta`: epsilon for numerical stability.  
  - Slots: accumulator `_r` of squared grads.
- **RMSProp(model_params, lr, weight_decay=0.0, beta=0.9, delta=1e-6, decoupled_wd=True)**  
  - `beta`: EMA coefficient for squared grads; `delta`: epsilon.  
  - Slots: accumulator `_r`.
- **Adam(model_params, lr, weight_decay=0.0, beta1=0.9, beta2=0.999, delta=1e-8, decoupled_wd=True)**  
  - `beta1`, `beta2`: EMA coefficients for first/second moments; `delta`: epsilon.  
  - Slots: `_v` (first moment), `_r` (second moment), `_t` (step counter).

Learning-rate schedulers (operate in-place on attached optimizer’s `lr`):
- **StepLR(optim, step_size, gamma=0.1, last_step=-1)**: decay `lr *= gamma` every `step_size` steps. `last_step` lets you resume. `step(n)` advances by `n` (default 1).
- **ExponentialLR(optim, gamma, last_step=-1)**: smooth per-step decay `lr *= gamma` each `step()`. `last_step` for resume.
- **CosineAnnealingLR(optim, T_max, eta_min=0.0, last_step=-1)**: half-cosine from `base_lr` to `eta_min` over `T_max` steps (no restarts; lr stays at `eta_min` after `T_max`). `step(n)` advances by `n` and returns new lr.

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
- Args: `mode` in {"train","test"}.  
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

---

## Contributing and Protocols

For development workflow, architecture contracts, protocol rules, and release/branching guidance, see `CONTRIBUTING.md` in the repository root.
