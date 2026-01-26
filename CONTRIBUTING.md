## How to Contribute
- Fork this repository and clone your fork.
- Make sure to carefully read the [development protocols](#development-protocols).
- Work on a feature branch based off your fork’s `development`.
- Run tests locally: `python -m unittest discover -s tests`.
- Open a focused Pull Request to `main` in the upstream repo. CI must pass before merge.
- Add/adjust tests when you change behavior and describe the rationale in the PR.

## Reporting Issues
- Use GitHub Issues with a clear title and description.
- Include reproduction steps, expected vs. actual behavior, and environment details (OS, Python version).
- Share minimal code/data to reproduce (or a failing test case).
- Tag appropriately (bug, enhancement, docs) to help triage.

## Project Philosophy
- Dependencies are kept minimal: **only Python stdlib + NumPy + Zarr**. Please do not introduce other libraries.
- Favor readability first, then efficiency: keep vectorized NumPy where it matters, but make control flow and math easy to follow.
- Stick to the existing packaging layout (`src/`), unittest-style tests, and the tag-on-main release flow (publishing is maintainer-only).

## Architecture (overview and contracts)
- **Autodiff core (`machinery.py`)**: `Tensor` wraps a NumPy array with `requires_grad`, `.grad`, and a `_creator` node. Graph nodes are `TensorValuedFunction` objects that hold the forward and grad callables, inputs, and an optional per-node `fwd_ctx` dict (forward caches output under `"out"` by default). `backward()` builds a reverse topological order, seeds with ones if no grad is provided, calls each node’s grad with optional `context`, accumulates grads, unbroadcasts shapes via `_shape_safe_grad`, and frees `fwd_ctx`. `no_grad` toggles a contextvar to skip graph building.
- **Ops and math**: Elementwise ops, matmul, reshape/flatten, slicing (scatter-add backward), log/sqrt, etc., are defined in `machinery.py`. Higher-level math lives in `general_math.py` (distances, stats helpers) and `activations.py` (sigmoid, relu, tanh, softmax/log-softmax) with explicit VJPs. Losses in `losses.py` (e.g., CCE) follow the same pattern with cached forward outputs when needed.
- **Layers (`layers.py`)**: `Layer` base tracks `training`, exposes `parameters` and `named_buffers`, and supports `apply_state`. Implementations include `Affine` (W stored as (n, m), optional bias, seed-aware init buffers), `Dropout` (inverted dropout with cached mask/scale, mode-aware), `BatchNorm1d` (running mean/var buffers, mode toggle), and `Embedding` (lookup with optional pad freezing). RNG seeding uses `util.rng_from_seed`.
- **Models (`base.py`, `models/*`)**: `BaseModel`/`NN` provide the fit/predict contract, training/eval mode propagation across contained `Layer`s, parameter/buffer collectors, and ArrayStorage-backed save/load helpers. Reference models live under `models/` (e.g., `MNIST_BEATER`).
- **Data utilities (`training_utils.py`, `datasets/*`)**: `Dataset`/`TensorDataset` normalize samples to `Tensor`s; `DataLoader` batches with shuffle/drop_last and uses `util.batches_of`, seeding via `util.get_random_seed`. Packaged datasets (MNIST) are loaded via `importlib.resources` from a bundled Zarr zip.
- **Storage and misc (`util.py`)**: `ArrayStorage` wraps Zarr (LocalStore/ZipStore) with `compress_and_cleanup` context manager for `.pureml.zip` artifacts, plus helpers (`compose_steps`, RNG helpers, JSON-literal checks). Logging config lives in `logging_util.py`. Optimizers/schedulers in `optimizers.py` (SGD, AdaGrad, RMSProp, Adam; StepLR/ExponentialLR/CosineAnnealingLR) checkpoint via ArrayStorage alongside params.

## Branching Model

This repository uses a **two-branch model**:

### `development`
- Active development branch.
- New features, experiments, and refactors happen here.
- Direct pushes are allowed.
- CI may run for early feedback, but failures do **not** block work.
- This branch is allowed to be unstable.

### `main`
- Clean, tested code. *vX.Y.Z*-tagged releases are stable.
- **Protected branch**.
- **All changes must come via Pull Requests**.
- Required CI checks (tests) must pass before merge.
- No direct pushes are allowed.

### Releases
- Releases are cut from `main`.
- After merging a PR into `main`, create a version tag `v*` on the target commit (e.g., `git tag -a vX.Y.Z`).
- Push the tag (`git push origin vX.Y.Z`) to trigger the release workflow, which runs tests, builds, checks, and publishes to PyPI (maintainers only, token-protected).

## Development Protocols:

Implicit contracts and implementation patterns used across the PureML codebase.
Treat these as normative rules for new development unless explicitly amended.

## 1) Autodiff Core (machinery)
- Tensor wraps a NumPy array; the authoritative data lives in `Tensor.data`.
- `Tensor.requires_grad` gates graph building. If no input requires grad, outputs
  are created with `requires_grad=False`.
- `Tensor.grad` is `None` until populated; gradients accumulate by addition.
- `no_grad` / `is_grad_enabled` control graph creation globally.
- All ops are expressed as `TensorValuedFunction(forward_fn, grad_fn)` where:
  - `forward_fn` takes raw `np.ndarray` inputs and returns a raw ndarray/scalar.
    It must not return a `Tensor`.
  - `grad_fn` takes `(upstream_grad, *inputs)` and returns a tuple of per-input
    gradients (or `None` for non-differentiable inputs).
  - The number of gradients returned must equal the number of inputs.
- If `forward_fn` and/or `grad_fn` need cached intermediates, they must declare a
  keyword-only parameter named `context`. The engine always supplies a per-node
  dict when the signature accepts it (or when `**kwargs` is present).
- Any parameter that should be differentiated must be passed as a Tensor input
  (positional) and will appear in `grad_fn` in the same order as the inputs.
- Non-differentiated hyperparameters should be passed as kwargs to the forward
  function; if they are needed in backward, store them in `context` during
  forward and read them from `context` in `grad_fn`.
- The caller may pass `context={...}` into `TensorValuedFunction.__call__`; this
  is merged into the node context (no overwrites) and then supplied to forward
  and backward if they accept `context`.
- The forward output is cached under `context["out"]` by default.
- Cached context is cleared after the node’s gradients are computed.
- Extra kwargs are forwarded only if the target function accepts them; unknown
  kwargs are dropped to avoid `TypeError`. `context` is never overwritten by
  forwarded kwargs.
- `_shape_safe_grad` is the standard wrapper for gradient functions that need
  unbroadcasting to input shapes.

## 2) Tensor Semantics
- `Tensor(data, requires_grad=True)` coerces int/bool arrays to float.
- Object dtype is disallowed when `requires_grad=True`.
- `Tensor.numpy(copy=True)` is the safe export path. `.data` is mutable and used
  directly by optimizers and checkpoints.
- `Tensor.detach()` creates a new leaf sharing storage, with no creator and
  `requires_grad=False`. `detach_()` is the in-place variant.

## 3) Broadcasting and Shape Safety
- Broadcasting is allowed in forward ops. Backward must reduce gradients back to
  original input shapes.
- `_unbroadcast` preserves a leading batch dimension if it already matches.
- VJPs assume the leading dimension is the batch axis when present; training
  data should be shaped with batch first.
- Grad functions should be wrapped with `_shape_safe_grad` unless they already
  return gradients in exact input shapes.

## 4) Core Math Ops (general_math)
- Reductions (`mean`, `variance`, `std`, `sum`) reduce along `axis` without
  `keepdims` (current behavior).
- Gradients expand the reduced axis and broadcast to the input shape.
- If a `keepdims` option is added, gradient logic must not expand when the
  upstream already includes the reduced axis.

## 5) Activations and Losses
- Activations and losses are implemented as `TensorValuedFunction`s with stable
  numerics and cached forward outputs in context.
- Losses return scalar tensors and use mean reduction over all elements.
- `from_logits` variants perform numerically stable transforms internally and
  cache needed intermediates (softmax/log-softmax or sigmoid).
- Label smoothing is applied in forward and must be accounted for in gradients.

## 6) Layer Protocol
- Every `Layer` subclass:
  - Exposes `parameters: tuple[Tensor, ...]` (possibly empty).
  - Optionally exposes `named_buffers()` for non-trainable state.
  - Optionally implements `apply_state(tunable, buffers)` for checkpoint restore.
- `Layer.training` toggles mode and calls `on_mode_change`.
- Layer `__call__` should validate input shapes and return a `Tensor`.
- Any trainable `Tensor` must have `requires_grad=True`.
- Buffers should be JSON-safe or NumPy arrays; string metadata should use
  `np.bytes_` for serialization compatibility.

## 7) Model Protocol (base)
- `BaseModel` defines `fit()` and `predict()`; `NN` is the default neural base.
- `NN.__call__` delegates to `predict(*args, **kwargs)`.
- `NN.train()` / `NN.eval()` propagate mode to all contained `Layer` instances.
- Parameter collection scans model attributes and (one-level) containers.

## 8) State and Checkpointing
- Model params are saved as `<layer>.param.<i>` blocks in a `.pureml.zip` archive.
- Buffers are saved as `<layer>.buf.<name>` blocks.
- Full-state archives include attrs: `meta.kind = "NNState"`, `model_class`,
  and `literals` (JSON-safe top-level fields).
- `load_state` with `strict=True` enforces shape checks for params and buffers.
- Optimizers store hypers in `optim.meta` attrs and per-parameter slots in
  `optim.<slot>.<i>` blocks, plus current params in `optim.param.<i>`.

## 9) Data Protocol
- `Dataset` must implement `__len__` and `__getitem__` (int or slice).
- `TensorDataset` always returns `Tensor` outputs with `requires_grad=False`.
- `DataLoader` supports both sliceable and non-sliceable datasets and optional
  deterministic shuffling via a stored seed.
- `combine_samples` stacks tuple samples into batched tensors/arrays.

## 10) Evaluation Protocol
- `evaluation.accuracy` expects model outputs to be logits/probs or class indices.
- Uses `no_grad` and preserves the model’s prior training/eval state.

## 11) Optimizer Protocol
- Optimizers mutate `param.data` in-place based on `param.grad`.
- `zero_grad()` sets each `param.grad` to `None`.
- Weight decay supports both coupled L2 and decoupled (AdamW-style).
- Schedulers own an optimizer and update `optim.lr` in-place.

## 12) Logging and Diagnostics
- Modules use module-level loggers (`logging.getLogger(__name__)`).
- Debug logs record shape and dtype metadata; ops should avoid large payloads.

## 13) Contributor Guidelines: Adding Differentiable Operations
- Implement `forward_fn(*arrays, *, context=None, **kwargs)` that operates on raw
  NumPy arrays and returns a raw ndarray/scalar (never a `Tensor`).
- Implement `grad_fn(upstream, *arrays, *, context=None, **kwargs)` that returns
  one gradient per input (tuple length must match the number of inputs).
- If you cache intermediates in forward, add a keyword-only `context` parameter
  to both forward and backward and read/write through it.
- Use `_update_ctx(context, ...)` to stash cached values; prefer lazy callables
  when the cache is expensive and only needed in backward.
- Wrap your grad with `_shape_safe_grad` unless you already return gradients in
  the exact input shapes (including broadcasted inputs).
- Expose the op as `TensorValuedFunction(forward_fn, grad_fn)(...)` in the
  public API; keep forward/backward private to the module.
- Validate or normalize dtype/shape at the Tensor API boundary, not inside the
  forward function.

## 14) Protocol Enforcement (current checks)
- `TensorValuedFunction` rejects forward outputs that are `Tensor` instances.
- Supplying `context=...` to a `TensorValuedFunction` requires a dict; otherwise
  a `TypeError` is raised.
- The autodiff engine enforces that grad functions return a tuple with one entry
  per input.
- `Layer.apply_state()` validates:
  - `parameters` are `Tensor` and `requires_grad=True`.
  - `named_buffers()` returns a dict with string keys and values that are
    `Tensor`, `np.ndarray`, or JSON-literals.
- `BaseModel.state()` calls `layer._validate_contract()` before serializing.

