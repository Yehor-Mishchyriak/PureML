## How to Contribute
- Fork this repository and clone your fork.
- Work on a feature branch based off your fork’s `development`.
- Run tests locally: `python -m unittest discover -s tests`.
- Open a focused Pull Request to `main` in the upstream repo. CI must pass before merge.
- Add/adjust tests when you change behavior and describe the rationale in the PR.

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
