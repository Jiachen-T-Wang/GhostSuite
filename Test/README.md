# Tests for Ghost Engines

This folder contains unit tests that validate the correctness and safety of the ghost engines on simple MLPs built with `nn.Linear`.

## Test Files

### Gradient Dot Product Engine
- `test_ghost_engines_mlp.py`: Tests for `GradDotProdEngine`
  - Verify per-training-sample gradient dot products match naive baselines
  - Verify that attaching the engine does not change training dynamics

### Gradient Projection Engine (NEW)
- `test_gradproj_mlp.py`: Tests for `GradProjLoraEngine`
  - **Dimension selection**: Tests optimal k_i, k_o computation
  - **Projection initialization**: Tests Gaussian JL and orthonormal projections
  - **Non-interference**: Verifies training is unchanged with engine attached
  - **Naive equality**: Compares projected gradients against materialized computation
  - **Storage**: Tests metadata and projection saving to disk

## What's Covered

### GradDotProdEngine Tests
- Computes per-training-sample gradient dot products equal to two independent naive baselines on a tiny 2-layer MLP with synthetic data
- Verifies that attaching the engine does not change training dynamics (validation loss equivalence over a few steps)

### GradProjLoraEngine Tests
- Verifies that LoRA-style projection branches have zero impact on model forward/backward passes
- Tests that projected gradients preserve similarity structure (Johnson-Lindenstrauss property)
- Validates projection dimension selection algorithm follows theoretical optimal ratios
- Ensures proper storage and retrieval of projected gradients for offline analysis

## How the Naive Dot-Products Are Computed

Consider a Linear layer with activations `A ∈ [B, ..., d]` and output backprops `B ∈ [B, ..., p]`, where `B = n_train + n_val` and the batch is `[X_train; X_val]`.

1) Materialized-gradient baseline (batched):
- Split into train and val: `(A_train, B_train)` and `(A_val, B_val)`.
- Weight gradients:
  - Per training sample b: `grad_train[b] = einsum('...p,...d->pd', B_train[b], A_train[b])`.
  - Validation aggregate: `grad_val = einsum('np,nd->pd', B_val.reshape(-1,p), A_val.reshape(-1,d))`.
  - Dot-product vector (length `n_train`): Frobenius inner product `dp_w[b] = sum(grad_val * grad_train[b])`.
- Bias gradients:
  - `grad_bias_train[b] = sum over non-feature dims of B_train[b]` (shape `[p]`).
  - `grad_bias_val = sum over non-feature dims of B_val` (shape `[p]`).
  - `dp_b[b] = dot(grad_bias_val, grad_bias_train[b])`.
- The test matches the engine’s bf16 behavior by casting `A` and `B` to `bfloat16` for the dot-product path.

2) Per-sample autograd baseline (batch_size=1 loop):
- Compute a single validation gradient using `reduction='sum'`, then scale by `1 / (n_train + n_val)` to match the engine’s single backward pass with `reduction='mean'` on the concatenated batch.
- For each training sample i, compute its gradient with `reduction='sum'` and scale by the same factor `1 / (n_train + n_val)`.
- For every Linear’s `weight` and `bias`, compute `dot(val_grad_param, train_grad_param_i)` and collect vectors of length `n_train`.
- This provides a very direct, slow baseline to cross-check engine outputs.

## How to Run

No dataset setup is required for any tests. They use synthetic data.

### Run all tests
```bash
# Gradient Dot Product Engine tests
python Test/test_ghost_engines_mlp.py

# Gradient Projection Engine tests  
python Test/test_gradproj_mlp.py -v
```

### Run specific test methods
```bash
# Test non-interference only
python Test/test_gradproj_mlp.py TestGradProjEngine.test_non_interference

# Test projection initialization
python Test/test_gradproj_mlp.py TestGradProjEngine.test_projection_initialization
```

All tests run on CPU by default to ensure deterministic behavior.