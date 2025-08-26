# Tests for Ghost Engines (MLP)

This folder contains unit tests that validate the correctness and safety of the gradient dot-product engine on simple MLPs built with `nn.Linear`.

## What’s Covered

- Verify that `GradDotProdEngine` computes per-training-sample gradient dot products equal to two independent naive baselines on a tiny 2-layer MLP with synthetic data.
- Verify that attaching the engine does not change training dynamics (validation loss equivalence over a few steps).

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
Run the single test file directly (CPU), no dataset setup is required.
```
python Test/test_ghost_engines_mlp.py
```