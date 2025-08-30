# LayerNorm Grad Dot-Product Fix Plan — 2025-08-30

Owner: GhostSuite Team  
Scope: Correct LayerNorm gradient dot-product reduction and ensure aggregator compatibility.

## Problem Statement

The current LayerNorm dot-product implementation returns a `[B_train, F]` tensor for both `weight` and `bias` parameters by multiplying per-sample training gradients `[B_train, F]` with a scalar total validation gradient. This is incorrect: the per-parameter dot-product must be a per-sample scalar `[B_train]` equal to the inner product between the (aggregated) validation gradient vector and each per-sample training gradient vector. This shape mismatch breaks `GradDotProdEngine._aggregate_and_log_dot_products`, which expects `[B_train]` across all parameters.

Reference: Development/CodeReview/code-reorg-review-2025-08-30.md (High-priority bug).

## Target Behavior

- `nn.LayerNorm.weight.grad_dot_prod`: 1D tensor `[B_train]` with values equal to `⟨g_val_weight, g_train_weight[i]⟩`.
- `nn.LayerNorm.bias.grad_dot_prod`: 1D tensor `[B_train]` with values equal to `⟨g_val_bias, g_train_bias[i]⟩`.
- `GradDotProdEngine.aggregate_and_log()` sums per-parameter `[B_train]` vectors without shape errors.

## Proposed Changes

1) Compute validation gradients preserving the feature dimension:
- Weight: `grad_weight_val_total = (B_val * normalized_A_val).sum(dim=list(range(ndims-1)))  # -> [F]`
- Bias: `grad_bias_val_total = B_val.sum(dim=list(range(ndims-1)))  # -> [F]`

2) Compute per-sample training gradients (already `[B_train, F]`) and reduce via a feature-wise dot product with validation totals:
- Weight: `layer.weight.grad_dot_prod = torch.einsum('bf,f->b', per_sample_grad_weight, grad_weight_val_total)`
- Bias: `layer.bias.grad_dot_prod = torch.einsum('bf,f->b', per_sample_grad_bias, grad_bias_val_total)`

3) Type/precision:
- Keep consistency with other ghost paths: compute dot-products in float32 accumulation (`.float()` or `dtype=torch.float32` in reductions) to reduce rounding error while inputs may be bf16.

4) Validation & Safety:
- Assert resulting shapes are 1D `[B_train]` for both weight and bias before assigning to `param.grad_dot_prod`.
- Add clear `ValueError` if `val_batch_size <= 0` to match other layers.

## Affected Files

- `ghostEngines/supported_layers_grad_samplers_dotprod.py`
  - Function `_compute_layernorm_dot_product` only.

No other code paths require changes.

## Test Plan

1) New unit tests (added in this change request):
- `Test/test_layernorm_dotprod.py::test_layernorm_dotprod_shape_and_value_mismatch`
  - Currently fails: engine returns `[B,F]` and values mismatch vs autograd baseline.
- `Test/test_layernorm_dotprod.py::test_aggregator_mismatch_with_layernorm`
  - Currently fails (or raises) during aggregation due to shape mismatch.

2) After fix, replace “mismatch” tests with correctness assertions:
- Expect engine LN `grad_dot_prod` to be 1D `[B_train]` and close to autograd baseline (`atol≈5e-4`).
- Aggregator should run without error and produce a single `[B_train]` vector.

3) Regression checks:
- Run existing linear MLP tests in `Test/test_ghost_engines_mlp.py` to ensure no regression to other layers.

## Rollout Steps

1. Implement the reduction changes in `_compute_layernorm_dot_product` as above.
2. Ensure bf16 casting mirrors other dot-product paths, with fp32 accumulation for sums.
3. Run unit tests on CPU; validate both LN tests pass and existing tests remain passing.
4. Spot-check on a tiny GPT2-Small forward/backward with a concatenated batch to ensure aggregator no longer errors when LayerNorms are present.

## Risks & Mitigations

- Risk: Precision differences after changing reduction order. Mitigate by accumulating in fp32 and keeping tolerances aligned (`atol=5e-4`).
- Risk: Unexpected input shapes (e.g., dict-shaped inputs). Hooks already pass tensors; use `list(range(ndims-1))` for generic reductions.

## Acceptance Criteria

- LN dot-products are 1D `[B_train]` and numerically match autograd baselines on synthetic tests.
- Aggregation completes without shape errors when models include LayerNorm and Linear layers.
- No change to training dynamics (engine non-interference remains intact).

