# Plan: Tensor Hook + Activation Masking + Norm grad_input Fix

## Goals
- Replace module-level full backward hooks with output tensor hooks to compute dot products early.
- Keep the saved-tensors pack_hook / activation resolution path intact.
- Use activation masking in `unpack_hook` so autograd computes **train-only** parameter gradients.
- For RMSNorm/LayerNorm, repair `grad_input` in a full backward hook to preserve upstream dot products.

## Current flow (baseline)
- `ghostEngines/autograd_grad_sample_dotprod.py:add_hooks` registers:
  - forward pre/post hooks for scope tracking and input shape capture
  - `register_full_backward_hook` to compute dot products and manual train gradients
- `_prepare_sample_grad_or_dotprod` computes dot products and stores backprops for `_apply_train_grad`.
- `_apply_train_grad` computes per-layer train-only gradients and writes `param.train_grad`.
- `GradDotProdEngine.prepare_gradients` copies `train_grad` into `.grad` for optimizer.

## Revised approach
1) **Output tensor hook for dot products**
   - In `ghostEngines/autograd_grad_sample_dotprod.py:add_hooks`, replace
     `register_full_backward_hook` with a forward hook that registers
     `output.register_hook(grad_hook)` (see `tests/batch_selection.py`).
   - `grad_hook` responsibilities:
     - **Dot product**: compute dot products using the full `grad_output` via a refactored helper.
     - **No grad modification**: return `grad_output` unchanged.
     - **Cache activation for norm fix**: for `RMSNorm`/`LayerNorm`, keep the resolved activation on the layer
       so the full backward hook can recompute `grad_input` later.
     - **Cleanup**: for safe layers, immediately drop temp attributes; for norm layers, defer cleanup until after the full backward hook runs.

2) **Activation masking in `_NamedSavedTensorManager.unpack_hook`**
   - Goal: mask **only** the validation slice of saved activations so autograd parameter gradients are computed from training samples.
   - Extend manager to store metadata for saved tensors:
     - `self._tensor_meta[id(x)] = {"layer_name": name}` in `pack_hook` when a scope is active.
     - `self._layers_by_name` registry populated in `add_hooks` to resolve `layer_name` → module.
     - Clear `_tensor_meta` on `disable()` and per-layer cleanup to avoid leaks.
   - Determine validation indices cleanly:
     - In forward pre-hook, store `layer._ghost_input_shape` and `layer._ghost_train_bs`:
       - `total_bs = inputs[0].shape[0]`
       - `train_bs = total_bs - val_batch_size`
       - `tokens_per_sample = prod(input_shape[1:-1])` when needed
     - In `unpack_hook`, if tensor matches activation shape for that layer:
       - **Exact shape**: if `tuple(x.shape) == input_shape`, zero `x[train_bs:]`.
       - **Flattened shape**: if `tuple(x.shape) == flat_shape`, compute
         `train_rows = train_bs * tokens_per_sample` and zero `x[train_rows:]`.
     - Return a **masked copy** (`x.clone()` or `x * mask`) so the saved activation used for dot products remains unmodified.

3) **Full backward hook to correct grad_input for RMSNorm/LayerNorm**
   - Still **mask activations** for these layers so parameter grads are train-only.
   - Add `register_full_backward_hook` **only** for `nn.RMSNorm` and `nn.LayerNorm`.
   - The hook recomputes **grad_input** using the original (unmasked) activation and the incoming `grad_output`,
     then returns the corrected `grad_input` so upstream gradients (and dot products for earlier layers) remain correct.
   - This runs after parameter gradients are computed, which is acceptable because we want train-only parameter grads.

4) **Grad_input formulas for norm layers**
   - Implement helpers (e.g., `_compute_rmsnorm_grad_input`, `_compute_layernorm_grad_input`)
     that match PyTorch math, using the **original activation** cached on the layer:
     - **RMSNorm** (normalize over last dim, size `D`):
       - `inv_rms = rsqrt(mean(x^2, dim=-1, keepdim=True) + eps)`
       - `x_hat = x * inv_rms`
       - `go = grad_output * weight`
       - `grad_input = inv_rms * (go - x_hat * mean(go * x_hat, dim=-1, keepdim=True))`
     - **LayerNorm** (normalize over `normalized_shape`, size `N`):
       - `mean = x.mean(dims, keepdim=True)`
       - `var = x.var(dims, unbiased=False, keepdim=True)`
       - `rstd = rsqrt(var + eps)`
       - `x_hat = (x - mean) * rstd`
       - `go = grad_output * weight`
       - `grad_input = (1/N) * rstd * (N*go - sum(go) - x_hat*sum(go*x_hat))`
   - Use a stable compute dtype (likely `float32`) and cast back to `grad_output.dtype` before returning.

5) **GradDotProdEngine updates**
   - `prepare_gradients()` should tolerate the absence of `train_grad` and simply use `.grad`.
   - `clear_gradients()` should delete `train_grad` if present, but not require it.

6) **Refactor dot-product helper(s)**
   - Split `_prepare_sample_grad_or_dotprod` into:
     - `_compute_dotprod_from_backprops(layer, backprops, val_batch_size, loss_reduction, log_grad_norms)`
       - activation resolution + reshaping
       - dot-product computation
     - `_cleanup_layer_state(layer)` for attribute cleanup + manager clearing.
   - `grad_hook` should call `_compute_dotprod_from_backprops` and:
     - immediately `_cleanup_layer_state` for safe layers
     - defer `_cleanup_layer_state` until the norm-layer backward hook finishes

7) **Tests / validation**
   - Add a small MLP regression to confirm:
     - `param.grad` on maskable layers matches a train-only baseline.
     - dot products remain identical to naive computation (`tests/benchmark_ghost_dotprod.py`).
   - Add a norm-layer case (LayerNorm/RMSNorm) to ensure:
     - parameter grads are train-only (from activation masking)
     - upstream dot products are correct (from grad_input correction).

## Evaluation of the norm-layer grad_input fix
- **Correctness:** Full backward hooks can return a new `grad_input` used by upstream modules, even though
  parameter grads are already computed. This fits our goal: train-only parameter grads + full upstream grads.
- **Cost:** Extra compute for norm layers (mean/var or RMS recomputation), but only for those layers.
- **Risk:** Numerical differences if formulas or dtypes differ from PyTorch kernels; mitigate by matching
  PyTorch’s math (biased variance, eps, dtype casts) and validating against a baseline.

## Key implementation note
This plan preserves upstream gradients (and thus earlier-layer dot products) by **not** modifying `grad_output`.
Train-only parameter gradients are produced by masking **activations** at `unpack_hook` time.
For RMSNorm/LayerNorm, a full backward hook recomputes and returns `grad_input` using the **original** activation.
