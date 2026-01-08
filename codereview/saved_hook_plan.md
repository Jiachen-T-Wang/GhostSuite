# Plan: Saved tensor hooks for GhostDotProd activations

## Goal
- Replace forward-hook activation capture with `torch.autograd.graph.saved_tensors_hooks` so GhostDotProd reuses the bf16 tensors autograd already saves for backward, avoiding extra activation copies.
- Keep `./tests/torchtitan/run_train_with_ghost.sh` working with the refactored path.

## Approach
- Use a pack/unpack hook pair to intercept autograd-saved tensors.
- Maintain a per-thread stack of active module names to label tensors (compiler-style scope stack).
- Use module forward pre/post hooks to push/pop names automatically.
- Provide a context manager on the engine to scope capture around forward+backward.

## Implementation steps
1) **Saved-tensor capture manager**
   - Add a small manager (in `ghostEngines/autograd_grad_sample_dotprod.py` or a new `ghostEngines/saved_tensor_hooks.py`) that owns:
     - A thread-local stack of active module names (only for supported layer types).
     - A per-module capture list keyed by layer name for raw saved tensors.
     - `pack_hook` and `unpack_hook` functions.
   - Stack-driven labeling in `pack_hook`:
     - Bail out fast if capture is disabled.
     - Append all saved tensors to a global list; if the scope stack is non-empty, also append under the top scope name.
     - Store the raw tensor (no cloning or casting); return it unchanged.
     - Record input shapes in the forward pre-hook to disambiguate captures.
     - Later, resolve the activation for each layer by first matching the recorded input shape among non-parameter tensors; if no match, fall back to the first non-parameter, non-leaf tensor; if still none, allow a single non-parameter leaf tensor (fail fast if ambiguous or missing).
     - If no tensors are found under the scope, optionally fall back to a global saved-tensor list and match by input/flattened shape.

2) **Hook registration updates**
   - In `autograd_grad_sample_dotprod.add_hooks`:
     - Replace `_capture_activations` with forward pre/post hooks that push/pop the module name on the manager stack.
     - Keep the existing full backward hook for dot-product and train-grad computation.
     - Store the manager on the model (e.g., `model._ghost_saved_tensor_mgr`) so the engine can access it.
   - In `remove_hooks`, clean up the manager and any per-module metadata attributes.

3) **Engine context manager**
   - Add `GradDotProdEngine.saved_tensors_context()` (or similar) that:
     - Enters `torch.autograd.graph.saved_tensors_hooks(manager.pack_hook, manager.unpack_hook)`.
     - Enables capture on entry and disables it on exit (also guard against re-entry).
   - Keep `attach/detach` behavior unchanged aside from initializing the manager reference.

4) **Training loop integration**
   - Wrap forward+loss+backward in the new context manager:
     - `tests/torchtitan/torchtitan/train_with_ghost.py` inside `forward_backward_step` (around `with self.train_context` and `with self.maybe_enable_amp`).
     - `tests/benchmark_ghost_dotprod.py` in `run_loop`, `compute_ghost_dot_products`, and `compute_ghost_grad_norms`.
   - This matches the intended usage of `saved_tensors_hooks` and ensures the pack hook sees the autocast-saved tensors.

5) **Guardrails and cleanup**
   - Fail fast: if a scope has no non-leaf tensors captured, raise a clear error (no fallback to the old forward-hook capture).
   - In `_prepare_sample_grad_or_dotprod`, if `layer.activations` is missing, raise a clear error that the saved-tensor context was not active or capture failed.
   - Keep deletion of `layer.activations` and `layer.backprops` in `_apply_train_grad` to release references promptly.
   - Gate the existing debug prints behind a flag or remove them to avoid noisy logs in the new path.

## Validation
- Sanity: run `python tests/saved_hook.py` or `python tests/amp_test.py` to confirm saved tensors are bf16 under autocast.
- Integration: run `./tests/torchtitan/run_train_with_ghost.sh` for 1 step and confirm dot-product logs are produced.
- Correctness (optional): `python tests/benchmark_ghost_dotprod.py --check-correctness` and compare dot products/norms.

## Open questions
- None (use scope stack + fail-fast capture).
