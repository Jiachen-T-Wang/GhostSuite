# Memory/MFU Investigation (v2) — Ghost GradDotProd

## Observations from the code
- **Saved tensors are retained for the entire backward.**
  - `ghostEngines/autograd_grad_sample_dotprod.py:_NamedSavedTensorManager.pack_hook` pushes every saved tensor into `_captured_all` (and per-layer lists), and nothing is dropped until `disable()` after backward. This keeps large SDPA tensors (q/k/v, softmax stats) alive across the whole backward, preventing layer‑by‑layer release.
- **Activation masking clones tensors.**
  - `_NamedSavedTensorManager.unpack_hook` does `masked = x.clone()` for every matched activation before zeroing the val slice. That doubles memory for each masked activation (attention/FFN inputs) and adds bandwidth/compute.
  - Embedding masking also clones `grad_output` (see `_mask_embedding_grad_output`).
- **Extra compute per layer.**
  - Dot-product ops run for every supported layer with FP32 accumulators (`_compute_*_dot_product`), plus RMSNorm/LayerNorm grad_input recomputation and LayerNorm train-grad override. These add matmuls/reductions on top of the model’s backward, lowering TPS/MFU.
- **Combined batch size is larger.**
  - Ghost concatenates train + val (e.g., 12+4), so activations/backprops are larger than baseline even before hooks.
- **Temporary activation caching.**
  - Norm layers keep `_ghost_saved_activation`/`activations` alive until their backward hook runs, extending lifetimes for those tensors during backward (though not the primary factor vs. the two items above).

## Likely dominant memory bottlenecks
1) **Pinned saved tensors (no per-layer release):** `_captured_all` holds references to every saved tensor until the very end of backward, so SDPA intermediates and large activations cannot be freed after their layer finishes. This inflates peak memory versus the baseline where autograd releases them layer-by-layer.
2) **Cloning during masking:** `unpack_hook` clones every matched activation; for large hidden states and sequence length, this is a near 2× multiplier on saved-activation footprint. The clone + zeroing also adds bandwidth, contributing to the MFU drop.

## Contributors to MFU drop
- Extra dot-product matmuls/reductions per layer (FP32 accum).
- Activation/grad clones for masking.
- Norm-layer grad_input recomputation + LayerNorm train-grad recompute.
- Larger batch (train+val) increases base compute/memory traffic.

## Double-check points (suggested next steps, not executed)
- Profile peak memory with autograd profiler to confirm retention spans the whole backward (look for SDPA saved tensors persisting until `disable()`).
- Add a toggle to skip masking (or to avoid cloning) and re-measure memory/TPS to isolate the clone cost.
- Restrict `_captured_all` growth (e.g., keep only per-layer captures and drop after `resolve_activation`) to see the impact on peak memory.

## On `_captured_all` — is it needed?
- Current usage: `resolve_activation` picks from `capture_pool = self._captured.get(name, []) or self._captured_all`. That means:
  - If we have per-layer entries (`_captured[name]`), `_captured_all` is only used when that list is empty.
  - In normal operation with proper scoping, per-layer captures should suffice.
- Cost: `_captured_all` keeps every saved tensor alive until `disable()`, inflating peak memory.
- Follow-up applied: `_captured_all` and its usage have been removed; capture now relies solely on per-layer `_captured[name]`, which should allow autograd to release intermediates once each layer’s backward completes.
