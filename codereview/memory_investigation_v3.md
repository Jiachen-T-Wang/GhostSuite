# Memory/MFU Investigation (v3) — Ghost GradDotProd

## State after removing `_captured_all`
- Per-layer captures are still retained until each layer’s cleanup, but peak memory remains ~66 GiB and MFU ~3–4%.
- The removal did not materially reduce memory, implying other dominant factors.

## Likely dominant factors now
1) **Activation masking clones saved tensors**
   - `_NamedSavedTensorManager.unpack_hook` returns `masked = x.clone()` for every matched activation, then zeros the val slice and rescales train.
   - The original saved activation must stay alive for dot-product computation (and norm grad_input fix), so the clone effectively doubles activation footprint for every masked layer.
   - For attention/FFN hidden states, this is a major multiplier; cloning also adds bandwidth, hurting MFU.

2) **Duplicated activations for norm layers**
   - Masked activation is used for autograd parameter grads; the original activation is retained to recompute `grad_input` (and LayerNorm train-only grads) in the full backward hook.
   - Net effect: two copies (masked + original) per norm layer during backward.

3) **Extra compute per layer**
   - Dot-product paths (FP32 accum) on every supported layer, plus RMSNorm/LayerNorm grad_input recomputation and LayerNorm train-grad override. These add matmuls/reductions beyond the model’s backward, reducing TPS/MFU.

4) **Larger effective batch**
   - Train+val concatenation increases baseline activation/backprop size versus `--ghost.enable=false`.

## Other contributors
- `_mask_embedding_grad_output` clones the embedding grad_output (added copy per step).
- Activations/backprops cached on layers until cleanup (transient but adds to lifetime).

## What to verify next (to pinpoint the main bottleneck)
- Instrument peak memory before/after masking by adding a flag to bypass masking (return `x` directly) to confirm clone impact.
- Track allocation counts for clones in `unpack_hook` (e.g., with autograd profiler).
- Measure memory if norm grad_input fix is disabled (temporarily skip norm backward hook) to see the cost of retaining original activations alongside masked ones.

## Candidate mitigations (not implemented)
- **Eliminate clones in `unpack_hook`:**
  - Use an in-place view/update on a detachable buffer (e.g., allocate a reusable mask tensor and apply `masked = x.mul(mask)` without cloning, or use custom autograd to avoid duplicating the saved tensor).
  - Consider a custom Function for masking that shares storage or performs selective scaling/zeroing without a full clone.
- **Reduce masking scope:**
  - Mask only layers where parameter grads must be train-only; consider leaving some layers unmasked to cut duplication.
  - Skip masking when val_batch_size is zero or when dot products are not needed for a layer.
- **Norm layers:**
  - Explore recomputing needed stats on the masked activation to avoid keeping the original around (or stash a lighter-weight summary instead of full activation).
- **Profiling guardrails:**
  - Add a debug flag to disable masking and/or norm correction to isolate their memory/MFU impact quickly.
