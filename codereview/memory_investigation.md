# Memory/MFU Investigation (Ghost GradDotProd)

## Summary of likely bottlenecks
The Ghost path introduces two major sources of GPU memory growth and a large drop in throughput:
1) **Pinned saved tensors for the entire backward pass** in `_NamedSavedTensorManager` prevent autograd from freeing intermediates early.
2) **Activation masking clones** in `unpack_hook` duplicate large saved activations.

Both inflate peak memory and increase bandwidth/compute pressure, which explains the large MFU drop.

## Primary memory drivers
1) **Global capture list keeps *all* saved tensors alive**
   - `ghostEngines/autograd_grad_sample_dotprod.py`, `_NamedSavedTensorManager.pack_hook` stores every saved tensor in `_captured_all` and keeps it until `disable()` is called.
   - `manager.clear_layer()` only removes per-layer entries; `_captured_all` is never pruned during backward.
   - This prevents autograd from freeing large saved tensors after each layer’s backward finishes.
   - The saved-tensor log shows large SDPA tensors (`q/k/v` and softmax stats); keeping these for the whole backward is a big memory multiplier.
   - **Impact:** baseline training can free these intermediates progressively; Ghost holds them all simultaneously → large peak memory.

2) **`unpack_hook` clones the activation tensor**
   - `ghostEngines/autograd_grad_sample_dotprod.py`, `_NamedSavedTensorManager.unpack_hook` does `masked = x.clone()` and returns the clone.
   - That creates an additional copy of each masked activation, roughly doubling saved-activation memory for those layers.
   - For large attention/FFN activations, this is significant.
   - **Impact:** higher peak memory + extra memory bandwidth (clone + writes), which also reduces throughput.

## Secondary contributors
1) **Dot-product computation cost**
   - `ghostEngines/supported_layers_grad_samplers_dotprod.py` computes per-layer dot products, often with FP32 accumulations and matmuls.
   - Even with “ghost” shortcuts, this is extra compute on every layer, reducing TPS/MFU.
   - The MFU drop (27% → 3–4%) is consistent with extra matmuls + memory pressure.

2) **Norm-layer corrections add extra compute**
   - `ghostEngines/autograd_grad_sample_dotprod.py`: RMSNorm/LayerNorm `grad_input` recomputation and LayerNorm train-grad override add per-layer overhead.
   - Smaller than the dot-product cost, but non-zero.

3) **Activation references on the layer**
   - `layer.activations` and `_ghost_saved_activation` are kept until the layer’s backward hook runs.
   - This is transient but still adds pressure if the backward order delays cleanup.

## Why memory jumps so much vs `--ghost.enable=false`
Without Ghost:
- Autograd saves what it needs, then frees intermediates layer-by-layer during backward.

With Ghost:
- `_captured_all` holds references to all saved tensors for the entire backward.
- `unpack_hook` clones masked tensors, so many saved activations exist in *two* copies.
- Combined with dot-product compute, this is enough to raise memory from ~25GiB to ~66GiB and reduce throughput.

## What to check next (if you want to mitigate)
Not requested to change code here, but the most direct fixes to reduce memory/MFU impact are:
- **Stop retaining `_captured_all`** or make it bounded; rely on per-layer `_captured[name]` whenever possible.
- **Avoid cloning in `unpack_hook`** (e.g., use views + in-place masking on a pre-allocated buffer or a custom autograd function).
- **Skip masking for tensors not strictly needed** (reduce the number of masked tensors).
- **Add an option to disable activation masking** for profiling to isolate its cost.

These align with the two primary bottlenecks above.
