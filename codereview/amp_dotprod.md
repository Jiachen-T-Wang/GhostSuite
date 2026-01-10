# AMP gradient dot-product plan

## Context
- Observed dtype mismatch in `_prepare_sample_grad_or_dotprod` for `tok_embeddings`: activations are `int64` indices, backprops are `float32`, and the hook promotes activations to `float32`. For embeddings this promotion is unnecessary and wastes bandwidth; accumulation should still happen in `float32`.
- Review target: ensure each dot-product implementation uses an appropriate compute dtype (fp16/bf16 for elementwise work is fine) while accumulating reductions in `float32`, and avoid forcing integer activations to fp32.

## Goals
- Keep embedding token indices as integers in the dot-product path; only promote gradients to the right compute dtype.
- Make accumulation (sums/einsums/matmuls that produce scalars or reduced tensors) happen in fp32 across all supported layers.
- Define a consistent dtype policy for `_prepare_sample_grad_or_dotprod` so layer functions receive the right inputs without ad-hoc promotions.
- Add lightweight validation to confirm AMP behavior matches full-precision baselines for dot-products.

## Current behavior to adjust
- Shared hook: uses `torch.promote_types` on activations/backprops; for embeddings this upcasts integer indices. Dot-product functions are unaware of the source dtypes.
- Linear: immediately casts A/B to bf16; ghost/materialized paths rely on bf16 matmul/einsum without explicit fp32 accumulation.
- Embedding: B is cast to bf16; products are cast to fp32 before reduction (good), but inputs arrive as promoted fp32 indices when coming from the hook.
- LayerNorm: A/B cast to bf16; normalization and reductions rely on bf16 unless explicitly cast to float during dot-product (partial coverage).
- RMSNorm: already converts to float for rms and reductions.
- Conv1D/Conv2d: cast A/B to bf16; sums/einsums mostly operate in bf16, with some late `.float()` casts on products; accumulation dtype is unclear/inconsistent.

## Plan
1) **Common dtype policy in hook**
   - Add a small helper to choose `compute_dtype` per layer: keep integer activations as-is for embeddings; otherwise align activations/backprops to a shared compute dtype (weight/backprop or autocast target).
   - Pass both raw activations and `compute_dtype` into layer dot-product functions so they can control casts locally.
2) **Embedding path**
   - Ensure activations remain integer; only cast B to the desired compute dtype (likely weight/backprop dtype).
   - Perform reductions in fp32 (explicit `.float()` on accumulation steps) while keeping intermediate lookups minimal.
3) **Linear**
   - Avoid unconditional bf16 cast; choose compute dtype based on weight/backprop/autocast, but force accumulations (matmul/einsum) to fp32.
   - Keep `grad_dot_prod` dtype consistent with existing expectations (likely fp32 tensor per sample).
4) **LayerNorm**
   - Run normalization in fp32 (or cast inputs to fp32 for the normalized result), then optionally downcast for storage but keep dot-product reductions in fp32.
   - Apply the same policy to bias path.
5) **RMSNorm**
   - Already largely fp32; ensure B is coerced to the compute dtype and accumulations stay in fp32 for consistency with other layers.
6) **Conv1D/Conv2d**
   - Choose compute dtype (likely bf16 when enabled) for elementwise work, but cast inputs to fp32 for sums/matmuls/einsums used in dot-product accumulation.
   - Ensure bias dot-product paths also accumulate in fp32.
7) **Testing/validation**
   - Add small unit/integration checks under autocast for each layer type: confirm embedding inputs stay integer, and dot-product outputs match fp32 reference within tolerance.
   - Include a conv case (Conv1D or Conv2d) to verify accumulation dtype and absence of overflow/under-accumulation in bf16.
   - Consider an opt-in debug log or assert to report the dtype used for accumulation per layer during tests.
