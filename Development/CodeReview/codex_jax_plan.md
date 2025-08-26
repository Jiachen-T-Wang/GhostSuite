JAX Migration Plan: ghostEngines Support

Goals
- Parity: Reproduce current GradDotProd functionality for GPT2-Small pretraining on Pile using JAX.
- Ergonomics: Keep a manager-style API mirroring `GhostEngineManager` so training loops remain clean.
- Performance: Provide a clear path from a correct baseline (per-example grads) to the efficient “ghost” dot-product formulas.
- Explicit config: No silent fallbacks. Users must select `--framework jax` and `--method` explicitly; error if unsupported.

Libraries & Runtime
- Core: `jax`, `jaxlib`, `flax` (models), `optax` (optim), `orbax-checkpoint` (ckpt, optional).
- Dtypes: default `bfloat16` on TPU/A100, fall back to `float32` by explicit flag (no silent fallback).
- Devices: start single-host, single-device; plan pjit/sharding later.

High-Level Architecture
- `ghostEngines_jax/`
  - `engine_manager_jax.py`: Manager selecting engine by `config.method`.
  - `graddotprod_engine_jax.py`: Computes per-layer gradient dot-products and accumulates train grads.
  - `supported_layers_jax.py`: Layer-specific math for Dense/Embed/LayerNorm/Conv1D (ghost formulas).
  - `intercepts.py`: Utilities to capture activations and derive per-layer output cotangents.
- Training loop (JAX):
  - `model_setup_jax.py`: Build Flax/Optax model, optimizer, loss fn.
  - `training_loop_jax.py`: Mirrors current loop; calls JAX ghost engine hooks.
  - CLI: add `--framework jax` to route to JAX stack; error if incompatible flags are used.

Activation And Derivative Extraction in JAX
- We need, per supported layer i:
  - Activations `A_i`: the tensor entering the layer.
  - Backprops `B_i`: d(loss)/d(y_i), where `y_i` is the layer output.
- Capture activations (Flax):
  - Use Flax’s `capture_intermediates` to record outputs of selected modules and `mutable=["intermediates"]` to retrieve them.
  - Example:
    - `y, collections = model.apply(params, x, train=True, mutable=["intermediates"],
      capture_intermediates=lambda m, _, __: isinstance(m, (nn.Dense, nn.LayerNorm, nn.Embed)))`
    - `collections["intermediates"][<ModulePath>]["__call__"][0]` yields `y_i` per layer; the corresponding activation is the input to that module. For modules where input is needed (e.g., Dense), define a small wrapper module that `sow`s both input and output so we can recover `A_i` robustly.
- Compute per-layer output derivatives (cotangents): three viable approaches; we will implement A then B for efficiency.
  - A) Tail-function VJP (initial, explicit and exact):
    - For each layer i, define a tail function `tail_i(params, h_i) -> logits` that applies layers i+1..end.
    - Then `B_i = jax.grad(lambda h: loss(tail_i(params, h), targets))(h_i)`.
    - Generate `tail_i` programmatically by composing the model as an explicit list/scan of layers. This computes dL/d(y_i) exactly without Jacobian materialization.
  - B) Reverse scan of VJP (single backward pass):
    - Structure Transformer blocks in a `lax.scan`. Use `jax.linearize`/`jax.linear_transpose` to build a reverse pass that returns the cotangent at each scan step, yielding all `B_i` in one sweep.
    - Benefit: single pass; required to scale to long depth.
  - C) Baseline per-example parameter gradients (for debugging only — guarded by config):
    - `perex_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))(params, xb, yb)`; dot with validation grads. Useful to cross-check A/B implementations on tiny shapes. Never used implicitly.

Layer-Wise Ghost Dot-Product Formulas (JAX)
- Dense (Linear): with `A ∈ [B,T,d]`, `B ∈ [B,T,p]`, `val_bs = V`, `train_bs = B − V`.
  - Ghost dot product per training example:
    - Flatten tokens: `A_tr=[(B−V)·T,d]`, `A_val=[V·T,d]`; `B_tr=[(B−V)·T,p]`, `B_val=[V·T,p]`.
    - `a_dot = A_tr @ A_val.T`, `b_dot = B_tr @ B_val.T`, `token_contrib = sum((a_dot * b_dot), axis=1)`.
    - Reshape `token_contrib` to `[B−V, T]` then sum over tokens → scalar per train example.
- Embedding: index-add to accumulate validation gradient, then gather for train indices and elementwise multiply with `B_train`; sum over dims.
- LayerNorm: use normalized activations for weight; bias uses only `B`.
- Conv1D: identical structure to Dense with conv-reshaped `A`/`B`.
- Implement these in `supported_layers_jax.py` using pure JAX ops: `jnp.dot`, `jnp.matmul`, `jax.lax.scatter_add`, etc.; match PyTorch bf16 accumulation with fp32 reduce.

JAX Ghost Engine Flow (GradDotProd)
- Attach/prepare (no autograd hooks):
  - Store `val_batch` on host/device.
  - Provide a `prepare_forward_batch(train_x, train_y)` that concatenates with val, mirroring PyTorch path.
- Forward pass (within training step):
  - Compute logits and loss on concatenated batch; simultaneously collect intermediates:
    - `logits, (intermediates, A_summaries)` where `A_summaries` holds inputs per module (via wrappers that `sow` inputs).
- Backward pass (within `engine.compute`):
  - Compute dL/d(logits) from loss.
  - Obtain `B_i` for supported layers (Approach A first, then B):
    - Tail VJP per layer to get `B_i` at layer output.
  - Compute per-layer ghost dot products from `A_i`, `B_i` and aggregate per-example scalars.
- Train gradient accumulation:
  - Compute average train gradient for parameters (if requested) using layer-specific closed forms (as in PyTorch) or using `jax.grad(loss)` when `config.ghost_grad_mode == "param"`. This is independent from dot-product computation and used to update optimizer state.
- Logging and saving:
  - Keep dot-product arrays on device; periodically transfer to CPU and save with `flax.serialization.to_bytes` or `numpy.savez`. Match current `dot_prod_log_iter_*.pt` naming with a JAX-friendly format (`.npz`), or keep `.pt` but documented change.

Training Loop (JAX) Parity
- `training_loop_jax.py` mirrors current logic:
  - Gradient accumulation across `gradient_accumulation_steps` via loop; scale loss by steps.
  - Scheduler: port `get_learning_rate` to JAX/Optax schedule; apply with `optax.inject_hyperparams` or manual lr in `optax.adamw`.
  - Evaluation path: run without ghost engine active (`engine.detach_for_evaluation()` is a no-op in JAX but keep API consistent).
- `GhostEngineManagerJax` exposes the same methods used by `Trainer`:
  - `attach_train_batch`, `prepare_forward_input`, `prepare_gradients`, `aggregate_and_log`, `clear_gradients`, `should_save_metrics`, `save_metrics`, `detach_for_evaluation`, `reattach_after_evaluation`, `cleanup`.
  - In JAX, `prepare_gradients/clear_gradients` are pure bookkeeping, not tensor mutation.

Config & CLI
- Add `--framework {torch,jax}` and `--jax_engine {graddotprod, perexample}`.
- Explicit layer support list; error on encountering unsupported layers for ghost formulas unless `--jax_engine perexample` is chosen.
- Enforce `--batch_size 2` policy in evaluation path for consistency with current guidelines.

Testing Strategy
- Unit checks (deterministic, CPU-only):
  - Dense/Embed/LayerNorm ghost dot-product kernels vs. materialized per-parameter grads on tiny shapes; assert close within tolerance.
  - Tail VJP `B_i` equals autodiff-computed dL/d(y_i) on toy MLP/TransformerBlock.
- Integration checks:
  - End-to-end step on GPT2-Small (Flax) with random data: engine returns non-empty dot products; shapes match; optimizer step completes.
- Parity test vs. PyTorch:
  - For small MLP with seeded weights, compute dot-product per train example in both frameworks; assert agreement.

Performance Notes & Roadmap
- Phase 1 (Correctness): Tail-VJP approach, single-device; param-grad via `jax.grad`.
- Phase 2 (Efficiency): Reverse-scan VJP to collect all `B_i` in one pass; bf16 compute + fp32 accumulators; prefetch to device; shard-safe.
- Phase 3 (Scale): pjit with named axes, sharded parameters and activations; aggregate dot-products across devices; streaming save.

Pitfalls & Mitigations
- Intermediates capture: `capture_intermediates` stores outputs, not inputs; wrap target modules to `sow` inputs explicitly to guarantee `A_i` availability.
- Side-effects: No autograd hooks; all capture via pure function returns. Never mutate state inside JAX transforms.
- Dtype promotion: mimic PyTorch path — cast `A`,`B` to bf16 for mats, accumulate in fp32.
- RNG/state: thread PRNGs explicitly in Flax; keep ghost engine stateless apart from logs.
- Memory: avoid storing full per-token intermediates for all layers at once; stream layerwise when possible or switch to reverse-scan.

Sketches (Illustrative Only)
- Capture intermediates and inputs:
  - Wrap module apply:
    - `class GhostDense(nn.Module):
         features: int
         @nn.compact
         def __call__(self, x):
             nn.sow('ghost','inputs', x)            # A_i
             y = nn.Dense(self.features)(x)
             nn.sow('ghost','outputs', y)           # y_i
             return y`
  - Apply with capture:
    - `logits, vars = model.apply(params, x, mutable=['ghost'])`
- Per-layer `B_i` via tail VJP:
  - `def tail(params, h, i):
         for j in range(i+1, L):
             h = block_apply(params.blocks[j], h)
         return head(params.head, h)`
  - `B_i = jax.grad(lambda h: loss(tail(params, h, i), y_true))(h_i)`
- Dense ghost dot product (bf16 compute, fp32 reduce):
  - `a_dot = (A_tr.astype(jnp.bfloat16) @ A_val.T.astype(jnp.bfloat16))
     b_dot = (B_tr.astype(jnp.bfloat16) @ B_val.T.astype(jnp.bfloat16))
     token_contrib = (a_dot * b_dot).sum(axis=1, dtype=jnp.float32)
     per_example = token_contrib.reshape((train_bs, T)).sum(axis=1)`

Deliverables & Milestones
- M1: `ghostEngines_jax` skeleton + per-example baseline + tests on toy nets.
- M2: Flax capture of `A_i`, tail-VJP `B_i`, Dense/Embed/LayerNorm ghost kernels + parity tests.
- M3: Reverse-scan VJP, Conv1D support, logging format, CLI integration, docstrings.
- M4: Sharding-ready implementation, performance sweep, examples in `Demo/`.

Open Questions
- Preferred model stack: Flax vs. Haiku. Plan assumes Flax (closer to HuggingFace Flax GPT-2).
- File format for logs: keep `.pt` (PyTorch) or switch to `.npz`/`.msgpack`? Recommendation: `.npz` + loader utility for compatibility.

