# Plan: Ghost dot-product integration for torchtitan (llama3-130M)

## Context
- Goal: run `./tests/torchtitan/run_train_with_ghost.sh` to train llama3-130M while computing per-sample train-vs-val gradient dot products using `GradDotProdEngine` on a validation batch from `pile_test`.
- Current state: `GradDotProdEngine` works for LayerNorm-based Llama-like model in `tests/benchmark_ghost_dotprod.py`. TorchTitan trainer has no ghost support and llama3 model uses `nn.RMSNorm` (not yet supported by ghost engine). Default llama3-130m config uses gradient accumulation (global_batch_size 120, local 12), compile is enabled, and validation is disabled.

## Plan
1) **Add ghost config + runner wiring**
   - Extend `JobConfig` with a `ghost` section (enable flag, val_batch_size, val_dataset/path defaults to `pile_test`, save_dir/save_interval, log_grad_norms, use_dummy_bias, maybe log_train_batch toggle). Defaults keep ghost off to avoid impacting existing runs.
   - Provide a small ghost-specific TOML overlay or rely on CLI overrides; set sensible defaults for single-GPU (e.g., disable compile when ghost is on, optionally cap global_batch_size to local_batch_size to avoid grad accumulation until we support it).

2) **Support llama3 layers in ghost engine**
   - Add dot-product and training-gradient samplers for `nn.RMSNorm` (weight-only in current model; handle optional bias defensively) and register it in `_supported_layers_dotprod`.
   - Mirror LayerNorm logic: compute per-sample train grads, aggregated val grads, dot products, and optional grad norms; ensure dtype/shape handling matches existing hooks.
   - Sanity-check other llama3 modules (Linear/Embedding already supported).

3) **Ghost helper to manage val batch and engine**
   - New helper (e.g., `torchtitan/ghost/dotprod_helper.py`) to load one validation batch via `build_text_validation_dataloader` (dataset/seq_len from config), move to training device, and track actual `val_batch_size`.
   - Instantiate `GradDotProdEngine` with the model (single part), optimizer container, val_batch_size, use_dummy_bias, log_grad_norms, and dot-prod save path under `job.dump_folder/ghost_dotprods` (or config override). Optionally save the val batch for reuse.
   - Guardrails: require no PP/TP/CP for v1; optionally assert world_size==1 and raise/warn otherwise; adjust loss reduction expectations if needed.

4) **Integrate helper into training loop**
   - Add a `GhostTrainer` (new module, e.g., `torchtitan/train_with_ghost.py`) that subclasses `Trainer`.
     * Before `super().__init__`, tweak config when ghost is enabled (e.g., set `training.global_batch_size = training.local_batch_size` to avoid grad accumulation, disable compile, switch validation dataset to ghost choice if not provided).
     * After base init, build ghost helper and keep cached val batch.
   - Override `train_step`/`forward_backward_step` for ghost mode:
     * For each microbatch, attach train batch metadata to the engine, combine train inputs/labels with cached val batch along batch dim, and run forward/backward on the combined batch.
     * Call `engine.prepare_gradients()` before optimizer step to move accumulated train grads into `.grad`, then run optimizer/scheduler.
     * After step, call `aggregate_and_log` (and `clear_gradients`) and optionally `save_dot_product_log` on a configurable interval; ensure tensors are moved off GPU to avoid growth.
   - Ensure cleanup/detach in `close()` and temporarily detach hooks around validation runs if those are enabled.

5) **CLI runner**
   - Add `run_train_with_ghost.sh` mirroring `run_train.sh` but setting `TRAIN_FILE=torchtitan.train_with_ghost`, exporting `PYTHONPATH` to repo root for `ghostEngines`, pointing to llama3_130m config, and applying recommended overrides (ghost.enable, ghost.val_batch_size default, validation.dataset=`pile_test`, compile disabled, steps/local/global batch sizes for single-GPU sanity).
   - Document expected outputs: dot-product logs saved under dump folder; how to adjust val batch size/save interval via CLI.

6) **Verification/docs**
   - Optional quick sanity script/test: 1 step on CPU or tiny seq_len to confirm dot_product_log is populated and shapes match train batch size.
   - Add short note (README or comments) about ghost run constraints (single GPU, no PP/TP, grad accumulation currently unsupported) and how to interpret log entries.

## Open questions
- What val batch size should we default to for llama3-130M, and should we shrink the train batch to keep total tokens constant?
- Is it acceptable to force `gradient_accumulation_steps=1` for ghost runs (by setting global_batch_size = local_batch_size), or should we invest in accumulating dot products across microbatches now?
- Should we disable `torch.compile` by default when ghost is enabled?
- Do we want to save full train/label tensors in `dot_product_log`, or only dot products/grad norms to reduce file size? Preferred save interval?
- Should ghost-enabled runs also execute the usual validation loop, or skip to save time/resources?
