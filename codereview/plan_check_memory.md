# Plan: Check Ghost GradDotProd Memory Release

## What I observed in code
- `examples/torchtitan/torchtitan/train_with_ghost.py` forces `global_batch_size = local_batch_size` and concatenates train+val. With `local_batch_size=2` and `val_batch_size=4`, forward/backward uses batch=6, versus batch=2 with gradient accumulation in the non-ghost path. This alone should raise activation memory by ~3x.
- Metrics use `labels.numel()` from the *train* batch (`examples/torchtitan/torchtitan/train.py:405-408`), so throughput/MFU is undercounted when ghost adds validation tokens.
- `ghostEngines/autograd_grad_sample_dotprod.py:_NamedSavedTensorManager.unpack_hook` does `x.clone()` when masking. That duplicates saved activations during backward.
- Norm layers keep a second copy of the original activation (`_ghost_saved_activation`) while the masked clone is used for autograd. That is an extra activation copy for each norm layer.
- Saved tensors are held in `_NamedSavedTensorManager._captured` until `_cleanup_layer_state` runs. If cleanup does not run for any layer, those tensors stay alive for the remainder of backward.
- `ghostEngines/graddotprod_engine.py` caches `X_train` / `Y_train` on GPU until the next step; small compared to activations but still resident.

## Hypotheses to validate
1) Most of the 66 GiB peak is explained by the larger effective batch size (train+val) plus activation masking clones.
2) Ghost keeps some saved tensors alive longer than standard autograd, causing a monotonic climb during backward.
3) TPS/MFU drop is partly a reporting artifact (train-only token count) plus extra dotprod compute.

## Tests / instrumentation to add
### A. Per-layer backward memory trace (ghost vs baseline)
- Create `examples/torchtitan/scripts/ghost_memory_trace.py` (or a new test under `examples/torchtitan/tests/`).
- Subclass `Trainer` and `GhostTrainer` to inject hooks without modifying core logic:
  - Register `register_full_backward_hook` on each supported layer.
  - At hook entry/exit, log `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` (plus `torch.cuda.synchronize()` for accurate timing).
  - If ghost is enabled, also log `len(manager._captured.get(layer.name, []))` and the presence of `layer.activations` / `_ghost_saved_activation`.
  - Save trace to JSON/CSV so we can plot memory vs layer index.
- Run three cases:
  1) Baseline: `--ghost.enable=false` (current training config).
  2) Ghost: `--ghost.enable=true` (current ghost config).
  3) Control: `--ghost.enable=false --training.local_batch_size=6 --training.global_batch_size=6` to match the combined batch size.
- Success criteria:
  - Memory should rise then drop as backward progresses; if it only rises or saved tensors persist late in backward, we have evidence of delayed release.

### B. Step-level memory checkpoints
- In the same harness, record memory at:
  - pre-forward, post-forward, post-loss, post-backward, post-optimizer.
- Use `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.max_memory_allocated()` per segment.
- Compare ghost vs baseline vs control to separate batch-size effects from ghost-only overhead.

### C. Memory history snapshots (optional, no code changes)
- Use built-in profiling: set `--profiling.enable_memory_snapshot=true --profiling.profile_freq=1`.
- Run 1-2 steps with and without ghost.
- Inspect `outputs/memory_snapshot/.../rank*_memory_snapshot.pickle` for allocation/free events during backward.

### D. Throughput normalization
- Recompute TPS/MFU using `train_tokens + val_tokens` for ghost runs to quantify reporting bias.

## Expected outcomes
- If control run (batch=6, ghost off) matches ghost peak memory, batch size is the primary driver.
- If ghost still higher, the clone+masking path and retained activations are additional overhead.
- If per-layer traces show `_captured` entries lingering after a layer finishes backward, cleanup needs tightening.

## Next steps after tests
- If memory release is delayed: tighten `_cleanup_layer_state` and scope handling in `ghostEngines/autograd_grad_sample_dotprod.py`.
- If clone is dominant: prototype a no-clone or partial-mask path in `unpack_hook` and re-measure.
