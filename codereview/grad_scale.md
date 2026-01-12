# Grad Scaling Review (Dot-Products + Train Grads)

## Findings from code review
- `ghostEngines/autograd_grad_sample_dotprod.py:_compute_dotprod_from_backprops` uses raw `grad_output` (already scaled by loss reduction). For `reduction="mean"`, `grad_output` is scaled by `1/total_bs`, so dot products are scaled by `1/total_bs^2`.
- The naive correctness check in `tests/benchmark_ghost_dotprod.py:376-409` explicitly multiplies each per-sample grad by `1/total_batch_size`, so current ghost dot products match that definition (mean over train+val).
- Training gradients are **not** using the same scaling as dot products:
  - For most layers, `_NamedSavedTensorManager.unpack_hook` scales masked activations by `total_bs/train_bs` (mean) or `1/train_bs` (sum) so that autograd produces **train-only mean** gradients.
  - For embeddings, `_mask_embedding_grad_output` applies the same scaling to `grad_output`.
  - For LayerNorm, `norm_backward_hook` rescales `grad_output` by `total_bs` under mean reduction before computing train-only grads.
- Result: dot products reflect gradients of the **combined mean loss**, while parameter updates use gradients of the **train-only mean loss**. This mismatch is likely the “under-scaled backprops” note in `codereview/v0.33_notes.md`.
- The weight-gradient masking path appears correctly scaled **for the intended design** (“train gradients always averaged over the training portion”), but it should be verified for all supported layers and both reduction modes.

## Intended dot-product definition
- **Per-sample train gradient** vs **mean validation gradient**.
  - Target: `dot_i = <g_train_i, (1/val_bs) * sum_{j in val} g_val_j>`.
  - Under `reduction="mean"`, raw `grad_output` is scaled by `1/total`, so dot products must be rescaled.
- Loss reduction is “mean”, but in LM it is **mean over tokens**, not just batch. If train/val seq_len differ, scaling should be by token count, not batch size.

## Revision plan (no code changes yet)
1) **Document semantics**
   - Update README and `GradDotProdEngine` docstring to state:
     - Dot product is between **per-sample train gradients** and the **mean validation gradient**.

2) **Update correctness references**
   - Extend `tests/benchmark_ghost_dotprod.py` to compute a naive reference using:
     - `g_train_i` from a single-sample loss (no batch scaling).
     - `g_val_mean = (1/val_bs) * sum_j g_val_j`.
   - Keep the old reference (combined mean) behind a flag for backward comparison.

3) **Rescale dot-product backprops**
   - In `_compute_dotprod_from_backprops`, apply explicit scaling for `reduction="mean"`:
     - `train_scale = total_bs` to recover per-sample train gradients.
     - `val_scale = total_bs / val_bs` to recover mean validation gradient.
   - For `reduction="sum"`:
     - `train_scale = 1.0` (already per-sample sum).
     - `val_scale = 1.0 / val_bs` (still want mean validation gradient).
   - Implement scaling either by:
     - splitting `backprops` into train/val and scaling in-place before calling `_supported_layers_dotprod`, or
     - adding scale parameters to dotprod helpers and applying there.
   - Update grad-norm logging to mirror the same scaling.

4) **Verify training-gradient scaling**
   - Add tests comparing ghost `param.grad` (after `prepare_gradients`) to baseline **train-only** gradients for:
     - `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.RMSNorm`.
     - `reduction="mean"` and `reduction="sum"`.
   - Specifically check layers **with bias** (Linear/Conv1D/Conv2d) since bias grads depend only on `grad_output` and are not masked by activation scaling.
   - If bias mismatch is confirmed, plan to:
     - mask `grad_output` for biased layers in the output hook, or
     - compute bias `train_grad` explicitly (similar to `_compute_train_grad_bias`).

5) **Token-count aware scaling (LM)**
   - If train/val seq_len can differ, use token counts (train_tokens/val_tokens) for scaling rather than batch size.
   - Add a test with differing seq_len to catch regressions.



## Observation: inconsistency between ghost and non-ghost run

### Ghost run for torchtitan
```
CONFIG_FILE="/scratch/gpfs/PMITTAL/tianhao/GhostSuite/tests/torchtitan/torchtitan/models/llama3/train_configs/llama3_130m_ghost.toml" ./tests/torchtitan/run_train_with_ghost.sh --training.steps=10
```
The above produces the following training log:
```
[rank0]:[titan] 2026-01-11 20:13:44,754 - root - INFO - step:  1  loss: 12.2521  grad_norm:  0.6113  memory: 43.52GiB(31.13%)  tps: 6,922  tflops: 13.35  mfu: 1.35%
[rank0]:[titan] 2026-01-11 20:13:44,755 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[rank0]:[titan] 2026-01-11 20:13:44,956 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=1, cp=1, tp=1, ep=1, etp=1
[rank0]:[titan] 2026-01-11 20:13:44,961 - root - INFO - Successfully created meshes with active dimensions: []
[rank0]:[titan] 2026-01-11 20:13:45,284 - root - INFO - step:  2  loss: 12.2273  grad_norm:  0.5423  memory: 47.71GiB(34.12%)  tps: 31,075  tflops: 59.95  mfu: 6.06%
[rank0]:[titan] 2026-01-11 20:13:45,604 - root - INFO - step:  3  loss: 12.1733  grad_norm:  0.6184  memory: 47.71GiB(34.12%)  tps: 51,352  tflops: 99.08  mfu: 10.02%
[rank0]:[titan] 2026-01-11 20:13:45,922 - root - INFO - step:  4  loss: 12.0984  grad_norm:  0.7449  memory: 47.71GiB(34.12%)  tps: 51,652  tflops: 99.65  mfu: 10.08%
[rank0]:[titan] 2026-01-11 20:13:46,243 - root - INFO - step:  5  loss: 11.9421  grad_norm:  1.0203  memory: 47.71GiB(34.12%)  tps: 51,347  tflops: 99.06  mfu: 10.02%
[rank0]:[titan] 2026-01-11 20:13:46,562 - root - INFO - step:  6  loss: 11.6608  grad_norm:  1.7049  memory: 47.71GiB(34.12%)  tps: 51,566  tflops: 99.49  mfu: 10.06%
[rank0]:[titan] 2026-01-11 20:13:46,884 - root - INFO - step:  7  loss: 11.1394  grad_norm:  1.8498  memory: 47.71GiB(34.12%)  tps: 51,076  tflops: 98.54  mfu: 9.96%
[rank0]:[titan] 2026-01-11 20:13:47,203 - root - INFO - step:  8  loss: 10.8308  grad_norm:  1.7437  memory: 47.71GiB(34.12%)  tps: 51,552  tflops: 99.46  mfu: 10.06%
[rank0]:[titan] 2026-01-11 20:13:47,524 - root - INFO - step:  9  loss: 10.6854  grad_norm:  1.5282  memory: 47.71GiB(34.12%)  tps: 51,301  tflops: 98.98  mfu: 10.01%
[rank0]:[titan] 2026-01-11 20:13:47,843 - root - INFO - step: 10  loss: 10.5298  grad_norm:  1.5870  memory: 47.71GiB(34.12%)  tps: 51,534  tflops: 99.43  mfu: 10.05%
```

### Analysis (no code changes)
- The ghost trainer computes **loss on the concatenated train+val batch** (`GhostTrainer.forward_backward_step`), so the logged loss is a weighted average of train and val losses:
  - `loss_logged = (train_bs/total_bs)*loss_train + (val_bs/total_bs)*loss_val`.
  - This is expected to differ from a train-only baseline unless `loss_val == loss_train`.
- Gradients are masked to be train-only, so the logged loss is **not** the objective being optimized. If you want identical loss numbers, log **train-only loss** in ghost runs (separate metric).
- If comparing ghost vs non-ghost with the same config, note that ghost forces `global_batch_size = local_batch_size` (no grad accumulation). Baseline runs with the original global batch will not be identical.
- Potential gradient mismatches if any biased layers are present:
  - Bias gradients depend only on `grad_output` and are not masked by activation scaling, so val contributions can leak into bias grads unless masked explicitly.
  - Llama3 typically uses bias-free linear layers, but other models may not.

### Standard run for torchtitan
```
CONFIG_FILE="/scratch/gpfs/PMITTAL/tianhao/GhostSuite/tests/torchtitan/torchtitan/models/llama3/train_configs/llama3_130m_ghost.toml" ./tests/torchtitan/run_train_with_ghost.sh --training.steps=10 --ghost.enable=false
```
The above produces the following training log:
```
[rank0]:[titan] 2026-01-11 20:13:44,754 - root - INFO - step:  1  loss: 12.2521  grad_norm:  0.6113  memory: 43.52GiB(31.13%)  tps: 6,922  tflops: 13.35  mfu: 1.35%
[rank0]:[titan] 2026-01-11 20:13:44,755 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[rank0]:[titan] 2026-01-11 20:13:44,956 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=1, cp=1, tp=1, ep=1, etp=1
[rank0]:[titan] 2026-01-11 20:13:44,961 - root - INFO - Successfully created meshes with active dimensions: []
[rank0]:[titan] 2026-01-11 20:13:45,284 - root - INFO - step:  2  loss: 12.2273  grad_norm:  0.5423  memory: 47.71GiB(34.12%)  tps: 31,075  tflops: 59.95  mfu: 6.06%
[rank0]:[titan] 2026-01-11 20:13:45,604 - root - INFO - step:  3  loss: 12.1733  grad_norm:  0.6184  memory: 47.71GiB(34.12%)  tps: 51,352  tflops: 99.08  mfu: 10.02%
[rank0]:[titan] 2026-01-11 20:13:45,922 - root - INFO - step:  4  loss: 12.0984  grad_norm:  0.7449  memory: 47.71GiB(34.12%)  tps: 51,652  tflops: 99.65  mfu: 10.08%
[rank0]:[titan] 2026-01-11 20:13:46,243 - root - INFO - step:  5  loss: 11.9421  grad_norm:  1.0203  memory: 47.71GiB(34.12%)  tps: 51,347  tflops: 99.06  mfu: 10.02%
[rank0]:[titan] 2026-01-11 20:13:46,562 - root - INFO - step:  6  loss: 11.6608  grad_norm:  1.7049  memory: 47.71GiB(34.12%)  tps: 51,566  tflops: 99.49  mfu: 10.06%
[rank0]:[titan] 2026-01-11 20:13:46,884 - root - INFO - step:  7  loss: 11.1394  grad_norm:  1.8498  memory: 47.71GiB(34.12%)  tps: 51,076  tflops: 98.54  mfu: 9.96%
[rank0]:[titan] 2026-01-11 20:13:47,203 - root - INFO - step:  8  loss: 10.8308  grad_norm:  1.7437  memory: 47.71GiB(34.12%)  tps: 51,552  tflops: 99.46  mfu: 10.06%
[rank0]:[titan] 2026-01-11 20:13:47,524 - root - INFO - step:  9  loss: 10.6854  grad_norm:  1.5282  memory: 47.71GiB(34.12%)  tps: 51,301  tflops: 98.98  mfu: 10.01%
[rank0]:[titan] 2026-01-11 20:13:47,843 - root - INFO - step: 10  loss: 10.5298  grad_norm:  1.5870  memory: 47.71GiB(34.12%)  tps: 51,534  tflops: 99.43  mfu: 10.05%
```


### Ghost run for torchtitan (after revision)
```
[rank0]:[titan] 2026-01-11 20:52:15,435 - root - INFO - step:  1  loss: 12.2435  grad_norm:  0.6113  memory: 43.52GiB(31.13%)  tps: 6,577  tflops: 12.69  mfu: 1.28%
[rank0]:[titan] 2026-01-11 20:52:15,436 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[rank0]:[titan] 2026-01-11 20:52:15,664 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=1, cp=1, tp=1, ep=1, etp=1
[rank0]:[titan] 2026-01-11 20:52:15,670 - root - INFO - Successfully created meshes with active dimensions: []
[rank0]:[titan] 2026-01-11 20:52:15,997 - root - INFO - step:  2  loss: 12.2122  grad_norm:  0.5423  memory: 47.71GiB(34.13%)  tps: 29,243  tflops: 56.42  mfu: 5.70%
[rank0]:[titan] 2026-01-11 20:52:16,322 - root - INFO - step:  3  loss: 12.1410  grad_norm:  0.6184  memory: 47.71GiB(34.13%)  tps: 50,603  tflops: 97.63  mfu: 9.87%
[rank0]:[titan] 2026-01-11 20:52:16,646 - root - INFO - step:  4  loss: 12.0513  grad_norm:  0.7449  memory: 47.71GiB(34.13%)  tps: 50,688  tflops: 97.79  mfu: 9.89%
[rank0]:[titan] 2026-01-11 20:52:16,972 - root - INFO - step:  5  loss: 11.8456  grad_norm:  1.0203  memory: 47.71GiB(34.13%)  tps: 50,411  tflops: 97.26  mfu: 9.83%
[rank0]:[titan] 2026-01-11 20:52:17,297 - root - INFO - step:  6  loss: 11.4986  grad_norm:  1.7050  memory: 47.71GiB(34.13%)  tps: 50,554  tflops: 97.54  mfu: 9.86%
[rank0]:[titan] 2026-01-11 20:52:17,622 - root - INFO - step:  7  loss: 10.8611  grad_norm:  1.8499  memory: 47.71GiB(34.13%)  tps: 50,578  tflops: 97.58  mfu: 9.87%
[rank0]:[titan] 2026-01-11 20:52:17,946 - root - INFO - step:  8  loss: 10.5487  grad_norm:  1.7437  memory: 47.71GiB(34.13%)  tps: 50,711  tflops: 97.84  mfu: 9.89%
[rank0]:[titan] 2026-01-11 20:52:18,272 - root - INFO - step:  9  loss: 10.4038  grad_norm:  1.5282  memory: 47.71GiB(34.13%)  tps: 50,357  tflops: 97.16  mfu: 9.82%
[rank0]:[titan] 2026-01-11 20:52:18,596 - root - INFO - step: 10  loss: 10.2069  grad_norm:  1.5869  memory: 47.71GiB(34.13%)  tps: 50,627  tflops: 97.68  mfu: 9.88%
```

### Additional analysis (after train-only logging)
- The ghost run still diverges from the non-ghost log because **the runs are not aligned in batch composition or update schedule**:
  - Ghost forces `global_batch_size = local_batch_size` (no accumulation).
  - Non-ghost keeps `global_batch_size=12`, so each logged step is an average over 6 microbatches.
  - Even if per-sample losses are identical, a single-microbatch loss will not match a 6-microbatch average in general.
- The train-only loss in ghost is computed from the **combined forward** (`pred[:train_bs]`), which should equal a train-only forward for per-sample models. Any remaining mismatch can come from:
  - **Kernel nondeterminism** (e.g., fused attention / norm kernels) that depends on batch size.
  - **RNG state differences** across runs (dataset iteration, CUDA kernels), since train and validation dataloaders advance independently.
- If you need step-by-step equality, the comparison must be done with:
  - identical batches per step (same seed + same dataloader schedule),
  - identical accumulation schedule (either both use global_batch_size=2 or both use accumulation),
  - deterministic kernels enabled (if possible).







## Observation: significant slow down on `time_metrics/end_to_end(s)`

### Standard run for torchtitan

```
CONFIG_FILE="/scratch/gpfs/PMITTAL/tianhao/GhostSuite/tests/torchtitan/torchtitan/models/llama3/train_configs/llama3_130m_ghost.toml" ./tests/torchtitan/run_train_with_ghost.sh --training.steps=10 --ghost.enable=false --training.local_batch_size=4 --training.global_batch_size=4
```
The above produces the following training log:
```
[rank0]:[titan] 2026-01-11 21:22:12,680 - root - INFO - step:  1  loss: 12.2326  grad_norm:  0.5244  memory: 43.52GiB(31.13%)  tps: 8,080  tflops: 15.59  mfu: 1.58%
[rank0]:[titan] 2026-01-11 21:22:12,680 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[rank0]:[titan] 2026-01-11 21:22:13,002 - root - INFO - step:  2  loss: 12.2097  grad_norm:  0.4884  memory: 45.30GiB(32.41%)  tps: 51,050  tflops: 98.49  mfu: 9.96%
[rank0]:[titan] 2026-01-11 21:22:13,124 - root - INFO - step:  3  loss: 12.1155  grad_norm:  0.5911  memory: 45.30GiB(32.41%)  tps: 135,808  tflops: 262.02  mfu: 26.49%
[rank0]:[titan] 2026-01-11 21:22:13,246 - root - INFO - step:  4  loss: 12.0053  grad_norm:  0.7386  memory: 45.30GiB(32.41%)  tps: 135,952  tflops: 262.29  mfu: 26.52%
[rank0]:[titan] 2026-01-11 21:22:13,367 - root - INFO - step:  5  loss: 11.7173  grad_norm:  1.2430  memory: 45.30GiB(32.41%)  tps: 136,051  tflops: 262.48  mfu: 26.54%
[rank0]:[titan] 2026-01-11 21:22:13,489 - root - INFO - step:  6  loss: 11.2483  grad_norm:  1.9420  memory: 45.30GiB(32.41%)  tps: 135,923  tflops: 262.24  mfu: 26.52%
[rank0]:[titan] 2026-01-11 21:22:13,611 - root - INFO - step:  7  loss: 10.8102  grad_norm:  1.6551  memory: 45.30GiB(32.41%)  tps: 135,646  tflops: 261.70  mfu: 26.46%
[rank0]:[titan] 2026-01-11 21:22:13,733 - root - INFO - step:  8  loss: 10.6356  grad_norm:  1.4500  memory: 45.30GiB(32.41%)  tps: 135,731  tflops: 261.87  mfu: 26.48%
[rank0]:[titan] 2026-01-11 21:22:13,854 - root - INFO - step:  9  loss: 10.4951  grad_norm:  1.4273  memory: 45.30GiB(32.41%)  tps: 135,642  tflops: 261.70  mfu: 26.46%
[rank0]:[titan] 2026-01-11 21:22:13,975 - root - INFO - step: 10  loss: 10.1709  grad_norm:  1.5206  memory: 45.30GiB(32.41%)  tps: 136,682  tflops: 263.70  mfu: 26.66%
```

### Ghost run for torchtitan

```
CONFIG_FILE="/scratch/gpfs/PMITTAL/tianhao/GhostSuite/tests/torchtitan/torchtitan/models/llama3/train_configs/llama3_130m_ghost.toml" ./tests/torchtitan/run_train_with_ghost.sh --training.steps=10 --training.local_batch_size=2 --training.global_batch_size=2
```
The above produces the following training log:
```
[rank0]:[titan] 2026-01-11 21:23:39,159 - root - INFO - step:  1  loss: 12.2521  grad_norm:  0.6113  memory: 43.52GiB(31.13%)  tps: 7,567  tflops: 14.60  mfu: 1.48%
[rank0]:[titan] 2026-01-11 21:23:39,160 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[rank0]:[titan] 2026-01-11 21:23:39,358 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=1, cp=1, tp=1, ep=1, etp=1
[rank0]:[titan] 2026-01-11 21:23:39,363 - root - INFO - Successfully created meshes with active dimensions: []
[rank0]:[titan] 2026-01-11 21:23:39,684 - root - INFO - step:  2  loss: 12.2273  grad_norm:  0.5423  memory: 47.71GiB(34.12%)  tps: 31,339  tflops: 60.46  mfu: 6.11%
[rank0]:[titan] 2026-01-11 21:23:40,002 - root - INFO - step:  3  loss: 12.1733  grad_norm:  0.6184  memory: 47.71GiB(34.12%)  tps: 51,674  tflops: 99.70  mfu: 10.08%
[rank0]:[titan] 2026-01-11 21:23:40,322 - root - INFO - step:  4  loss: 12.0984  grad_norm:  0.7449  memory: 47.71GiB(34.12%)  tps: 51,474  tflops: 99.31  mfu: 10.04%
[rank0]:[titan] 2026-01-11 21:23:40,641 - root - INFO - step:  5  loss: 11.9422  grad_norm:  1.0203  memory: 47.71GiB(34.12%)  tps: 51,528  tflops: 99.41  mfu: 10.05%
[rank0]:[titan] 2026-01-11 21:23:40,959 - root - INFO - step:  6  loss: 11.6608  grad_norm:  1.7049  memory: 47.71GiB(34.12%)  tps: 51,648  tflops: 99.64  mfu: 10.08%
[rank0]:[titan] 2026-01-11 21:23:41,279 - root - INFO - step:  7  loss: 11.1394  grad_norm:  1.8498  memory: 47.71GiB(34.12%)  tps: 51,401  tflops: 99.17  mfu: 10.03%
[rank0]:[titan] 2026-01-11 21:23:41,597 - root - INFO - step:  8  loss: 10.8308  grad_norm:  1.7437  memory: 47.71GiB(34.12%)  tps: 51,710  tflops: 99.76  mfu: 10.09%
[rank0]:[titan] 2026-01-11 21:23:41,916 - root - INFO - step:  9  loss: 10.6854  grad_norm:  1.5282  memory: 47.71GiB(34.12%)  tps: 51,499  tflops: 99.36  mfu: 10.05%
[rank0]:[titan] 2026-01-11 21:23:42,234 - root - INFO - step: 10  loss: 10.5298  grad_norm:  1.5869  memory: 47.71GiB(34.12%)  tps: 51,719  tflops: 99.78  mfu: 10.09%
```

### Analysis (no code changes)
- If `val_batch_size=2` in the ghost run (total batch size = 2 train + 2 val), then **total tokens per step match** the standard run. The slowdown still makes sense because ghost adds extra compute and synchronization even when total tokens are equal.
- If `val_batch_size` is larger (the config default is 4), then the ghost run also does **more** forward/backward work per step, which further increases end-to-end time. (Worth double-checking the effective `val_batch_size` passed at runtime.)
- Ghost adds substantial per-step overhead even at equal total batch size:
  - **Dot-product computation** for every supported layer (extra matmuls/reductions in `ghostEngines/supported_layers_grad_samplers_dotprod.py`).
  - **Saved-tensors hooks** for every saved activation (pack/unpack, masking clones) and norm-layer grad_input fixes.
  - **CPU transfers** inside `GradDotProdEngine.aggregate_and_log()` (dot products + X_train/Y_train moved to CPU each step, even if `save_train_batch=false`).
  - **Extra tensor ops** (`torch.cat` for train+val, activation masking clones) that increase memory bandwidth use.
- `time_metrics/end_to_end(s)` includes data loading + compute time; ghost’s additional compute + CPU sync points (dotprod aggregation) directly increase this metric.
- To isolate overhead, compare against a non-ghost run with **total batch size = train + val** (same token count) and verify the runtime `val_batch_size` used by ghost.
