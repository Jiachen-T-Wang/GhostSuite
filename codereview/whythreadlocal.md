# Why threading.local() in _NamedSavedTensorManager

## Context
The refactor switched GhostDotProd activation capture to `torch.autograd.graph.saved_tensors_hooks`.
These hooks are global and do not indicate which module saved a tensor, so the implementation
uses a scope stack (push in forward-pre, pop in forward-post) to label saved tensors by the
module that is *currently* executing.

## Why not just use global storage
There are two separate needs:
- Cross-thread visibility of saved tensors (forward vs backward threads).
- Correct per-layer labeling during forward.

Global shared storage only solves the first need. It does not solve per-layer labeling,
and it is fragile when multiple forwards overlap: a single global scope stack would
interleave push/pop from different threads and label tensors under the wrong module.
That leads to wrong activation selection even if backward can see the tensors.

## Why thread-local for the scope stack
`threading.local()` is only used for the scope stack, not for the captured tensors:
- It isolates the "current forward" per thread to avoid interleaving when multiple forwards
  overlap in the same process.
- Without a thread-local stack, concurrent forwards could corrupt the module labeling and
  mis-associate saved tensors.

Examples of concurrent forward scenarios (not typical in a single-thread loop):
- `nn.DataParallel` or other threaded per-device forwards.
- Pipeline/micro-batch parallelism that overlaps forwards.
- `torch.jit.fork` or user-spawned threads calling the model concurrently.
- Re-entrant forward during backward (checkpointing / autograd recompute on worker threads).

In single-GPU, single-thread training, this behaves the same as a global stack.

## Why captured tensors are NOT thread-local
Autograd may run backward hooks on worker threads (CUDA). If the captured buffers were
thread-local, forward (pack hook) and backward would see different storage and activations
would be "missing." The refactor keeps only the scope stack thread-local; the captured
tensors and `used_ids` are shared and protected by a lock so backward threads can see them.

## Summary
The design intentionally splits responsibilities:
- Shared storage so backward threads can read forward-saved tensors.
- Thread-local scope stack so concurrent forwards do not mislabel tensors.

## Efficiency impact for single-GPU training
The overhead is minimal:
- `threading.local()` is just per-thread attribute access and list push/pop.
- The lock sees no contention in a single-thread run.
- The Python-level work is small relative to autograd compute.
