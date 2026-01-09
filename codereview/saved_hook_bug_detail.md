# Saved tensor hook failure (TorchTitan) – detailed review

## What broke
- `./tests/torchtitan/run_train_with_ghost.sh` failed during the first backward pass with `RuntimeError: Failed to capture saved activations...` originating from `ghostEngines/autograd_grad_sample_dotprod.py`.
- A minimal CUDA repro (TwoLayerMLP using `GradDotProdEngine` on GPU) produced the same error, while the CPU-only `Examples/ghost_mlp.py` ran successfully.

## Root cause
- `_NamedSavedTensorManager` kept all capture buffers (`captured`, `captured_all`, `used_ids`) in `threading.local()` state. On CUDA, autograd executes module backward hooks on worker threads, so the forward-side pack hook populated the main thread’s buffers, but the backward hook read a different, empty thread-local store → missing activations and the runtime error.
- Because `used_ids` was also thread-local, even if tensors were visible they could be re-consumed on different autograd threads, risking mis-association of activations when backward executes in parallel.
- The CPU example worked only because autograd stayed single-threaded, so forward and backward shared the same thread-local storage; TorchTitan’s GPU run exercised multi-threaded backward.

## Fix implemented
- Refactored `_NamedSavedTensorManager` to keep capture buffers and `used_ids` in shared state guarded by a lock, while keeping only the scope stack thread-local (`ghostEngines/autograd_grad_sample_dotprod.py:25-165`). `enable/disable` now reset the shared buffers so backward hooks on worker threads see the tensors saved during forward, and activation reuse is coordinated across threads.
- The selection logic in `resolve_activation` still prefers exact input-shape matches (and flatted shape) and marks tensors consumed under the shared lock to prevent double usage across threads.

## Validation
- Minimal CUDA repro (TwoLayerMLP + `GradDotProdEngine` on GPU inside `saved_tensors_context`) now completes without error and reports loss.
- `Examples/ghost_mlp.py` still runs end-to-end on CPU after the refactor.
- (Recommended follow-up) Re-run `./tests/torchtitan/run_train_with_ghost.sh --training.steps=1` to verify the original failure path; expect the saved-activation lookup to succeed now that backward threads share the captured tensors.
