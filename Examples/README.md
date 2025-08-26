# Examples

This folder contains small, runnable examples that demonstrate how to use the Ghost Engines in isolation from the full training stack.

## Files
- `ghost_mlp.py`: Minimal two‑layer MLP classification demo using `GradDotProdEngine`.
  - Trains for 10 steps on synthetic data (CPU by default).
  - Prints per‑parameter gradient dot‑products for each iteration and an aggregated vector.
  - Reports final validation loss at the end.

## Quick Start
- Install deps from repo root: `pip install -r requirements.txt`
- Run the demo: `python Examples/ghost_mlp.py`

Expected output (abridged):

```
[Iter 0] Per-parameter gradient dot products (val ⋅ train):
  fc1.weight           shape=(8,) values=[...]
  fc1.bias             shape=(8,) values=[...]
  fc2.weight           shape=(8,) values=[...]
  fc2.bias             shape=(8,) values=[...]
[Iter 0] Aggregated dot product across parameters: tensor([...])
...
Validation loss after 10 steps: 1.23...
```

## How It Works
- Builds a toy dataset and a two‑layer MLP.
- Concatenates train and validation batches, then runs a single forward/backward pass per step.
- `GradDotProdEngine`:
  - Computes per‑parameter gradient dot‑products between the fixed validation batch and each training sample (`grad_dot_prod`).
  - Accumulates training gradients separately, then moves them to `.grad` before the optimizer step.
  - Aggregates per‑parameter dot‑products into a single vector per step via `engine.aggregate_and_log()`.

Important: The script prints per‑parameter `grad_dot_prod` before calling `engine.aggregate_and_log()`, because aggregation clears those per‑parameter attributes to save memory.

## Tuning The Demo
- Steps: edit `steps = 10` to change the number of iterations.
- Batch sizes: change `n_train` / `n_val` near the top of the script.
- Learning rate: adjust the SGD `lr` in the optimizer definition.
- Device: set `device = "cuda"` if CUDA is available and desired.

## Extending
Use this pattern to instrument your own modules:
- Create your model and optimizer.
- Initialize `GradDotProdEngine(module=model, val_batch_size=..., ...)` and call `engine.attach(optimizer)`.
- Each iteration:
  - Run forward/backward on concatenated (train + val) batch.
  - Optionally inspect `param.grad_dot_prod` and/or `engine.aggregate_and_log()`.
  - Call `engine.prepare_gradients()`, `optimizer.step()`, then `engine.clear_gradients()`.

For full training on GPT models and Pile, see `main.py`, `training_loop.py`, and the SLURM launcher under `Scripts/`.
