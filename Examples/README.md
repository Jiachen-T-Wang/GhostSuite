# Examples

This folder contains small, runnable examples that demonstrate how to use the Ghost Engines in isolation from the full training stack.

## Files

### Gradient Dot Product Engine
- `ghost_mlp.py`: Minimal two‑layer MLP classification demo using `GradDotProdEngine`.
  - Trains for 10 steps on synthetic data (CPU by default).
  - Prints per‑parameter gradient dot‑products for each iteration and an aggregated vector.
  - Reports final validation loss at the end.

### Gradient Projection Engine (NEW)
- `ghost_gradproj_mlp.py`: MLP example with gradient projection using LoRA-style architecture.
  - Three execution modes:
    - `--mode project`: Compute and save projected gradients to disk
    - `--mode non_interf`: Verify engine doesn't interfere with training
    - `--mode naive_check`: Compare projections against naive computation
  - Demonstrates efficient per-sample gradient storage without materializing full gradients
  
- `ghost_gradproj_lm.py`: GPT-2 language model gradient projection example.
  - Projects gradients for transformer layers (attention, MLP)
  - Shows how to load and compute similarities from saved projections
  - Integrates with existing transformers_support utilities

## Quick Start

### Gradient Dot Product
```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python Examples/ghost_mlp.py
```

### Gradient Projection
```bash
# Run MLP projection example
python Examples/ghost_gradproj_mlp.py --mode project --proj_rank_total 64

# Test non-interference
python Examples/ghost_gradproj_mlp.py --mode non_interf

# Run GPT-2 projection
python Examples/ghost_gradproj_lm.py --proj_layers "attn.c_attn,mlp.c_fc" --proj_rank_total 256
```

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
